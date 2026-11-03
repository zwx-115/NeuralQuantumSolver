"""可选的 Optuna 搜索，只调少数全局训练超参数。

每个 trial 都走与固定配置相同的 _simulate_once 流程。
"""

from __future__ import annotations

import copy
from pathlib import Path

from .config import Config, TuningConfig
from .engine import SimulationResult, _simulate_once
from .records import RunRecorder, dump_json, new_run_id


def _score(result: SimulationResult) -> float:
    infidelity = result.metrics.get("total_infidelity")
    if infidelity is not None:
        return float(infidelity)
    loss = result.metrics.get("final_loss")
    if loss is None:
        return float("inf")
    return float(loss)


def run_study(
    circuit,
    config: Config,
    observables=None,
    shots: int = 10_000,
    bitstrings=None,
    output_dir=None,
    save_state: bool | None = None,
    evaluate_exact: bool | None = None,
    tuning: TuningConfig | None = None,
) -> SimulationResult:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("tune=True requires optuna; install it in the same environment") from exc

    tuning = tuning or TuningConfig()
    parent_id = new_run_id() if output_dir is None else Path(output_dir).name
    recorder = RunRecorder(output_dir=output_dir, run_id=parent_id)
    recorder.write_config({
        **config.to_dict(),
        "tuning": tuning.to_dict(),
        "shots": shots,
        "save_state": bool(config.save_state if save_state is None else save_state),
        "evaluate_exact": evaluate_exact,
        "mode": "optuna_global_hyperparameters",
        "score": "有精确参考时用累计 infidelity，否则用最终训练 loss（仅诊断）",
    })
    from .circuits import Circuit
    circuit_obj = Circuit.from_dict(circuit) if not isinstance(circuit, Circuit) else circuit
    recorder.write_circuit(circuit_obj.to_dict())

    history = []
    best_result = None
    best_score = float("inf")

    def objective(trial):
        nonlocal best_result, best_score
        cfg = copy.deepcopy(config)
        if tuning.search_learning_rate:
            cfg.learning_rate = trial.suggest_float(
                "learning_rate", tuning.lr_low, tuning.lr_high, log=True)
        if tuning.search_eta:
            cfg.uniform_mix_eta = trial.suggest_float(
                "uniform_mix_eta", tuning.eta_low, tuning.eta_high)
        if tuning.search_steps:
            cfg.train_steps_per_update = trial.suggest_categorical(
                "train_steps_per_update", list(tuning.steps_choices))
        trial_dir = recorder.dir / f"trial_{trial.number:03d}"
        try:
            result = _simulate_once(
                circuit_obj, cfg, observables=observables, shots=shots, bitstrings=bitstrings,
                output_dir=trial_dir, save_state=save_state, evaluate_exact=evaluate_exact,
                parent_id=recorder.run_id, run_id=trial_dir.name,
            )
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")
        score = _score(result)
        row = {
            "trial": trial.number,
            "score": score,
            "score_name": "total_infidelity" if result.metrics.get("total_infidelity") is not None else "final_loss",
            "params": trial.params,
            "run_id": result.run_id,
            "train_seconds": result.metrics.get("train_seconds"),
            "status": result.status,
        }
        history.append(row)
        if score < best_score:
            best_score = score
            best_result = result
        return score

    sampler = optuna.samplers.TPESampler(seed=tuning.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=recorder.run_id)
    study.optimize(objective, n_trials=tuning.n_trials, timeout=tuning.timeout)

    if best_result is None:
        recorder.fail("no successful Optuna trial")
        raise RuntimeError("Optuna produced no successful trial")

    payload = best_result.to_dict()
    payload["run_id"] = recorder.run_id
    payload["output_dir"] = str(recorder.dir)
    payload["metrics"] = dict(best_result.metrics)
    payload["metrics"]["tuning"] = {
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "best_score": float(study.best_value),
        "n_trials": len(study.trials),
        "history": history,
        "best_child_run": best_result.run_id,
        "best_child_dir": best_result.output_dir,
    }
    dump_json(recorder.dir / "tuning.json", payload["metrics"]["tuning"])
    result = SimulationResult(
        run_id=recorder.run_id,
        counts=best_result.counts,
        requested_amplitudes=best_result.requested_amplitudes,
        expectations=best_result.expectations,
        metrics=payload["metrics"],
        status="succeeded",
        output_dir=str(recorder.dir),
        updates=best_result.updates,
    )
    recorder.finish(result.to_dict(), status="succeeded")
    return result
