"""冻结教师的逐步训练，以及对外的 simulate() 入口。"""

from __future__ import annotations

import copy
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from .circuits import Circuit, apply_update, initial_state_vector, plan_updates
from .config import Config, TuningConfig
from .evaluate import evaluate_step, exact_evolve, infidelity, neural_vector, sample_check
from .losses import diagonal_loss, local_loss, local_teacher, mixed_samples
from .model import Statebank, bits_to_strings, expected_parameter_count, strings_to_bits
from .observables import estimate_observables, exact_observables
from .records import RunRecorder


@dataclass
class SimulationResult:
    run_id: str
    counts: dict
    requested_amplitudes: dict
    expectations: dict
    metrics: dict
    status: str
    output_dir: str
    updates: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "counts": self.counts,
            "requested_amplitudes": self.requested_amplitudes,
            "expectations": self.expectations,
            "metrics": self.metrics,
            "status": self.status,
            "output_dir": self.output_dir,
            "updates": self.updates,
        }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def select_device(cfg: Config) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _learning_rate(cfg: Config, update) -> float:
    if not update.diagonal and cfg.local_learning_rate is not None:
        return cfg.local_learning_rate
    return cfg.learning_rate


def _scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach())
    return float(value)


def simulate(
    circuit,
    config: Config | None = None,
    observables=None,
    shots: int = 10_000,
    tune: bool | TuningConfig = False,
    bitstrings=None,
    output_dir=None,
    save_state: bool | None = None,
    evaluate_exact: bool | None = None,
    parent_id: str | None = None,
    run_id: str | None = None,
) -> SimulationResult:
    """用自回归 NQS 近似一条无噪声线路。

    返回 counts、指定振幅、采样期望值、指标和 run_id。
    精确 2^N 态矢量只用于评估，不作为训练目标。
    """
    cfg = copy.deepcopy(config) if config is not None else Config()
    if tune:
        from .tuning import run_study
        tuning = tune if isinstance(tune, TuningConfig) else TuningConfig()
        return run_study(
            circuit, cfg, observables=observables, shots=shots, bitstrings=bitstrings,
            output_dir=output_dir, save_state=save_state, evaluate_exact=evaluate_exact,
            tuning=tuning,
        )
    return _simulate_once(
        circuit, cfg, observables=observables, shots=shots, bitstrings=bitstrings,
        output_dir=output_dir, save_state=save_state, evaluate_exact=evaluate_exact,
        parent_id=parent_id, run_id=run_id,
    )


def _simulate_once(
    circuit,
    cfg: Config,
    observables=None,
    shots: int = 10_000,
    bitstrings=None,
    output_dir=None,
    save_state: bool | None = None,
    evaluate_exact: bool | None = None,
    parent_id: str | None = None,
    run_id: str | None = None,
) -> SimulationResult:
    circuit = Circuit.from_dict(circuit) if not isinstance(circuit, Circuit) else circuit
    keep_weights = cfg.save_state if save_state is None else bool(save_state)
    recorder = RunRecorder(output_dir=output_dir, run_id=run_id, parent_id=parent_id)
    result = SimulationResult(run_id=recorder.run_id, counts={}, requested_amplitudes={},
                              expectations={}, metrics={}, status="running",
                              output_dir=str(recorder.dir))
    try:
        result = _run_training(
            circuit, cfg, observables, shots, bitstrings, keep_weights,
            evaluate_exact, recorder,
        )
        recorder.finish(result.to_dict(), status=result.status)
        return result
    except Exception as exc:
        result.status = "failed"
        result.metrics = dict(result.metrics or {})
        result.metrics["error"] = str(exc)
        recorder.fail(str(exc), result.to_dict())
        raise


def _run_training(circuit, cfg, observables, shots, bitstrings, keep_weights, evaluate_exact, recorder):
    seed_everything(cfg.seed)
    device = select_device(cfg)
    updates, plan_note = plan_updates(circuit)
    n = circuit.n_qubits
    do_exact = n <= cfg.exact_max_qubits if evaluate_exact is None else bool(evaluate_exact)
    if do_exact and n > cfg.exact_max_qubits:
        raise ValueError("exact evaluation requested above exact_max_qubits")

    model = Statebank(**cfg.model_config(n)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    expected = expected_parameter_count(n)
    if parameters != expected:
        raise AssertionError(f"Parameter mismatch: {parameters} != {expected}")
    initial = plan_note.get("effective_initial_state", circuit.initial_state)
    if initial == "plus":
        model.initialize_plus()
    else:
        model.initialize_zero()

    recorder.write_circuit(circuit.to_dict())
    recorder.write_config({
        **cfg.to_dict(),
        **recorder.software_versions(device),
        "n_qubits": n,
        "shots": shots,
        "save_state": keep_weights,
        "evaluate_exact": do_exact,
        "run_id": recorder.run_id,
        "parent_id": recorder.parent_id,
        "plan": plan_note,
        "actual_schedule": [update.to_dict() for update in updates],
        "requested_bitstrings": list(bitstrings or []),
        "observables": observables,
    })
    if plan_note.get("skipped_initial_hadamards"):
        recorder.log_change(0, "initial_state", circuit.initial_state, "plus",
                            "开头全体 H 由 plus 输出头精确表示")

    exact = None
    remaining_for_exact = None
    if do_exact:
        exact, _ = initial_state_vector(circuit)

    step_rows = []
    total_train = 0.0
    last_loss = None
    wall_start = time.perf_counter()

    for step, update in enumerate(updates, 1):
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        model.train()
        lr = _learning_rate(cfg, update)
        if lr != cfg.learning_rate:
            recorder.log_change(step, "learning_rate", cfg.learning_rate, lr,
                                "非对角更新使用 local_learning_rate")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps, weight_decay=cfg.weight_decay,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sync(device)
        start = time.perf_counter()
        components = {}
        loss = None
        for iteration in range(1, cfg.train_steps_per_update + 1):
            bits = mixed_samples(teacher, cfg.batch_size, cfg.uniform_mix_eta)
            if update.diagonal:
                loss, components = diagonal_loss(model, teacher, bits, update.gates)
            else:
                configurations, target = local_teacher(teacher, bits, update.gates[0])
                loss, components = local_loss(model, configurations, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
            if iteration == 1 or iteration % cfg.log_every == 0 or iteration == cfg.train_steps_per_update:
                recorder.log_training({
                    "step": step, "physical_layer": update.physical_layer, "gate": update.name,
                    "learning_rate": lr, "iteration": iteration, "loss": _scalar(loss),
                    "magnitude": _scalar(components.get("magnitude")),
                    "kl": _scalar(components.get("kl")),
                    "phase": _scalar(components.get("phase")),
                    "mass": _scalar(components.get("mass")),
                    "overlap": _scalar(components.get("overlap")),
                    "grad_norm": _scalar(norm),
                })
        sync(device)
        train_seconds = time.perf_counter() - start
        total_train += train_seconds
        last_loss = _scalar(loss)
        model.eval()

        row = {
            "step": step, "physical_layer": update.physical_layer, "gate": update.name,
            "diagonal": update.diagonal, "learning_rate": lr,
            "iterations": cfg.train_steps_per_update, "final_loss": last_loss,
            "train_seconds": train_seconds,
            "step_infidelity": None, "total_infidelity": None, "normalization": None,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        }
        if do_exact:
            exact_after = apply_update(exact, update, n)
            eval_metrics = evaluate_step(model, teacher, update, exact, exact_after, cfg.eval_batch_size)
            row.update(eval_metrics)
            exact = exact_after
        recorder.log_step(row)
        step_rows.append(row)
        inf_step = row["step_infidelity"]
        inf_total = row["total_infidelity"]
        print(
            f"{step:02d}/{len(updates)} {update.name:20s} "
            f"loss={last_loss:.6g} "
            f"step={inf_step if inf_step is None else f'{inf_step:.6g}'} "
            f"total={inf_total if inf_total is None else f'{inf_total:.6g}'} "
            f"train={train_seconds:.1f}s",
            flush=True,
        )
        del teacher, optimizer

    model.eval()
    with torch.no_grad():
        samples = None
        counts = {}
        if shots:
            samples = model.sample(int(shots))
            counts = dict(Counter(bits_to_strings(samples)))
        requested = {}
        if bitstrings:
            query = strings_to_bits(list(bitstrings), n, device=device)
            amps = model.amplitude(query)
            for text, value in zip(bitstrings, amps):
                requested[str(text)] = {"real": float(value.real), "imag": float(value.imag)}
        expectations = (
            estimate_observables(model, observables, samples)
            if observables and samples is not None else {}
        )

    metrics = {
        "n_qubits": n,
        "n_updates": len(updates),
        "shots": int(shots),
        "seed": cfg.seed,
        "parameters": parameters,
        "device": str(device),
        "train_seconds": total_train,
        "wall_seconds": time.perf_counter() - wall_start,
        "final_loss": last_loss,
        "total_infidelity": step_rows[-1]["total_infidelity"] if step_rows else (0.0 if do_exact else None),
        "step_infidelity": step_rows[-1]["step_infidelity"] if step_rows else None,
        "normalization": step_rows[-1]["normalization"] if step_rows else None,
        "evaluate_exact": do_exact,
        "save_state": keep_weights,
        "effective_initial_state": initial,
        "plan": plan_note,
    }
    if not updates and do_exact:
        current = neural_vector(model, cfg.eval_batch_size)
        metrics["total_infidelity"] = infidelity(current, exact if exact is not None else exact_evolve(circuit))
        metrics["normalization"] = float(current.abs().square().sum())

    if do_exact:
        final_exact = exact if updates else exact_evolve(circuit)
        current = neural_vector(model, cfg.eval_batch_size)
        metrics["total_infidelity"] = infidelity(current, final_exact)
        metrics["normalization"] = float(current.abs().square().sum())
        if observables:
            exact_obs = exact_observables(final_exact, n, observables)
            for name, payload in expectations.items():
                payload["exact"] = exact_obs.get(name, {}).get("value")
                payload["exact_imag"] = exact_obs.get(name, {}).get("imag")
                if payload["exact"] is not None:
                    payload["abs_error"] = abs(payload["value"] - payload["exact"])
        if n <= cfg.exact_max_qubits and shots:
            metrics["sample_check"] = sample_check(model, min(shots, 20000), current.abs().square())
        metrics["fidelity_note"] = "独立精确重叠；未用于训练"
    else:
        metrics["total_infidelity"] = None
        metrics["fidelity_note"] = (
            "该宽度无可靠精确参考；训练 loss 仅作诊断，不能当作真实保真度"
        )

    if keep_weights:
        recorder.save_state(model, len(updates), None)
        metrics["model_file"] = "model_final.pt"
    else:
        metrics["model_file"] = None

    return SimulationResult(
        run_id=recorder.run_id,
        counts=counts,
        requested_amplitudes=requested,
        expectations=expectations,
        metrics=metrics,
        status="succeeded",
        output_dir=str(recorder.dir),
        updates=step_rows,
    )
