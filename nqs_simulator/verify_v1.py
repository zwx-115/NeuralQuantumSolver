"""初版端到端验证。报告真实数值，不伪造成功。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .checks import run_checks
from .circuits import Circuit, Gate, make_bell, make_hczch
from .config import Config, TuningConfig
from .engine import simulate
from .records import dump_json, json_default


def _cfg(**overrides) -> Config:
    cfg = Config()
    cfg.batch_size = 64
    cfg.log_every = 50
    cfg.train_steps_per_update = 20
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _weight_files(directory: Path) -> list[str]:
    return [path.name for path in Path(directory).rglob("*.pt")]


def _summary(result, directory: Path, extra=None) -> dict:
    payload = {
        "run_id": result.run_id,
        "status": result.status,
        "n_updates": result.metrics.get("n_updates"),
        "total_infidelity": result.metrics.get("total_infidelity"),
        "normalization": result.metrics.get("normalization"),
        "final_loss": result.metrics.get("final_loss"),
        "train_seconds": result.metrics.get("train_seconds"),
        "sample_check": result.metrics.get("sample_check"),
        "counts_top": dict(sorted(result.counts.items(), key=lambda kv: -kv[1])[:6]),
        "amplitudes": result.requested_amplitudes,
        "expectations": {
            name: {k: v.get(k) for k in ("value", "std_error", "imag", "exact", "abs_error")}
            for name, v in result.expectations.items()
        },
        "weight_files": _weight_files(directory),
        "has_config": (directory / "config.json").exists(),
        "has_result": (directory / "result.json").exists(),
        "has_changes": (directory / "changes.json").exists(),
    }
    if extra:
        payload.update(extra)
    return payload


def t_layer(n: int) -> Circuit:
    return Circuit(n, tuple(Gate("T", (q,), 1) for q in range(n)), "plus")


def cz_circuit() -> Circuit:
    return Circuit(2, (Gate("CZ", (0, 1), 1),), "plus")


def main() -> dict:
    root = Path(__file__).resolve().parent.parent / "runs" / f"verify_v1_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=False)
    report = {"output_root": str(root), "checks": run_checks()}

    plus_t = simulate(
        t_layer(4),
        config=_cfg(train_steps_per_update=120, seed=42),
        observables=["Z0", "X0"],
        shots=2000,
        bitstrings=["0000", "1111"],
        output_dir=root / "plus_t_n4",
    )
    report["plus_t_n4"] = _summary(plus_t, root / "plus_t_n4")

    plus_cz = simulate(
        cz_circuit(),
        config=_cfg(train_steps_per_update=120, seed=42),
        observables=["Z0Z1"],
        shots=2000,
        bitstrings=["00", "11"],
        output_dir=root / "plus_cz_n2",
    )
    report["plus_cz_n2"] = _summary(plus_cz, root / "plus_cz_n2")

    bell = simulate(
        make_bell(2),
        config=_cfg(train_steps_per_update=80, seed=42),
        observables=["Z0", "Z1", "Z0Z1", "X0X1"],
        shots=2000,
        bitstrings=["00", "11", "01"],
        output_dir=root / "bell_n2",
    )
    report["bell_n2"] = _summary(bell, root / "bell_n2")

    hczch = simulate(
        make_hczch(4),
        config=_cfg(train_steps_per_update=50, seed=42),
        observables=["Z0", "X0", "Z0Z2"],
        shots=1000,
        bitstrings=["0000", "1111"],
        output_dir=root / "hczch_n4",
    )
    report["hczch_n4"] = _summary(
        hczch, root / "hczch_n4",
        extra={"step_infidelities": [row.get("total_infidelity") for row in hczch.updates]},
    )

    tuned = simulate(
        cz_circuit(),
        config=_cfg(train_steps_per_update=40, seed=0),
        observables=["Z0Z1"],
        shots=500,
        tune=TuningConfig(n_trials=2, seed=0, search_eta=False, search_steps=False,
                          lr_low=1e-4, lr_high=1e-3),
        output_dir=root / "optuna_cz_n2",
    )
    report["optuna_cz_n2"] = _summary(
        tuned, root / "optuna_cz_n2",
        extra={
            "best_score": tuned.metrics.get("tuning", {}).get("best_score"),
            "best_params": tuned.metrics.get("tuning", {}).get("best_params"),
            "history": tuned.metrics.get("tuning", {}).get("history"),
        },
    )

    report["default_skips_weights"] = all(
        not report[name]["weight_files"]
        for name in ("plus_t_n4", "plus_cz_n2", "bell_n2", "hczch_n4", "optuna_cz_n2")
    )
    dump_json(root / "summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=json_default))
    return report


if __name__ == "__main__":
    main()
