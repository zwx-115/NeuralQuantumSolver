"""初版模拟器的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .circuits import Circuit, make_bell, make_hczch
from .config import Config, TuningConfig
from .engine import simulate
from .records import json_default


def _load_circuit(name: str, n_qubits: int) -> Circuit:
    path = Path(name)
    if path.suffix == ".json" and path.exists():
        return Circuit.from_dict(json.loads(path.read_text(encoding="utf-8")))
    key = name.lower()
    if key == "hczch":
        return make_hczch(n_qubits)
    if key == "bell":
        return make_bell(n_qubits)
    raise ValueError("Use 'hczch', 'bell', or a circuit JSON file")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NQS 量子线路模拟器（初版）")
    parser.add_argument("--circuit", default="hczch", help="hczch、bell，或 circuit.json 路径")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20, help="每个门更新的 Adam 迭代次数")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--observables", default="Z0,X0", help="逗号分隔的 Pauli 标记")
    parser.add_argument("--bitstrings", default="", help="逗号分隔的计算基 bitstring")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args(argv)

    circuit = _load_circuit(args.circuit, args.n_qubits)
    cfg = Config()
    cfg.train_steps_per_update = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.save_state = args.save_state
    observables = [item.strip() for item in args.observables.split(",") if item.strip()]
    bitstrings = [item.strip() for item in args.bitstrings.split(",") if item.strip()]
    tuning = False
    if args.tune:
        tuning = TuningConfig(n_trials=args.n_trials, timeout=args.timeout, seed=cfg.seed)
    result = simulate(
        circuit, config=cfg, observables=observables or None, shots=args.shots,
        tune=tuning, bitstrings=bitstrings or None, output_dir=args.output,
        save_state=args.save_state,
    )
    summary = {
        "run_id": result.run_id,
        "status": result.status,
        "output_dir": result.output_dir,
        "counts_preview": dict(list(result.counts.items())[:8]),
        "requested_amplitudes": result.requested_amplitudes,
        "expectations": result.expectations,
        "metrics": {k: result.metrics.get(k) for k in (
            "n_qubits", "n_updates", "train_seconds", "total_infidelity",
            "final_loss", "sample_check", "fidelity_note", "device")},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
