"""任务记录：配置、中途变更、结果和状态。

默认不保存网络权重。初版用一份 SQLite 索引，加上每次任务一个目录即可。
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def json_default(obj):
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot JSON-encode {type(obj)!r}")


def dump_json(path: Path, value) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    tmp.replace(path)


def new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


def default_runs_root() -> Path:
    return Path(__file__).resolve().parent.parent / "runs"


class RunRecorder:
    def __init__(self, output_dir: Path | None = None, run_id: str | None = None, parent_id: str | None = None):
        self.run_id = run_id or new_run_id()
        self.dir = Path(output_dir) if output_dir is not None else default_runs_root() / self.run_id
        if self.dir.exists() and any(self.dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {self.dir}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.parent_id = parent_id
        self.changes = []
        self.training_path = self.dir / "training.csv"
        self.metrics_path = self.dir / "metrics.csv"
        self._train_writer = None
        self._metric_writer = None
        self._train_file = None
        self._metric_file = None
        self.index_path = default_runs_root() / "runs.sqlite"
        self.status = "running"
        dump_json(self.dir / "status.json", {"run_id": self.run_id, "status": self.status, "parent_id": parent_id})

    def software_versions(self, device) -> dict:
        versions = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        }
        try:
            import optuna
            versions["optuna"] = optuna.__version__
        except ImportError:
            versions["optuna"] = None
        return versions

    def write_config(self, payload: dict) -> None:
        dump_json(self.dir / "config.json", payload)

    def write_circuit(self, payload: dict) -> None:
        dump_json(self.dir / "circuit.json", payload)

    def log_change(self, step: int, key: str, old, new, reason: str) -> None:
        row = {"step": step, "key": key, "old": old, "new": new, "reason": reason,
               "time": datetime.now(timezone.utc).isoformat()}
        self.changes.append(row)
        with (self.dir / "changes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=json_default) + "\n")

    def _ensure_train(self, row: dict):
        if self._train_writer is None:
            self._train_file = self.training_path.open("w", newline="", encoding="utf-8")
            self._train_writer = csv.DictWriter(self._train_file, list(row.keys()))
            self._train_writer.writeheader()
        self._train_writer.writerow(row)
        self._train_file.flush()

    def _ensure_metrics(self, row: dict):
        if self._metric_writer is None:
            self._metric_file = self.metrics_path.open("w", newline="", encoding="utf-8")
            self._metric_writer = csv.DictWriter(self._metric_file, list(row.keys()))
            self._metric_writer.writeheader()
        self._metric_writer.writerow({k: row.get(k, "") for k in self._metric_writer.fieldnames})
        self._metric_file.flush()

    def log_training(self, row: dict) -> None:
        self._ensure_train({k: "" if v is None else v for k, v in row.items()})

    def log_step(self, row: dict) -> None:
        self._ensure_metrics({k: "" if v is None else v for k, v in row.items()})

    def save_state(self, model, step: int, update=None) -> Path:
        path = self.dir / ("model_final.pt" if update is None else f"step_{step:03d}.pt")
        payload = {
            "model_config": model.config,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "step": step,
            "update": None if update is None else update.to_dict(),
        }
        torch.save(payload, path)
        return path

    def finish(self, result: dict, status: str = "succeeded") -> None:
        self.status = status
        dump_json(self.dir / "result.json", result)
        dump_json(self.dir / "changes.json", self.changes)
        dump_json(self.dir / "status.json", {
            "run_id": self.run_id, "status": status, "parent_id": self.parent_id,
        })
        self.close()
        self._index_run(result, status)

    def fail(self, message: str, result: dict | None = None) -> None:
        payload = dict(result or {})
        payload["error"] = message
        self.finish(payload, status="failed")

    def close(self) -> None:
        for handle in (self._train_file, self._metric_file):
            if handle is not None:
                handle.close()
        self._train_file = self._metric_file = None
        self._train_writer = self._metric_writer = None

    def _index_run(self, result: dict, status: str) -> None:
        metrics = result.get("metrics", {})
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.index_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    created_at TEXT,
                    status TEXT,
                    n_qubits INTEGER,
                    n_updates INTEGER,
                    shots INTEGER,
                    seed INTEGER,
                    total_infidelity REAL,
                    train_seconds REAL,
                    output_dir TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS trials (
                    run_id TEXT,
                    trial_number INTEGER,
                    params_json TEXT,
                    score REAL,
                    seconds REAL,
                    status TEXT
                )"""
            )
            conn.execute(
                """INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    self.run_id,
                    self.parent_id,
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    metrics.get("n_qubits"),
                    metrics.get("n_updates"),
                    metrics.get("shots"),
                    metrics.get("seed"),
                    metrics.get("total_infidelity"),
                    metrics.get("train_seconds"),
                    str(self.dir),
                ),
            )
            conn.commit()
