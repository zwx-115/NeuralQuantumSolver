"""无额外依赖的轻量级可复现诊断结果写入器。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import torch


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def save_diagnostic_artifacts(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    checkpoints: dict[str, dict[str, torch.Tensor]] | None = None,
) -> Path:
    """写入配置、标量结果行、摘要和可选的 CPU checkpoint。"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text(json.dumps(_json_value(config), indent=2, sort_keys=True))
    (destination / "summary.json").write_text(json.dumps(_json_value(summary), indent=2, sort_keys=True))
    fields = sorted({key for row in rows for key in row})
    with (destination / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _json_value(value) for key, value in row.items()} for row in rows])
    if checkpoints:
        torch.save(
            {name: {key: value.detach().cpu() for key, value in state.items()} for name, state in checkpoints.items()},
            destination / "checkpoints.pt",
        )
    return destination
