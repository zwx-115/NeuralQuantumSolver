from __future__ import annotations

from pathlib import Path
import platform
import torch

from .models import NeuralQuantumState


def save_checkpoint(
    path: str | Path,
    model: NeuralQuantumState,
    *,
    model_config: dict,
    run_config: dict,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "run_config": run_config,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "step": step,
            "metadata": {
                "torch_version": str(torch.__version__),
                "python_version": platform.python_version(),
            },
        },
        Path(path),
    )


def load_checkpoint(
    path: str | Path,
    model: NeuralQuantumState,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
) -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
