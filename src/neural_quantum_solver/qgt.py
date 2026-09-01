from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import torch

from .devices import synchronize_devices


@dataclass(frozen=True)
class QGTDiagnostics:
    eigenvalues: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: int
    condition_number: float
    regularization: float
    gradient_norm: float
    update_norm: float
    fs_step_norm: float
    solve_seconds: float


def quantum_geometric_tensor(
    log_derivatives: torch.Tensor, probabilities: torch.Tensor
) -> torch.Tensor:
    """Compute S_ij=<O_i* O_j>-<O_i*><O_j>."""
    probabilities = probabilities / probabilities.sum()
    mean = torch.sum(probabilities[:, None] * log_derivatives, dim=0)
    centered = log_derivatives - mean
    return (centered.conj().mT * probabilities) @ centered


def solve_sr(
    qgt: torch.Tensor, force: torch.Tensor, *, learning_rate: float,
    regularization: float, rcond: float = 1e-12,
    solver_device: torch.device | str | None = "auto",
) -> tuple[torch.Tensor, QGTDiagnostics]:
    original_device = qgt.device
    if solver_device == "auto":
        work_device = qgt.device
    elif solver_device is None:
        work_device = qgt.device
    else:
        work_device = torch.device(solver_device)
    qgt = qgt.to(work_device)
    force = force.to(work_device)
    shifted = qgt + regularization * torch.eye(qgt.shape[0], dtype=qgt.dtype, device=qgt.device)
    synchronize_devices((work_device,))
    solve_started = perf_counter()
    update = torch.linalg.solve(shifted, -learning_rate * force)
    synchronize_devices((work_device,))
    solve_seconds = perf_counter() - solve_started
    eigenvalues = torch.linalg.eigvalsh(qgt).real
    singular_values = torch.linalg.svdvals(qgt).real
    threshold = rcond * singular_values.max()
    kept = singular_values[singular_values > threshold]
    condition = float((kept.max() / kept.min()).item()) if kept.numel() else float("inf")
    qgt_update = (qgt @ update[:, None]).squeeze(-1)
    fs_squared = torch.vdot(update, qgt_update).real.clamp_min(0)
    diagnostics = QGTDiagnostics(
        eigenvalues, singular_values, int(kept.numel()), condition, regularization,
        float(torch.linalg.vector_norm(force).item()),
        float(torch.linalg.vector_norm(update).item()), float(torch.sqrt(fs_squared).item()),
        solve_seconds,
    )
    return update.to(original_device), diagnostics
