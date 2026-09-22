from __future__ import annotations

from dataclasses import dataclass
import torch


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
) -> tuple[torch.Tensor, QGTDiagnostics]:
    shifted = qgt + regularization * torch.eye(qgt.shape[0], dtype=qgt.dtype, device=qgt.device)
    update = -learning_rate * (torch.linalg.pinv(shifted, rtol=rcond) @ force)
    eigenvalues = torch.linalg.eigvalsh(qgt).real
    singular_values = torch.linalg.svdvals(qgt).real
    threshold = rcond * singular_values.max()
    kept = singular_values[singular_values > threshold]
    condition = float((kept.max() / kept.min()).item()) if kept.numel() else float("inf")
    fs_squared = torch.vdot(update, qgt @ update).real.clamp_min(0)
    diagnostics = QGTDiagnostics(
        eigenvalues, singular_values, int(kept.numel()), condition, regularization,
        float(torch.linalg.vector_norm(force).item()),
        float(torch.linalg.vector_norm(update).item()), float(torch.sqrt(fs_squared).item()),
    )
    return update, diagnostics
