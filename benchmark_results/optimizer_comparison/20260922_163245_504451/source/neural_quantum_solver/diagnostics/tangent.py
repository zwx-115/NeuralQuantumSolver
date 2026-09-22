"""薛定谔演化方向的精确切空间残差。"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ..derivatives import LogJacobian
from ..estimators import exact_state
from ..models import NeuralQuantumState
from ..systems import PhysicalSystem


@dataclass(frozen=True)
class TangentResidual:
    """将 ``-i H|psi>`` 投影到 NQS 切空间的结果。

    记录的归一化残差为 ``1 - F^H S^+ F / Var(H)``。通过显式构造
    Hilbert 空间 Jacobian，``least_squares_residual`` 独立得到同一量。
    """

    normalized_residual: torch.Tensor
    absolute_residual: torch.Tensor
    variance: torch.Tensor
    captured_variance: torch.Tensor
    force: torch.Tensor
    qgt: torch.Tensor
    least_squares_residual: torch.Tensor
    pseudoinverse_threshold: torch.Tensor
    defined: bool


def tangent_space_residual(
    model: NeuralQuantumState,
    system: PhysicalSystem,
    *,
    jacobian: LogJacobian | None = None,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> TangentResidual:
    """计算精确求和切空间残差，并进行显式最小二乘交叉验证。"""
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    state = exact_state(model, system)
    psi = state.amplitudes
    derivatives = (jacobian or LogJacobian())(model, state.configurations)
    mean_derivative = torch.sum(state.probabilities[:, None] * derivatives, dim=0)
    tangent_jacobian = psi[:, None] * (derivatives - mean_derivative)
    hamiltonian = system.hamiltonian.dense_matrix(
        system.hilbert, dtype=psi.dtype, device=psi.device
    )
    energy = torch.vdot(psi, hamiltonian @ psi).real
    centered_direction = hamiltonian @ psi - energy * psi
    variance = torch.vdot(centered_direction, centered_direction).real
    qgt = tangent_jacobian.mH @ tangent_jacobian
    force = tangent_jacobian.mH @ centered_direction
    eigenvalues, eigenvectors = torch.linalg.eigh(qgt)
    nonnegative = eigenvalues.real.clamp_min(0)
    maximum = nonnegative.max() if nonnegative.numel() else torch.zeros((), device=psi.device)
    threshold = torch.maximum(
        torch.as_tensor(atol, dtype=psi.real.dtype, device=psi.device), maximum * rtol
    )
    inverse = torch.where(nonnegative > threshold, nonnegative.reciprocal(), torch.zeros_like(nonnegative))
    pseudoinverse = (eigenvectors * inverse) @ eigenvectors.mH
    captured = torch.vdot(force, pseudoinverse @ force).real.clamp_min(0)
    absolute = (variance - captured).clamp_min(0)
    epsilon = torch.finfo(psi.real.dtype).eps
    defined = bool(variance > epsilon)
    normalized = absolute / variance if defined else torch.full_like(variance, float("nan"))

    # 显式 Jacobian 的 SVD 提供独立的 Hilbert 空间最小二乘计算。保留与
    # S^+ 完全相同的奇异子空间，因为 S 的本征值是 Jacobian 奇异值的平方。
    left, singular, right_h = torch.linalg.svd(tangent_jacobian, full_matrices=False)
    retained_singular = singular.square() > threshold
    inverse_singular = torch.where(
        retained_singular, singular.reciprocal(), torch.zeros_like(singular)
    )
    coefficients = right_h.mH @ (inverse_singular * (left.mH @ (-1j * centered_direction)))
    ls_absolute = torch.linalg.vector_norm(tangent_jacobian @ coefficients + 1j * centered_direction).square().real
    ls_normalized = ls_absolute / variance if defined else torch.full_like(variance, float("nan"))
    return TangentResidual(
        normalized_residual=normalized.detach(),
        absolute_residual=absolute.detach(),
        variance=variance.detach(),
        captured_variance=captured.detach(),
        force=force.detach(),
        qgt=qgt.detach(),
        least_squares_residual=ls_normalized.detach(),
        pseudoinverse_threshold=threshold.detach(),
        defined=defined,
    )
