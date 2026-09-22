"""与 SR 更新解耦的 QGT 谱诊断。"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ..derivatives import LogJacobian
from ..estimators import exact_state
from ..models import NeuralQuantumState
from ..qgt import quantum_geometric_tensor
from ..systems import PhysicalSystem


@dataclass(frozen=True)
class QGTSpectrum:
    """未正则化、精确求和 QGT 的谱。

    ``numerical_rank`` 使用 ``max(atol, rtol * lambda_max)`` 阈值；
    ``effective_rank`` 则是连续的参与率 Tr(S)^2/Tr(S^2)。
    """

    eigenvalues: torch.Tensor
    singular_values: torch.Tensor
    numerical_rank: int
    effective_rank: torch.Tensor
    retained_condition_number: torch.Tensor
    effective_rank_fraction: torch.Tensor
    threshold: torch.Tensor
    num_parameters: int


def qgt_spectrum(
    model: NeuralQuantumState,
    system: PhysicalSystem,
    *,
    jacobian: LogJacobian | None = None,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> QGTSpectrum:
    """计算全希尔伯特空间 QGT 诊断，但不修改 ``model``。"""
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    state = exact_state(model, system)
    derivatives = (jacobian or LogJacobian())(model, state.configurations)
    qgt = quantum_geometric_tensor(derivatives, state.probabilities)
    eigenvalues = torch.linalg.eigvalsh(qgt).real
    # 舍入误差可能产生极小负本征值；谱定义使用半正定部分。
    spectrum = eigenvalues.clamp_min(0)
    singular_values = torch.linalg.svdvals(qgt).real
    maximum = spectrum.max() if spectrum.numel() else torch.zeros((), device=qgt.device)
    threshold = torch.maximum(
        torch.as_tensor(atol, dtype=spectrum.dtype, device=spectrum.device), maximum * rtol
    )
    retained = spectrum[spectrum > threshold]
    trace = spectrum.sum()
    trace_squared = spectrum.square().sum()
    effective_rank = torch.where(
        trace_squared > 0, trace.square() / trace_squared, torch.zeros_like(trace)
    )
    condition = (
        retained.max() / retained.min()
        if retained.numel() > 0
        else torch.full((), float("inf"), dtype=spectrum.dtype, device=spectrum.device)
    )
    parameter_count = derivatives.shape[1]
    return QGTSpectrum(
        eigenvalues=eigenvalues.detach(),
        singular_values=singular_values.detach(),
        numerical_rank=int(retained.numel()),
        effective_rank=effective_rank.detach(),
        retained_condition_number=condition.detach(),
        effective_rank_fraction=(effective_rank / parameter_count).detach(),
        threshold=threshold.detach(),
        num_parameters=parameter_count,
    )
