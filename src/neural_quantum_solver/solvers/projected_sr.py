"""用于冻结态保真度拟合和 p-tVMC 投影的全求和 SR 求解器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from ..derivatives import LogJacobian
from ..estimators import exact_state
from ..models import NeuralQuantumState
from ..qgt import quantum_geometric_tensor, solve_sr
from ..systems import PhysicalSystem


@dataclass(frozen=True)
class OverlapSRSettings:
    """直接 overlap loss 的 SR 参数。"""

    learning_rate: float = 0.05
    regularization: float = 1e-3
    rcond: float = 1e-12
    jacobian_chunk_size: int = 256
    max_backtracks: int = 8
    backtrack_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.regularization < 0 or self.rcond <= 0:
            raise ValueError("invalid overlap SR settings")
        if self.jacobian_chunk_size <= 0 or self.max_backtracks < 0:
            raise ValueError("invalid overlap SR Jacobian/backtracking settings")
        if not 0 < self.backtrack_factor < 1:
            raise ValueError("backtrack_factor must lie between zero and one")

    def config(self) -> dict[str, float | int]:
        """返回可直接写入 JSON 的参数字典。"""
        return asdict(self)


@dataclass(frozen=True)
class OverlapSRStep:
    """一次 SR 更新的 loss 和 QGT 数值诊断。"""

    loss_before: float
    loss_after: float
    accepted: bool
    backtracks: int
    step_scale: float
    gradient_norm: float
    update_norm: float
    fs_step_norm: float
    qgt_rank: int
    qgt_condition_number: float

    def row(self, step: int) -> dict[str, float | int | bool]:
        """转换为适合写入 CSV 的标量行。"""
        return {"iteration": step} | asdict(self)


def overlap_infidelity(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """计算两个已归一化态之间的 ``1-|<target|candidate>|^2``。"""
    return 1.0 - torch.abs(torch.vdot(target, candidate)) ** 2


def _flatten_gradients(model: NeuralQuantumState) -> torch.Tensor:
    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            gradients.append(torch.zeros_like(parameter).reshape(-1))
        else:
            gradients.append(parameter.grad.reshape(-1))
    return torch.cat(gradients)


@torch.no_grad()
def _apply_update(model: NeuralQuantumState, update: torch.Tensor, scale: float) -> None:
    """将展平 SR 更新按给定回溯系数加到模型参数。"""
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.add_(scale * update[offset : offset + count].reshape_as(parameter))
        offset += count
    if offset != update.numel():
        raise RuntimeError("update and model parameter sizes differ")


def _state_dict(model: NeuralQuantumState) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _evaluate_loss(
    model: NeuralQuantumState, system: PhysicalSystem, target: torch.Tensor
) -> torch.Tensor:
    return overlap_infidelity(target, exact_state(model, system).amplitudes)


def overlap_sr_step(
    model: NeuralQuantumState,
    system: PhysicalSystem,
    target: torch.Tensor,
    *,
    settings: OverlapSRSettings,
) -> OverlapSRStep:
    """对直接 overlap loss 执行一次带回溯的全求和 SR 更新。"""
    model.zero_grad(set_to_none=True)
    state = exact_state(model, system)
    loss = overlap_infidelity(target, state.amplitudes)
    loss.backward()
    gradient = _flatten_gradients(model).detach()
    jacobian = LogJacobian(method="vmap", chunk_size=settings.jacobian_chunk_size)
    derivatives = jacobian(model, state.configurations)
    qgt = quantum_geometric_tensor(derivatives, state.probabilities.detach())
    update, diagnostics = solve_sr(
        qgt,
        gradient,
        learning_rate=settings.learning_rate,
        regularization=settings.regularization,
        rcond=settings.rcond,
    )
    before = float(loss.detach().item())
    original = _state_dict(model)
    accepted = False
    scale = 1.0
    after = before
    backtracks = 0
    for backtracks in range(settings.max_backtracks + 1):
        _apply_update(model, update, scale)
        with torch.no_grad():
            trial = float(_evaluate_loss(model, system, target).item())
        if torch.isfinite(torch.as_tensor(trial)) and trial < before:
            accepted = True
            after = trial
            break
        model.load_state_dict(original)
        scale *= settings.backtrack_factor
    if not accepted:
        backtracks = settings.max_backtracks
        scale = 0.0
    return OverlapSRStep(
        loss_before=before,
        loss_after=after,
        accepted=accepted,
        backtracks=backtracks,
        step_scale=scale,
        gradient_norm=diagnostics.gradient_norm,
        update_norm=diagnostics.update_norm * scale,
        fs_step_norm=diagnostics.fs_step_norm * scale,
        qgt_rank=diagnostics.effective_rank,
        qgt_condition_number=diagnostics.condition_number,
    )


def optimize_overlap_sr(
    model: NeuralQuantumState,
    system: PhysicalSystem,
    target: torch.Tensor,
    *,
    max_steps: int,
    tolerance: float,
    settings: OverlapSRSettings,
    on_step: Callable[[OverlapSRStep, int], None] | None = None,
) -> tuple[float, float, int, bool, dict[str, torch.Tensor]]:
    """原地运行多步 SR，并返回初始、最佳和最终 loss 及最佳参数。"""
    if max_steps < 1 or tolerance < 0:
        raise ValueError("max_steps must be positive and tolerance non-negative")
    target = target.to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    ).detach()
    initial = float(_evaluate_loss(model, system, target).detach().item())
    best = initial
    best_state = _state_dict(model)
    final = initial
    converged = initial <= tolerance
    for step in range(1, max_steps + 1):
        if converged:
            return initial, best, final, step - 1, converged, best_state
        record = overlap_sr_step(model, system, target, settings=settings)
        final = record.loss_after
        if final < best:
            best = final
            best_state = _state_dict(model)
        if on_step is not None:
            on_step(record, step)
        converged = final <= tolerance
    return initial, best, final, max_steps, converged, best_state
