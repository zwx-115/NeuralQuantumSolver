"""冻结精确目标态与彼此隔离的时间切片拟合实验。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable
import torch

from ..exact import ExactDiagonalizer
from ..estimators import exact_state
from ..models import NeuralQuantumState
from ..overlap import direct_overlap_loss
from ..systems import PhysicalSystem
from .overlap_sr import OverlapSRSettings, OverlapSRStep, optimize_overlap_sr


@dataclass(frozen=True)
class ExactSnapshot:
    time: float
    configurations: torch.Tensor
    amplitudes: torch.Tensor


@dataclass(frozen=True)
class SnapshotFitResult:
    time: float
    initial_loss: float
    best_loss: float
    final_loss: float
    iterations: int
    converged: bool
    best_state_dict: dict[str, torch.Tensor]


def evolve_exact_snapshots(
    system: PhysicalSystem,
    initial_state: torch.Tensor,
    times: list[float],
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
) -> list[ExactSnapshot]:
    """返回指定非负时间点的归一化 ED 态矢量。"""
    if any(time < 0 for time in times):
        raise ValueError("snapshot times must be non-negative")
    diagonalizer = ExactDiagonalizer(dtype=dtype, device=device)
    eigenvalues, eigenvectors = diagonalizer.diagonalize(system)
    state = initial_state.to(dtype=dtype, device=device)
    if state.shape != (system.hilbert.size,):
        raise ValueError("initial_state has the wrong Hilbert-space dimension")
    state = state / torch.linalg.vector_norm(state)
    coefficients = eigenvectors.mH @ state
    configurations = system.hilbert.all_states(device=device)
    return [
        ExactSnapshot(
            time=float(time),
            configurations=configurations.detach().cpu(),
            amplitudes=(eigenvectors @ (torch.exp(-1j * time * eigenvalues) * coefficients)).detach().cpu(),
        )
        for time in times
    ]


def fit_snapshot(
    model: NeuralQuantumState,
    system: PhysicalSystem,
    snapshot: ExactSnapshot,
    *,
    optimizer_factory: Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer],
    max_steps: int,
    tolerance: float = 1e-10,
) -> SnapshotFitResult:
    """通过直接保真度将 ``model`` 的副本拟合到冻结 ED 时间切片。

    此函数绝不修改传入模型，因此可独立比较轨迹模型、同尺寸重启和更大
    RBM 重启。
    """
    if max_steps < 1 or tolerance < 0:
        raise ValueError("max_steps must be positive and tolerance non-negative")
    candidate = deepcopy(model)
    optimizer = optimizer_factory(list(candidate.parameters()))
    device = next(candidate.parameters()).device
    target = snapshot.amplitudes.to(device=device, dtype=next(candidate.parameters()).dtype).detach()
    identity = torch.eye(system.hilbert.size, dtype=target.dtype, device=device)
    initial = direct_overlap_loss(target, exact_state(candidate, system).amplitudes, identity).loss
    best_loss = float(initial.detach().item())
    best_state = {key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()}
    final_loss = initial
    converged = False
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        candidate_state = exact_state(candidate, system).amplitudes
        final_loss = direct_overlap_loss(target, candidate_state, identity).loss
        final_loss.backward()
        optimizer.step()
        value = float(final_loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_state = {key: item.detach().cpu().clone() for key, item in candidate.state_dict().items()}
        if value <= tolerance:
            converged = True
            break
    return SnapshotFitResult(
        time=snapshot.time,
        initial_loss=float(initial.detach().item()),
        best_loss=best_loss,
        final_loss=float(final_loss.detach().item()),
        iterations=step,
        converged=converged,
        best_state_dict=best_state,
    )


def fit_snapshot_sr(
    model: NeuralQuantumState,
    system: PhysicalSystem,
    snapshot: ExactSnapshot,
    *,
    settings: OverlapSRSettings,
    max_steps: int,
    tolerance: float = 1e-10,
    on_step: Callable[[OverlapSRStep, int], None] | None = None,
) -> SnapshotFitResult:
    """以全求和 SR 将模型副本拟合到冻结 ED 时间切片。"""
    candidate = deepcopy(model)
    initial, best, final, iterations, converged, best_state = optimize_overlap_sr(
        candidate,
        system,
        snapshot.amplitudes,
        max_steps=max_steps,
        tolerance=tolerance,
        settings=settings,
        on_step=on_step,
    )
    return SnapshotFitResult(
        time=snapshot.time,
        initial_loss=initial,
        best_loss=best,
        final_loss=final,
        iterations=iterations,
        converged=converged,
        best_state_dict=best_state,
    )
