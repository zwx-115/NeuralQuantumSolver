from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import torch

from .exact import ExactDiagonalizer
from .estimators import exact_state
from .models import NeuralQuantumState
from .systems import PhysicalSystem


@dataclass(frozen=True)
class FrozenTarget:
    configurations: torch.Tensor
    amplitudes: torch.Tensor
    time: float


def exact_ground_state_target(
    system: PhysicalSystem, *, dtype: torch.dtype = torch.complex128
) -> FrozenTarget:
    result = ExactDiagonalizer(dtype=dtype).ground_state(system)
    return FrozenTarget(system.hilbert.all_states(), result.state.detach(), 0.0)


def nqs_ground_state_target(
    model: NeuralQuantumState, system: PhysicalSystem
) -> FrozenTarget:
    """Materialize a trained NQS ground state as the t=0 evolution target."""
    state = exact_state(model, system)
    return FrozenTarget(state.configurations, state.amplitudes.detach(), 0.0)


def freeze_model_target(model: NeuralQuantumState, system: PhysicalSystem) -> NeuralQuantumState:
    target = deepcopy(model).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target


def evolve_target(
    system: PhysicalSystem, target: FrozenTarget, dt: float, *,
    dtype: torch.dtype = torch.complex128,
) -> FrozenTarget:
    state = ExactDiagonalizer(dtype=dtype).evolve(system, target.amplitudes, dt).detach()
    return FrozenTarget(target.configurations, state, target.time + dt)
