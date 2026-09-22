from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import torch

from .derivatives import (
    LogJacobian,
    LogJacobianStrategy,
    sequential_log_derivative_matrix,
    vmap_log_derivative_matrix,
)
from .estimators import SampledEnergy
from .models import NeuralQuantumState
from .qgt import QGTDiagnostics, quantum_geometric_tensor, solve_sr


@dataclass(frozen=True)
class OptimizerStep:
    gradient_norm: float
    update_norm: float
    metrics: dict[str, float | int | torch.Tensor] = field(default_factory=dict)


class GroundStateOptimizer(ABC):
    """Strategy interface used by GroundStateDriver."""

    @abstractmethod
    def step(self, model: NeuralQuantumState, statistics: SampledEnergy) -> OptimizerStep:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict:
            raise ValueError(f"{type(self).__name__} has no restorable state")


def _parameter_norm(tensors) -> float:
    return math.sqrt(sum(float(torch.sum(torch.abs(tensor) ** 2)) for tensor in tensors))


class Adam(GroundStateOptimizer):
    """PyTorch Adam using the VMC score-function energy gradient."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._optimizer: torch.optim.Adam | None = None

    def _get_optimizer(self, model: NeuralQuantumState) -> torch.optim.Adam:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        return self._optimizer

    def step(self, model: NeuralQuantumState, statistics: SampledEnergy) -> OptimizerStep:
        optimizer = self._get_optimizer(model)
        optimizer.zero_grad(set_to_none=True)
        log_values = model.log_psi(statistics.configurations)
        centered_energy = (statistics.local_energies - statistics.energy).detach()
        surrogate = 2 * torch.real(
            torch.sum(statistics.weights * centered_energy * log_values.conj())
        )
        surrogate.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        gradient_norm = _parameter_norm(gradients)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.step()
        updates = [
            parameter.detach() - previous
            for parameter, previous in zip(model.parameters(), before)
        ]
        return OptimizerStep(gradient_norm, _parameter_norm(updates))

    def state_dict(self) -> dict:
        return {} if self._optimizer is None else self._optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        if self._optimizer is None:
            raise RuntimeError("Adam must perform setup/one step before loading its state")
        self._optimizer.load_state_dict(state_dict)


def flatten_tensors(tensors) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def log_derivative_matrix(
    model: NeuralQuantumState, configurations: torch.Tensor
) -> torch.Tensor:
    """Compatibility wrapper for the original sequential implementation."""
    return sequential_log_derivative_matrix(model, configurations)


def log_derivative_matrix_vmap(
    model: NeuralQuantumState,
    configurations: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Compatibility wrapper for the batched vmap implementation."""
    return vmap_log_derivative_matrix(
        model, configurations, chunk_size=chunk_size
    )


@torch.no_grad()
def apply_flat_update(model: NeuralQuantumState, update: torch.Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.add_(update[offset : offset + count].reshape_as(parameter))
        offset += count
    if offset != update.numel():
        raise RuntimeError("update and model parameter sizes differ")


class SR(GroundStateOptimizer):
    """Dense stochastic reconfiguration/natural-gradient optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        *,
        regularization: float = 1e-3,
        rcond: float = 1e-12,
        jacobian: LogJacobianStrategy | None = None,
    ) -> None:
        if learning_rate <= 0 or regularization < 0 or rcond <= 0:
            raise ValueError("invalid SR settings")
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.rcond = rcond
        if jacobian is not None and not callable(jacobian):
            raise TypeError("jacobian must be callable")
        # Keep the pre-refactor SR behavior unless a strategy is injected.
        self.jacobian = (
            jacobian if jacobian is not None else LogJacobian(method="sequential")
        )

    def step(self, model: NeuralQuantumState, statistics: SampledEnergy) -> OptimizerStep:
        derivatives = self.jacobian(model, statistics.configurations)
        qgt = quantum_geometric_tensor(derivatives, statistics.weights)
        mean = torch.sum(statistics.weights[:, None] * derivatives, dim=0)
        centered = derivatives - mean
        force = torch.sum(
            statistics.weights[:, None]
            * centered.conj()
            * (statistics.local_energies - statistics.energy).detach()[:, None],
            dim=0,
        )
        update, diagnostics = solve_sr(
            qgt,
            force,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            rcond=self.rcond,
        )
        apply_flat_update(model, update)
        return OptimizerStep(
            diagnostics.gradient_norm,
            diagnostics.update_norm,
            _diagnostic_metrics(diagnostics),
        )


def _diagnostic_metrics(diagnostics: QGTDiagnostics) -> dict:
    return {
        "qgt_eigenvalues": diagnostics.eigenvalues,
        "qgt_singular_values": diagnostics.singular_values,
        "effective_rank": diagnostics.effective_rank,
        "condition_number": diagnostics.condition_number,
        "regularization": diagnostics.regularization,
        "fs_step_norm": diagnostics.fs_step_norm,
    }


StochasticReconfiguration = SR
