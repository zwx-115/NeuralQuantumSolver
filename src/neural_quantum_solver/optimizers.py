from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
from time import perf_counter
import torch

from .derivatives import (
    LogJacobian,
    LogJacobianStrategy,
    sequential_log_derivative_matrix,
    vmap_log_derivative_matrix,
)
from .estimators import SampledEnergy, ShardedSampledEnergy
from .devices import synchronize_devices
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
    def step(
        self,
        model: NeuralQuantumState,
        statistics: SampledEnergy | ShardedSampledEnergy,
    ) -> OptimizerStep:
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

    @staticmethod
    def _backward_shard(
        model: NeuralQuantumState, statistics: SampledEnergy
    ) -> None:
        log_values = model.log_psi(statistics.configurations)
        centered_energy = (statistics.local_energies - statistics.energy).detach()
        surrogate = 2 * torch.real(
            torch.sum(statistics.weights * centered_energy * log_values.conj())
        )
        surrogate.backward()

    def step(
        self,
        model: NeuralQuantumState,
        statistics: SampledEnergy | ShardedSampledEnergy,
    ) -> OptimizerStep:
        optimizer = self._get_optimizer(model)
        optimizer.zero_grad(set_to_none=True)
        devices = (
            tuple(next(replica.parameters()).device for replica in statistics.models)
            if isinstance(statistics, ShardedSampledEnergy)
            else (next(model.parameters()).device,)
        )
        synchronize_devices(devices)
        backward_started = perf_counter()
        if isinstance(statistics, ShardedSampledEnergy):
            if statistics.models[0] is not model:
                raise ValueError("the first statistics replica must be the primary model")
            for replica in statistics.models[1:]:
                replica.zero_grad(set_to_none=True)
            with ThreadPoolExecutor(max_workers=len(statistics.models)) as executor:
                tuple(
                    executor.map(
                        lambda pair: self._backward_shard(pair[0], pair[1]),
                        zip(statistics.models, statistics.shards),
                    )
                )
            parameter_groups = zip(
                *(tuple(replica.parameters()) for replica in statistics.models)
            )
            for parameters in parameter_groups:
                primary = parameters[0]
                for replica_parameter in parameters[1:]:
                    if replica_parameter.grad is None:
                        continue
                    replica_gradient = replica_parameter.grad.to(primary.device)
                    if primary.grad is None:
                        primary.grad = replica_gradient.clone()
                    else:
                        primary.grad.add_(replica_gradient)
        else:
            self._backward_shard(model, statistics)
        synchronize_devices(devices)
        backward_seconds = perf_counter() - backward_started
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
        metrics = (
            {"num_devices": len(statistics.models)}
            if isinstance(statistics, ShardedSampledEnergy)
            else {}
        )
        metrics["backward_seconds"] = backward_seconds
        return OptimizerStep(gradient_norm, _parameter_norm(updates), metrics)

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
        solver_device: torch.device | str | None = "auto",
    ) -> None:
        if learning_rate <= 0 or regularization < 0 or rcond <= 0:
            raise ValueError("invalid SR settings")
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.rcond = rcond
        self.solver_device = solver_device
        if jacobian is not None and not callable(jacobian):
            raise TypeError("jacobian must be callable")
        # Keep the pre-refactor SR behavior unless a strategy is injected.
        self.jacobian = (
            jacobian if jacobian is not None else LogJacobian(method="sequential")
        )

    def _sharded_qgt_and_force(
        self, statistics: ShardedSampledEnergy
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        devices = tuple(next(model.parameters()).device for model in statistics.models)
        synchronize_devices(devices)
        jacobian_started = perf_counter()
        with ThreadPoolExecutor(max_workers=len(statistics.models)) as executor:
            derivatives = tuple(
                executor.map(
                    lambda pair: self.jacobian(pair[0], pair[1].configurations),
                    zip(statistics.models, statistics.shards),
                )
            )
        synchronize_devices(devices)
        jacobian_seconds = perf_counter() - jacobian_started
        qgt_started = perf_counter()
        primary = next(statistics.models[0].parameters()).device
        parameter_count = derivatives[0].shape[1]
        dtype = derivatives[0].dtype
        mean = torch.zeros(parameter_count, dtype=dtype, device=primary)
        second_moment = torch.zeros(
            (parameter_count, parameter_count), dtype=dtype, device=primary
        )
        for values, shard in zip(derivatives, statistics.shards):
            weights = shard.weights
            mean.add_(torch.sum(weights[:, None] * values, dim=0).to(primary))
            second_moment.add_(
                ((values.conj().mT * weights) @ values).to(primary)
            )
        qgt = second_moment - mean.conj()[:, None] * mean[None, :]
        force = torch.zeros(parameter_count, dtype=dtype, device=primary)
        for values, shard in zip(derivatives, statistics.shards):
            centered = values - mean.to(values.device)
            force.add_(
                torch.sum(
                    shard.weights[:, None]
                    * centered.conj()
                    * (shard.local_energies - shard.energy).detach()[:, None],
                    dim=0,
                ).to(primary)
            )
        synchronize_devices(devices)
        qgt_force_seconds = perf_counter() - qgt_started
        return qgt, force, jacobian_seconds, qgt_force_seconds

    def step(
        self,
        model: NeuralQuantumState,
        statistics: SampledEnergy | ShardedSampledEnergy,
    ) -> OptimizerStep:
        if isinstance(statistics, ShardedSampledEnergy):
            qgt, force, jacobian_seconds, qgt_force_seconds = (
                self._sharded_qgt_and_force(statistics)
            )
        else:
            device = next(model.parameters()).device
            synchronize_devices((device,))
            jacobian_started = perf_counter()
            derivatives = self.jacobian(model, statistics.configurations)
            synchronize_devices((device,))
            jacobian_seconds = perf_counter() - jacobian_started
            qgt_started = perf_counter()
            qgt = quantum_geometric_tensor(derivatives, statistics.weights)
            weighted_derivatives = statistics.weights[:, None] * derivatives
            mean = torch.complex(
                torch.sum(weighted_derivatives.real, dim=0),
                torch.sum(weighted_derivatives.imag, dim=0),
            )
            centered = derivatives - mean
            force_terms = (
                statistics.weights[:, None]
                * centered.conj()
                * (statistics.local_energies - statistics.energy).detach()[:, None]
            )
            force = torch.complex(
                torch.sum(force_terms.real, dim=0),
                torch.sum(force_terms.imag, dim=0),
            )
            synchronize_devices((device,))
            qgt_force_seconds = perf_counter() - qgt_started
        update, diagnostics = solve_sr(
            qgt,
            force,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            rcond=self.rcond,
            solver_device=self.solver_device,
        )
        apply_flat_update(model, update.to(next(model.parameters()).device))
        metrics = _diagnostic_metrics(diagnostics)
        metrics["jacobian_seconds"] = jacobian_seconds
        metrics["qgt_force_seconds"] = qgt_force_seconds
        metrics["solve_seconds"] = diagnostics.solve_seconds
        metrics["qgt_dimension"] = qgt.shape[0]
        if isinstance(statistics, ShardedSampledEnergy):
            metrics["num_devices"] = len(statistics.models)
        return OptimizerStep(
            diagnostics.gradient_norm,
            diagnostics.update_norm,
            metrics,
        )


def _diagnostic_metrics(diagnostics: QGTDiagnostics) -> dict:
    return {
        "qgt_eigenvalues": diagnostics.eigenvalues,
        "qgt_singular_values": diagnostics.singular_values,
        "effective_rank": diagnostics.effective_rank,
        "condition_number": diagnostics.condition_number,
        "regularization": diagnostics.regularization,
        "fs_step_norm": diagnostics.fs_step_norm,
        "solve_seconds": diagnostics.solve_seconds,
    }


StochasticReconfiguration = SR
