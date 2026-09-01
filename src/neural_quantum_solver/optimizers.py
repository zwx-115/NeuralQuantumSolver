from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from time import perf_counter
import torch

from .derivatives import (
    LogJacobian,
    LogJacobianStrategy,
    RealLogDerivativeParts,
    sequential_log_derivative_matrix,
    vmap_log_derivative_matrix,
)
from .estimators import SampledEnergy, ShardedSampledEnergy
from .real_estimators import (
    DistributedRealPairSampledEnergy,
    RealPairSampledEnergy,
    RealPairShardedSampledEnergy,
)
from .devices import run_on_device, synchronize_devices, transfer_tensor
from .models import NeuralQuantumState
from .qgt import (
    QGTDiagnostics,
    quantum_geometric_tensor,
    real_quantum_geometric_tensor,
    solve_sr,
)


@dataclass(frozen=True)
class OptimizerStep:
    gradient_norm: float
    update_norm: float
    metrics: dict[str, float | int | torch.Tensor] = field(default_factory=dict)


class GroundStateOptimizer(ABC):
    """供 GroundStateDriver 调用的优化策略接口。"""

    @abstractmethod
    def step(
        self,
        model: NeuralQuantumState,
        statistics: (
            SampledEnergy
            | ShardedSampledEnergy
            | DistributedRealPairSampledEnergy
            | RealPairSampledEnergy
            | RealPairShardedSampledEnergy
        ),
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
    """使用 VMC 得分函数能量梯度的 PyTorch Adam。"""

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

    @staticmethod
    def _backward_real_shard(
        model: NeuralQuantumState, statistics: RealPairSampledEnergy
    ) -> None:
        parts = model.log_psi_parts(statistics.configurations)
        centered_real = (statistics.local_energy_real - statistics.energy).detach()
        centered_imag = (
            statistics.local_energy_imag - statistics.energy_imag
        ).detach()
        surrogate = 2 * torch.sum(
            statistics.weights
            * (
                centered_real * parts.log_amplitude
                + centered_imag * parts.phase
            )
        )
        surrogate.backward()

    def step(
        self,
        model: NeuralQuantumState,
        statistics: (
            SampledEnergy
            | ShardedSampledEnergy
            | DistributedRealPairSampledEnergy
            | RealPairSampledEnergy
            | RealPairShardedSampledEnergy
        ),
    ) -> OptimizerStep:
        optimizer = self._get_optimizer(model)
        optimizer.zero_grad(set_to_none=True)
        devices = (
            tuple(next(replica.parameters()).device for replica in statistics.models)
            if isinstance(
                statistics,
                (ShardedSampledEnergy, RealPairShardedSampledEnergy),
            )
            else (next(model.parameters()).device,)
        )
        synchronize_devices(devices)
        backward_started = perf_counter()
        if isinstance(
            statistics, (ShardedSampledEnergy, RealPairShardedSampledEnergy)
        ):
            if statistics.models[0] is not model:
                raise ValueError("the first statistics replica must be the primary model")
            for replica in statistics.models[1:]:
                replica.zero_grad(set_to_none=True)

            def backward_shard(pair):
                replica, shard = pair
                operation = (
                    self._backward_real_shard
                    if isinstance(shard, RealPairSampledEnergy)
                    else self._backward_shard
                )
                return run_on_device(
                    next(replica.parameters()).device,
                    operation,
                    replica,
                    shard,
                )

            tuple(
                backward_shard(pair)
                for pair in zip(statistics.models, statistics.shards)
            )
            parameter_groups = zip(
                *(tuple(replica.parameters()) for replica in statistics.models)
            )
            for parameters in parameter_groups:
                primary = parameters[0]
                for replica_parameter in parameters[1:]:
                    if replica_parameter.grad is None:
                        continue
                    replica_gradient = transfer_tensor(
                        replica_parameter.grad, primary.device
                    )
                    if primary.grad is None:
                        primary.grad = replica_gradient.clone()
                    else:
                        primary.grad.add_(replica_gradient)
        else:
            if isinstance(statistics, RealPairSampledEnergy):
                self._backward_real_shard(model, statistics)
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
            if isinstance(
                statistics,
                (ShardedSampledEnergy, RealPairShardedSampledEnergy),
            )
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
    """旧版逐样本实现的兼容包装函数。"""
    return sequential_log_derivative_matrix(model, configurations)


def log_derivative_matrix_vmap(
    model: NeuralQuantumState,
    configurations: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """批量 vmap 实现的兼容包装函数。"""
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
    """稠密随机重构（自然梯度）优化器。"""

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
        # 未显式传入策略时，保持重构前 SR 的默认行为。
        self.jacobian = (
            jacobian if jacobian is not None else LogJacobian(method="sequential")
        )

    def _sharded_qgt_and_force(
        self, statistics: ShardedSampledEnergy
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        devices = tuple(next(model.parameters()).device for model in statistics.models)
        synchronize_devices(devices)
        jacobian_started = perf_counter()

        def evaluate_jacobian(pair):
            replica, shard = pair
            return run_on_device(
                next(replica.parameters()).device,
                self.jacobian,
                replica,
                shard.configurations,
            )

        derivatives = tuple(
            evaluate_jacobian(pair)
            for pair in zip(statistics.models, statistics.shards)
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
            mean.add_(
                transfer_tensor(
                    torch.sum(weights[:, None] * values, dim=0),
                    primary,
                )
            )
            second_moment.add_(
                transfer_tensor(
                    (values.conj().mT * weights) @ values,
                    primary,
                )
            )
        qgt = second_moment - mean.conj()[:, None] * mean[None, :]
        force = torch.zeros(parameter_count, dtype=dtype, device=primary)
        for values, shard in zip(derivatives, statistics.shards):
            centered = values - transfer_tensor(mean, values.device)
            force.add_(
                transfer_tensor(
                    torch.sum(
                        shard.weights[:, None]
                        * centered.conj()
                        * (shard.local_energies - shard.energy).detach()[:, None],
                        dim=0,
                    ),
                    primary,
                )
            )
        synchronize_devices(devices)
        qgt_force_seconds = perf_counter() - qgt_started
        return qgt, force, jacobian_seconds, qgt_force_seconds

    def _real_qgt_and_force(
        self,
        model: NeuralQuantumState,
        statistics: RealPairSampledEnergy,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        device = next(model.parameters()).device
        synchronize_devices((device,))
        jacobian_started = perf_counter()
        derivatives = self.jacobian(model, statistics.configurations)
        if not isinstance(derivatives, RealLogDerivativeParts):
            raise TypeError("real-pair SR requires real log-derivative parts")
        synchronize_devices((device,))
        jacobian_seconds = perf_counter() - jacobian_started
        qgt_started = perf_counter()
        weights = statistics.weights
        qgt = real_quantum_geometric_tensor(
            derivatives.log_amplitude, derivatives.phase, weights
        )
        amplitude_mean = torch.sum(
            weights[:, None] * derivatives.log_amplitude, dim=0
        )
        phase_mean = torch.sum(weights[:, None] * derivatives.phase, dim=0)
        amplitude_centered = derivatives.log_amplitude - amplitude_mean
        phase_centered = derivatives.phase - phase_mean
        centered_real = (
            statistics.local_energy_real - statistics.energy
        ).detach()
        centered_imag = (
            statistics.local_energy_imag - statistics.energy_imag
        ).detach()
        force = torch.sum(
            weights[:, None]
            * (
                amplitude_centered * centered_real[:, None]
                + phase_centered * centered_imag[:, None]
            ),
            dim=0,
        )
        synchronize_devices((device,))
        return (
            qgt,
            force,
            jacobian_seconds,
            perf_counter() - qgt_started,
        )

    def _sharded_real_qgt_and_force(
        self, statistics: RealPairShardedSampledEnergy
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        devices = tuple(next(model.parameters()).device for model in statistics.models)
        synchronize_devices(devices)
        jacobian_started = perf_counter()

        def evaluate_jacobian(pair):
            replica, shard = pair
            return run_on_device(
                next(replica.parameters()).device,
                self.jacobian,
                replica,
                shard.configurations,
            )

        derivatives = tuple(
            evaluate_jacobian(pair)
            for pair in zip(statistics.models, statistics.shards)
        )
        if any(not isinstance(value, RealLogDerivativeParts) for value in derivatives):
            raise TypeError("real-pair SR requires real log-derivative parts")
        synchronize_devices(devices)
        jacobian_seconds = perf_counter() - jacobian_started
        qgt_started = perf_counter()
        primary = devices[0]
        parameter_count = derivatives[0].log_amplitude.shape[1]
        dtype = derivatives[0].log_amplitude.dtype
        amplitude_mean = torch.zeros(
            parameter_count, dtype=dtype, device=primary
        )
        phase_mean = torch.zeros_like(amplitude_mean)
        second_moment = torch.zeros(
            (parameter_count, parameter_count), dtype=dtype, device=primary
        )
        for values, shard in zip(derivatives, statistics.shards):
            weights = shard.weights
            amplitude_mean.add_(
                transfer_tensor(
                    torch.sum(
                        weights[:, None] * values.log_amplitude, dim=0
                    ),
                    primary,
                )
            )
            phase_mean.add_(
                transfer_tensor(
                    torch.sum(weights[:, None] * values.phase, dim=0),
                    primary,
                )
            )
            second_moment.add_(
                transfer_tensor(
                    (values.log_amplitude.mT * weights) @ values.log_amplitude
                    + (values.phase.mT * weights) @ values.phase,
                    primary,
                )
            )
        qgt = second_moment - (
            amplitude_mean[:, None] * amplitude_mean[None, :]
            + phase_mean[:, None] * phase_mean[None, :]
        )
        force = torch.zeros(parameter_count, dtype=dtype, device=primary)
        for values, shard in zip(derivatives, statistics.shards):
            amplitude_centered = (
                values.log_amplitude
                - transfer_tensor(
                    amplitude_mean, values.log_amplitude.device
                )
            )
            phase_centered = (
                values.phase
                - transfer_tensor(phase_mean, values.phase.device)
            )
            centered_real = (
                shard.local_energy_real - shard.energy
            ).detach()
            centered_imag = (
                shard.local_energy_imag - shard.energy_imag
            ).detach()
            force.add_(
                transfer_tensor(
                    torch.sum(
                        shard.weights[:, None]
                        * (
                            amplitude_centered * centered_real[:, None]
                            + phase_centered * centered_imag[:, None]
                        ),
                        dim=0,
                    ),
                    primary,
                )
            )
        synchronize_devices(devices)
        return qgt, force, jacobian_seconds, perf_counter() - qgt_started

    def _distributed_real_qgt_and_force(
        self, statistics: DistributedRealPairSampledEnergy
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        context = statistics.context
        device = context.device
        synchronize_devices((device,))
        context.barrier()
        jacobian_started = perf_counter()
        derivatives = self.jacobian(
            statistics.model, statistics.shard.configurations
        )
        if not isinstance(derivatives, RealLogDerivativeParts):
            raise TypeError("real-pair SR requires real log-derivative parts")
        synchronize_devices((device,))
        context.barrier()
        jacobian_seconds = perf_counter() - jacobian_started

        qgt_started = perf_counter()
        weights = statistics.shard.weights
        amplitude_mean = context.all_reduce(
            torch.sum(weights[:, None] * derivatives.log_amplitude, dim=0)
        )
        phase_mean = context.all_reduce(
            torch.sum(weights[:, None] * derivatives.phase, dim=0)
        )
        second_moment = context.all_reduce(
            (derivatives.log_amplitude.mT * weights)
            @ derivatives.log_amplitude
            + (derivatives.phase.mT * weights) @ derivatives.phase
        )
        qgt = second_moment - (
            amplitude_mean[:, None] * amplitude_mean[None, :]
            + phase_mean[:, None] * phase_mean[None, :]
        )
        amplitude_centered = derivatives.log_amplitude - amplitude_mean
        phase_centered = derivatives.phase - phase_mean
        centered_real = (
            statistics.shard.local_energy_real - statistics.energy
        ).detach()
        centered_imag = (
            statistics.shard.local_energy_imag - statistics.energy_imag
        ).detach()
        force = context.all_reduce(
            torch.sum(
                weights[:, None]
                * (
                    amplitude_centered * centered_real[:, None]
                    + phase_centered * centered_imag[:, None]
                ),
                dim=0,
            )
        )
        synchronize_devices((device,))
        context.barrier()
        return qgt, force, jacobian_seconds, perf_counter() - qgt_started

    def _distributed_real_step(
        self,
        model: NeuralQuantumState,
        statistics: DistributedRealPairSampledEnergy,
    ) -> OptimizerStep:
        context = statistics.context
        qgt, force, jacobian_seconds, qgt_force_seconds = (
            self._distributed_real_qgt_and_force(statistics)
        )
        if context.is_main:
            update, diagnostics = solve_sr(
                qgt,
                force,
                learning_rate=self.learning_rate,
                regularization=self.regularization,
                rcond=self.rcond,
                solver_device=self.solver_device,
            )
            scalars = torch.tensor(
                [
                    diagnostics.gradient_norm,
                    diagnostics.update_norm,
                    diagnostics.fs_step_norm,
                    float(diagnostics.effective_rank),
                    diagnostics.condition_number,
                    diagnostics.solve_seconds,
                ],
                dtype=force.dtype,
                device=context.device,
            )
            metrics = _diagnostic_metrics(diagnostics)
        else:
            update = torch.zeros_like(force)
            scalars = torch.zeros(6, dtype=force.dtype, device=context.device)
            metrics = {}
        context.broadcast(update)
        context.broadcast(scalars)
        apply_flat_update(model, update)
        metrics.update(
            {
                "effective_rank": int(scalars[3].item()),
                "condition_number": float(scalars[4].item()),
                "regularization": self.regularization,
                "fs_step_norm": float(scalars[2].item()),
                "solve_seconds": float(scalars[5].item()),
                "jacobian_seconds": jacobian_seconds,
                "qgt_force_seconds": qgt_force_seconds,
                "qgt_dimension": qgt.shape[0],
                "num_devices": context.world_size,
            }
        )
        return OptimizerStep(
            float(scalars[0].item()),
            float(scalars[1].item()),
            metrics,
        )

    def step(
        self,
        model: NeuralQuantumState,
        statistics: (
            SampledEnergy
            | ShardedSampledEnergy
            | DistributedRealPairSampledEnergy
            | RealPairSampledEnergy
            | RealPairShardedSampledEnergy
        ),
    ) -> OptimizerStep:
        if isinstance(statistics, DistributedRealPairSampledEnergy):
            return self._distributed_real_step(model, statistics)
        if isinstance(statistics, RealPairShardedSampledEnergy):
            qgt, force, jacobian_seconds, qgt_force_seconds = (
                self._sharded_real_qgt_and_force(statistics)
            )
        elif isinstance(statistics, RealPairSampledEnergy):
            qgt, force, jacobian_seconds, qgt_force_seconds = (
                self._real_qgt_and_force(model, statistics)
            )
        elif isinstance(statistics, ShardedSampledEnergy):
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
        if isinstance(
            statistics, (ShardedSampledEnergy, RealPairShardedSampledEnergy)
        ):
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
