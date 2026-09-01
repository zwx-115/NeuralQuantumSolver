from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol
import torch

from .models import AmplitudePhaseNQS, NeuralQuantumState


@dataclass(frozen=True)
class RealLogDerivativeParts:
    log_amplitude: torch.Tensor
    phase: torch.Tensor


LogJacobianMethod = Literal["sequential", "vmap", "analytic", "auto"]


class LogJacobianStrategy(Protocol):
    """为 SR 类优化器生成 O[n, i] 的可调用策略。"""

    def __call__(
        self, model: NeuralQuantumState, configurations: torch.Tensor
    ) -> torch.Tensor | RealLogDerivativeParts: ...


def _materialize_gradients(gradients, parameters) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(gradients, parameters)
    )


def sequential_real_log_derivative_parts(
    model: AmplitudePhaseNQS, configurations: torch.Tensor
) -> RealLogDerivativeParts:
    parameters = tuple(model.parameters())
    values = model.log_psi_parts(configurations)
    amplitude_rows, phase_rows = [], []
    for index in range(values.log_amplitude.numel()):
        amplitude_gradients = torch.autograd.grad(
            values.log_amplitude[index],
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        phase_gradients = torch.autograd.grad(
            values.phase[index],
            parameters,
            retain_graph=index + 1 < values.phase.numel(),
            create_graph=False,
            allow_unused=True,
        )
        amplitude_rows.append(
            _flatten_tensors(
                _materialize_gradients(amplitude_gradients, parameters)
            )
        )
        phase_rows.append(
            _flatten_tensors(_materialize_gradients(phase_gradients, parameters))
        )
    return RealLogDerivativeParts(
        torch.stack(amplitude_rows), torch.stack(phase_rows)
    )


def vmap_real_log_derivative_parts(
    model: AmplitudePhaseNQS,
    configurations: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> RealLogDerivativeParts:
    if configurations.ndim < 2 or configurations.shape[0] == 0:
        raise ValueError("configurations must contain a leading sample dimension")
    num_samples = configurations.shape[0]
    chunk_size = num_samples if chunk_size is None else chunk_size
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer or None")
    parameters = tuple(model.parameters())
    amplitude_chunks, phase_chunks = [], []
    for configuration_chunk in configurations.split(chunk_size, dim=0):
        values = model.log_psi_parts(configuration_chunk)
        basis = torch.eye(
            configuration_chunk.shape[0],
            dtype=values.log_amplitude.dtype,
            device=values.log_amplitude.device,
        )

        def amplitude_vjp(cotangent: torch.Tensor):
            gradients = torch.autograd.grad(
                values.log_amplitude,
                parameters,
                grad_outputs=cotangent,
                retain_graph=True,
                allow_unused=True,
            )
            return _materialize_gradients(gradients, parameters)

        def phase_vjp(cotangent: torch.Tensor):
            gradients = torch.autograd.grad(
                values.phase,
                parameters,
                grad_outputs=cotangent,
                retain_graph=True,
                allow_unused=True,
            )
            return _materialize_gradients(gradients, parameters)

        amplitude_gradients = torch.vmap(amplitude_vjp)(basis)
        phase_gradients = torch.vmap(phase_vjp)(basis)
        amplitude_chunks.append(
            torch.cat(
                [gradient.flatten(start_dim=1) for gradient in amplitude_gradients],
                dim=1,
            )
        )
        phase_chunks.append(
            torch.cat(
                [gradient.flatten(start_dim=1) for gradient in phase_gradients],
                dim=1,
            )
        )
    return RealLogDerivativeParts(
        torch.cat(amplitude_chunks, dim=0),
        torch.cat(phase_chunks, dim=0),
    )


def _flatten_tensors(tensors) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def sequential_log_derivative_matrix(
    model: NeuralQuantumState, configurations: torch.Tensor
) -> torch.Tensor:
    """全纯复数模型的逐样本对数 Jacobian 参考实现。"""
    parameters = tuple(model.parameters())
    log_values = model.log_psi(configurations)
    rows = []
    for index in range(log_values.numel()):
        gradients = torch.autograd.grad(
            log_values[index].real,
            parameters,
            retain_graph=index + 1 < log_values.numel(),
            create_graph=False,
        )
        # PyTorch 返回实标量对应的共轭 Wirtinger 梯度。
        rows.append(_flatten_tensors(gradients).conj())
    return torch.stack(rows)


def vmap_log_derivative_matrix(
    model: NeuralQuantumState,
    configurations: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """使用批量 VJP 计算逐样本对数 Jacobian。

    ``chunk_size`` 用于限制 VJP 基矩阵及对应前向计算图的大小。返回结果的
    约定与 :func:`sequential_log_derivative_matrix` 完全一致。
    """
    if configurations.ndim < 2:
        raise ValueError("configurations must have a leading sample dimension")
    num_samples = configurations.shape[0]
    if num_samples == 0:
        raise ValueError("configurations must contain at least one sample")
    if chunk_size is None:
        chunk_size = num_samples
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer or None")

    parameters = tuple(model.parameters())
    chunks = []
    for configuration_chunk in configurations.split(chunk_size, dim=0):
        log_values = model.log_psi(configuration_chunk)
        if log_values.shape != configuration_chunk.shape[:-1]:
            raise ValueError("log_psi must return one scalar per configuration")
        log_values = log_values.reshape(-1)
        basis = torch.eye(
            log_values.numel(),
            dtype=log_values.real.dtype,
            device=log_values.device,
        )

        def vector_jacobian_product(cotangent: torch.Tensor):
            return torch.autograd.grad(
                log_values.real,
                parameters,
                grad_outputs=cotangent,
                create_graph=False,
            )

        batched_gradients = torch.vmap(vector_jacobian_product)(basis)
        chunks.append(
            torch.cat(
                [gradient.flatten(start_dim=1) for gradient in batched_gradients],
                dim=1,
            ).conj()
        )
    return torch.cat(chunks, dim=0)


class LogJacobian:
    """仅供需要逐样本导数的优化器使用的可配置对数 Jacobian 策略。

    模型提供 ``log_derivatives`` 时，``auto`` 优先使用解析实现；否则使用
    通用 ``vmap`` 实现。
    """

    _METHODS = frozenset({"sequential", "vmap", "analytic", "auto"})

    def __init__(
        self,
        method: LogJacobianMethod = "auto",
        *,
        chunk_size: int | None = None,
    ) -> None:
        if method not in self._METHODS:
            choices = ", ".join(sorted(self._METHODS))
            raise ValueError(f"unknown log-Jacobian method {method!r}; choose {choices}")
        if (
            chunk_size is not None
            and (
                isinstance(chunk_size, bool)
                or not isinstance(chunk_size, int)
                or chunk_size <= 0
            )
        ):
            raise ValueError("chunk_size must be a positive integer or None")
        self.method = method
        self.chunk_size = chunk_size

    def __call__(
        self, model: NeuralQuantumState, configurations: torch.Tensor
    ) -> torch.Tensor | RealLogDerivativeParts:
        method = self.method
        if isinstance(model, AmplitudePhaseNQS):
            analytic_parts = getattr(model, "log_derivative_parts", None)
            if method == "auto":
                method = "analytic" if callable(analytic_parts) else "vmap"
            if method == "sequential":
                result = sequential_real_log_derivative_parts(
                    model, configurations
                )
            elif method == "vmap":
                result = vmap_real_log_derivative_parts(
                    model, configurations, chunk_size=self.chunk_size
                )
            else:
                if not callable(analytic_parts):
                    raise TypeError(
                        f"{type(model).__name__} does not provide "
                        "log_derivative_parts()"
                    )
                amplitude, phase = analytic_parts(configurations)
                result = RealLogDerivativeParts(amplitude, phase)
            expected = (
                configurations.shape[0],
                sum(parameter.numel() for parameter in model.parameters()),
            )
            if (
                result.log_amplitude.shape != expected
                or result.phase.shape != expected
            ):
                raise ValueError(
                    "real log-Jacobian strategy returned an invalid shape"
                )
            return result
        analytic = getattr(model, "log_derivatives", None)
        if method == "auto":
            method = "analytic" if callable(analytic) else "vmap"

        if method == "sequential":
            result = sequential_log_derivative_matrix(model, configurations)
        elif method == "vmap":
            result = vmap_log_derivative_matrix(
                model, configurations, chunk_size=self.chunk_size
            )
        else:
            if not callable(analytic):
                raise TypeError(
                    f"{type(model).__name__} does not provide log_derivatives()"
                )
            result = analytic(configurations)

        expected_shape = (
            configurations.shape[0],
            sum(parameter.numel() for parameter in model.parameters()),
        )
        if result.shape != expected_shape:
            raise ValueError(
                "log-Jacobian strategy returned shape "
                f"{tuple(result.shape)}, expected {expected_shape}"
            )
        return result
