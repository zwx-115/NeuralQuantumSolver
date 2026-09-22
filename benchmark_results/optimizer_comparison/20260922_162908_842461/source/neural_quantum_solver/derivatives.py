from __future__ import annotations

from typing import Literal, Protocol
import torch

from .models import NeuralQuantumState


LogJacobianMethod = Literal["sequential", "vmap", "analytic", "auto"]


class LogJacobianStrategy(Protocol):
    """Callable strategy producing O[n, i] for SR-like optimizers."""

    def __call__(
        self, model: NeuralQuantumState, configurations: torch.Tensor
    ) -> torch.Tensor: ...


def _flatten_tensors(tensors) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def sequential_log_derivative_matrix(
    model: NeuralQuantumState, configurations: torch.Tensor
) -> torch.Tensor:
    """Reference per-sample log-Jacobian for holomorphic complex models."""
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
        # PyTorch returns the conjugate Wirtinger gradient of a real scalar.
        rows.append(_flatten_tensors(gradients).conj())
    return torch.stack(rows)


def vmap_log_derivative_matrix(
    model: NeuralQuantumState,
    configurations: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute the per-sample log-Jacobian using batched VJPs.

    ``chunk_size`` bounds the VJP basis matrix and the corresponding forward
    graph.  The returned convention is identical to
    :func:`sequential_log_derivative_matrix`.
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
    """Configurable log-Jacobian strategy used only by optimizers that need it.

    ``auto`` selects a model-provided ``log_derivatives`` method when present,
    and otherwise uses the generic ``vmap`` implementation.
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
    ) -> torch.Tensor:
        method = self.method
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

