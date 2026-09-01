from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import torch
from torch import nn


class NeuralQuantumState(nn.Module, ABC):
    @abstractmethod
    def log_psi(self, configurations: torch.Tensor) -> torch.Tensor:
        """Return complex log amplitudes with shape configurations.shape[:-1]."""

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        return self.log_psi(configurations)


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("NQS parameters require torch.complex64 or torch.complex128")
    return dtype


class ComplexRBM(NeuralQuantumState):
    """Complex RBM matching the legacy log-wavefunction convention."""

    def __init__(
        self, num_visible: int, num_hidden: int, *,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu", seed: int | None = None,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        dtype = _complex_dtype(dtype)
        if seed is not None:
            torch.manual_seed(seed)
        self.num_visible, self.num_hidden = num_visible, num_hidden
        self.visible_bias = nn.Parameter(torch.empty(num_visible, dtype=dtype, device=device))
        self.hidden_bias = nn.Parameter(torch.empty(num_hidden, dtype=dtype, device=device))
        self.weight = nn.Parameter(torch.empty(num_hidden, num_visible, dtype=dtype, device=device))
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=init_std)

    def log_psi(self, configurations: torch.Tensor) -> torch.Tensor:
        x = configurations.to(dtype=self.weight.dtype, device=self.weight.device)
        theta = x @ self.weight.mT + self.hidden_bias
        # MUSA does not currently support the ComplexFloat matrix-vector (mv)
        # kernel. A one-column matrix uses the mathematically identical mm path
        # on both CUDA and MUSA and preserves autograd.
        visible = (x @ self.visible_bias[:, None]).squeeze(-1)
        return visible + torch.log(2 * torch.cosh(theta)).sum(dim=-1)


class ComplexFNN(NeuralQuantumState):
    def __init__(
        self, layer_sizes: Sequence[int], *, dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu", seed: int | None = None,
    ) -> None:
        super().__init__()
        dtype = _complex_dtype(dtype)
        if len(layer_sizes) < 2 or layer_sizes[-1] != 1:
            raise ValueError("layer_sizes must end in one complex output")
        if seed is not None:
            torch.manual_seed(seed)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            linear = nn.Linear(left, right, dtype=dtype, device=device)
            nn.init.normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def log_psi(self, configurations: torch.Tensor) -> torch.Tensor:
        parameter = next(self.parameters())
        x = configurations.to(dtype=parameter.dtype, device=parameter.device)
        return self.network(x).squeeze(-1)


class LogAmplitudeTable(NeuralQuantumState):
    """Fully expressive small-system ansatz and diagnostic reference."""

    def __init__(
        self, num_sites: int, *, dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.num_sites = num_sites
        self.log_amplitudes = nn.Parameter(
            torch.zeros(1 << num_sites, dtype=_complex_dtype(dtype), device=device)
        )

    def log_psi(self, configurations: torch.Tensor) -> torch.Tensor:
        bits = (1 - configurations.to(torch.int64)) // 2
        weights = 1 << torch.arange(
            self.num_sites - 1, -1, -1, device=configurations.device, dtype=torch.int64
        )
        return self.log_amplitudes[torch.sum(bits * weights, dim=-1)]


def validate_nqs(model: NeuralQuantumState, num_sites: int) -> None:
    device = next(model.parameters()).device
    batch = torch.tensor([[1] * num_sites, [-1] * num_sites], dtype=torch.int8, device=device)
    output = model.log_psi(batch)
    if output.shape != (2,) or not output.is_complex():
        raise ValueError("log_psi must return one complex scalar per configuration")
    individual = torch.stack([model.log_psi(x[None])[0] for x in batch])
    if not torch.allclose(output, individual):
        raise ValueError("batched and individual log_psi evaluations differ")
    if not output.requires_grad:
        raise ValueError("log_psi detached the model parameters")
