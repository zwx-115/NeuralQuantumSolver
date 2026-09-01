from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class LogPsiParts:
    """用两个实数张量表示 log(psi)=log_amplitude+i*phase。"""

    log_amplitude: torch.Tensor
    phase: torch.Tensor


class NeuralQuantumState(nn.Module, ABC):
    @abstractmethod
    def log_psi(self, configurations: torch.Tensor) -> torch.Tensor:
        """返回形状为 configurations.shape[:-1] 的复数对数波函数。"""

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        return self.log_psi(configurations)

    def log_psi_parts(self, configurations: torch.Tensor) -> LogPsiParts:
        values = self.log_psi(configurations)
        return LogPsiParts(values.real, values.imag)


class AmplitudePhaseNQS(NeuralQuantumState, ABC):
    """在加速器上使用两个实数张量表示波函数的 NQS 基类。"""

    @abstractmethod
    def log_psi_parts(self, configurations: torch.Tensor) -> LogPsiParts:
        pass

    def log_psi(self, configurations: torch.Tensor) -> torch.Tensor:
        parts = self.log_psi_parts(configurations)
        return torch.complex(parts.log_amplitude, parts.phase)

    def log_amplitude(self, configurations: torch.Tensor) -> torch.Tensor:
        return self.log_psi_parts(configurations).log_amplitude

    def forward(self, configurations: torch.Tensor) -> LogPsiParts:
        return self.log_psi_parts(configurations)


def is_amplitude_phase_model(model: NeuralQuantumState) -> bool:
    return isinstance(model, AmplitudePhaseNQS)


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("real-pair NQS parameters require torch.float32 or torch.float64")
    return dtype


class RealRBM(nn.Module):
    """用作振幅网络或相位网络的单个实数 RBM。"""

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        seed: int | None = None,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        dtype = _real_dtype(dtype)
        if seed is not None:
            torch.manual_seed(seed)
        self.num_visible, self.num_hidden = num_visible, num_hidden
        self.visible_bias = nn.Parameter(
            torch.empty(num_visible, dtype=dtype, device=device)
        )
        self.hidden_bias = nn.Parameter(
            torch.empty(num_hidden, dtype=dtype, device=device)
        )
        self.weight = nn.Parameter(
            torch.empty(num_hidden, num_visible, dtype=dtype, device=device)
        )
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=init_std)

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        x = configurations.to(dtype=self.weight.dtype, device=self.weight.device)
        theta = x @ self.weight.mT + self.hidden_bias
        visible = (x @ self.visible_bias[:, None]).squeeze(-1)
        absolute = torch.abs(theta)
        log_2cosh = absolute + torch.log1p(torch.exp(-2 * absolute))
        return visible + log_2cosh.sum(dim=-1)

    def log_derivatives(self, configurations: torch.Tensor) -> torch.Tensor:
        x = configurations.to(dtype=self.weight.dtype, device=self.weight.device)
        theta = x @ self.weight.mT + self.hidden_bias
        hidden = torch.tanh(theta)
        return torch.cat(
            (
                x,
                hidden,
                (hidden[:, :, None] * x[:, None, :]).flatten(start_dim=1),
            ),
            dim=1,
        )


class AmplitudePhaseRBM(AmplitudePhaseNQS):
    """使用两个独立实数 RBM 分别表示对数振幅和相位。"""

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        seed: int | None = None,
        init_std: float = 0.01,
        phase_init_std: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_visible, self.num_hidden = num_visible, num_hidden
        self.amplitude = RealRBM(
            num_visible,
            num_hidden,
            dtype=dtype,
            device=device,
            seed=seed,
            init_std=init_std,
        )
        self.phase = RealRBM(
            num_visible,
            num_hidden,
            dtype=dtype,
            device=device,
            seed=None if seed is None else seed + 1,
            init_std=phase_init_std,
        )

    def log_psi_parts(self, configurations: torch.Tensor) -> LogPsiParts:
        return LogPsiParts(
            self.amplitude(configurations),
            self.phase(configurations),
        )

    def log_amplitude(self, configurations: torch.Tensor) -> torch.Tensor:
        return self.amplitude(configurations)

    def log_derivative_parts(
        self, configurations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        amplitude = self.amplitude.log_derivatives(configurations)
        phase = self.phase.log_derivatives(configurations)
        amplitude_zeros = torch.zeros_like(amplitude)
        phase_zeros = torch.zeros_like(phase)
        return (
            torch.cat((amplitude, phase_zeros), dim=1),
            torch.cat((amplitude_zeros, phase), dim=1),
        )


def _real_fnn(
    layer_sizes: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    seed: int | None,
) -> nn.Sequential:
    dtype = _real_dtype(dtype)
    if len(layer_sizes) < 2 or layer_sizes[-1] != 1:
        raise ValueError("layer_sizes must end in one real output")
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
    return nn.Sequential(*layers)


class AmplitudePhaseFNN(AmplitudePhaseNQS):
    """使用两个独立实数前馈网络分别表示振幅和相位。"""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_sizes = tuple(layer_sizes)
        self.amplitude = _real_fnn(
            layer_sizes, dtype=dtype, device=device, seed=seed
        )
        self.phase = _real_fnn(
            layer_sizes,
            dtype=dtype,
            device=device,
            seed=None if seed is None else seed + 1,
        )

    def log_psi_parts(self, configurations: torch.Tensor) -> LogPsiParts:
        parameter = next(self.parameters())
        x = configurations.to(dtype=parameter.dtype, device=parameter.device)
        return LogPsiParts(
            self.amplitude(x).squeeze(-1),
            self.phase(x).squeeze(-1),
        )

    def log_amplitude(self, configurations: torch.Tensor) -> torch.Tensor:
        parameter = next(self.parameters())
        x = configurations.to(dtype=parameter.dtype, device=parameter.device)
        return self.amplitude(x).squeeze(-1)


class AmplitudePhaseTable(AmplitudePhaseNQS):
    """用于小系统验证的完备双实数波函数表。"""

    def __init__(
        self,
        num_sites: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        dtype = _real_dtype(dtype)
        self.num_sites = num_sites
        self.log_amplitudes = nn.Parameter(
            torch.zeros(1 << num_sites, dtype=dtype, device=device)
        )
        self.phases = nn.Parameter(
            torch.zeros(1 << num_sites, dtype=dtype, device=device)
        )

    def _indices(self, configurations: torch.Tensor) -> torch.Tensor:
        bits = (1 - configurations.to(torch.int64)) // 2
        weights = 1 << torch.arange(
            self.num_sites - 1,
            -1,
            -1,
            device=configurations.device,
            dtype=torch.int64,
        )
        return torch.sum(bits * weights, dim=-1)

    def log_psi_parts(self, configurations: torch.Tensor) -> LogPsiParts:
        indices = self._indices(configurations)
        return LogPsiParts(self.log_amplitudes[indices], self.phases[indices])

    def log_amplitude(self, configurations: torch.Tensor) -> torch.Tensor:
        return self.log_amplitudes[self._indices(configurations)]


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("NQS parameters require torch.complex64 or torch.complex128")
    return dtype


class ComplexRBM(NeuralQuantumState):
    """与旧版对数波函数约定一致的复数 RBM。"""

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
        # MUSA 当前不支持 ComplexFloat 的矩阵-向量（mv）内核。
        # 将向量临时扩展为单列矩阵，使 CUDA 和 MUSA 使用数学上等价的
        # 矩阵-矩阵（mm）路径，同时保留自动微分。
        visible = (x @ self.visible_bias[:, None]).squeeze(-1)
        # MUSA 还缺少复数 cosh/log 内核，因此使用等价的实数运算计算：
        # cosh(a+ib) = cosh(a)cos(b) + i*sinh(a)sin(b)。
        theta_real, theta_imag = theta.real, theta.imag
        cosh_real = torch.cosh(theta_real) * torch.cos(theta_imag)
        cosh_imag = torch.sinh(theta_real) * torch.sin(theta_imag)
        log_cosh_real = (
            torch.log(torch.tensor(2.0, dtype=theta_real.dtype, device=theta.device))
            + 0.5 * torch.log(cosh_real.square() + cosh_imag.square())
        )
        log_cosh_imag = torch.atan2(cosh_imag, cosh_real)
        return torch.complex(
            visible.real + log_cosh_real.sum(dim=-1),
            visible.imag + log_cosh_imag.sum(dim=-1),
        )

    def log_derivatives(self, configurations: torch.Tensor) -> torch.Tensor:
        """不调用复数自动微分内核，直接返回解析对数 Jacobian。"""
        x = configurations.to(
            dtype=self.weight.real.dtype, device=self.weight.device
        )
        theta = (
            x.to(self.weight.dtype) @ self.weight.mT + self.hidden_bias
        )
        denominator = torch.cosh(2 * theta.real) + torch.cos(2 * theta.imag)
        tanh_real = torch.sinh(2 * theta.real) / denominator
        tanh_imag = torch.sin(2 * theta.imag) / denominator
        visible_derivatives = torch.complex(x, torch.zeros_like(x))
        hidden_derivatives = torch.complex(tanh_real, tanh_imag)
        weight_derivatives = torch.complex(
            tanh_real[:, :, None] * x[:, None, :],
            tanh_imag[:, :, None] * x[:, None, :],
        )
        return torch.cat(
            (
                visible_derivatives,
                hidden_derivatives,
                weight_derivatives.flatten(start_dim=1),
            ),
            dim=1,
        )


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
    """用于小系统的完备变分波函数表和诊断参考。"""

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
    if isinstance(model, AmplitudePhaseNQS):
        output = model.log_psi_parts(batch)
        if (
            output.log_amplitude.shape != (2,)
            or output.phase.shape != (2,)
            or output.log_amplitude.is_complex()
            or output.phase.is_complex()
        ):
            raise ValueError("real-pair log_psi_parts must return two real scalars")
        if not output.log_amplitude.requires_grad or not output.phase.requires_grad:
            raise ValueError("log_psi_parts detached the model parameters")
        return
    output = model.log_psi(batch)
    if output.shape != (2,) or not output.is_complex():
        raise ValueError("log_psi must return one complex scalar per configuration")
    individual = torch.stack([model.log_psi(x[None])[0] for x in batch])
    if not torch.allclose(output, individual):
        raise ValueError("batched and individual log_psi evaluations differ")
    if not output.requires_grad:
        raise ValueError("log_psi detached the model parameters")
