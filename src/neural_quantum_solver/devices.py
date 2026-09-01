from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar
import torch


SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "musa"})
T = TypeVar("T")


def expand_device_names(device: str | torch.device, num_gpus: int) -> tuple[str, ...]:
    """将 ``cuda:1`` 或 ``musa`` 展开为连续的同后端设备。"""
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    text = str(device).lower()
    backend, separator, index_text = text.partition(":")
    if backend not in SUPPORTED_DEVICE_TYPES:
        raise ValueError("device must use the cpu, cuda, or musa backend")
    if backend == "cpu":
        if separator or num_gpus != 1:
            raise ValueError("CPU execution requires device='cpu' and num_gpus=1")
        return ("cpu",)
    try:
        first_index = int(index_text) if separator else 0
    except ValueError as error:
        raise ValueError(f"invalid device index in {text!r}") from error
    if first_index < 0:
        raise ValueError("device index must be non-negative")
    return tuple(f"{backend}:{first_index + offset}" for offset in range(num_gpus))


def _backend_module(backend: str):
    module = getattr(torch, backend, None)
    if module is None:
        raise RuntimeError(
            f"PyTorch does not expose torch.{backend}; use the matching GPU image"
        )
    return module


@dataclass(frozen=True)
class DeviceMesh:
    """用于基态数据并行的单主机同构设备组。"""

    devices: tuple[torch.device, ...]

    @classmethod
    def create(
        cls, device: str | torch.device, num_gpus: int = 1
    ) -> "DeviceMesh":
        names = expand_device_names(device, num_gpus)
        backend = names[0].split(":", 1)[0]
        if backend != "cpu":
            module = _backend_module(backend)
            if not module.is_available():
                raise RuntimeError(f"requested {backend} backend is not available")
            available = module.device_count()
            last_index = int(names[-1].split(":", 1)[1])
            if last_index >= available:
                raise ValueError(
                    f"requested through {names[-1]}, but only {available} {backend} "
                    "device(s) are visible"
                )
        return cls(tuple(torch.device(name) for name in names))

    @property
    def primary(self) -> torch.device:
        return self.devices[0]

    @property
    def backend(self) -> str:
        return self.primary.type

    @property
    def num_devices(self) -> int:
        return len(self.devices)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    """在 CPU、CUDA 或 MUSA 上创建可复现的随机数生成器。"""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def run_on_device(
    device: torch.device,
    operation: Callable[..., T],
    *args,
    **kwargs,
) -> T:
    """在线程内显式激活目标加速器后执行操作。"""
    if device.type == "cpu":
        return operation(*args, **kwargs)
    backend = _backend_module(device.type)
    with backend.device(device):
        return operation(*args, **kwargs)


def transfer_tensor(tensor: torch.Tensor, target: torch.device) -> torch.Tensor:
    """跨加速器复制时通过 CPU 暂存，避免依赖设备 P2P 支持。"""
    target = torch.device(target)
    if tensor.device == target:
        return tensor
    if tensor.device.type != "cpu" and target.type != "cpu":
        tensor = tensor.to("cpu")
    return run_on_device(target, tensor.to, target)


def synchronize_devices(devices: tuple[torch.device, ...]) -> None:
    """同步加速器任务，保证各阶段墙钟计时准确。"""
    for device in devices:
        if device.type != "cpu":
            _backend_module(device.type).synchronize(device)
