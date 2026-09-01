from __future__ import annotations

from dataclasses import dataclass
import torch


SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "musa"})


def expand_device_names(device: str | torch.device, num_gpus: int) -> tuple[str, ...]:
    """Expand ``cuda:1``/``musa`` into consecutive same-backend devices."""
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
    """One-host homogeneous device group used by ground-state data parallelism."""

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
    """Create a reproducible generator on CPU, CUDA, or MUSA."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def synchronize_devices(devices: tuple[torch.device, ...]) -> None:
    """Synchronize accelerator work so wall-clock stage timings are accurate."""
    for device in devices:
        if device.type != "cpu":
            _backend_module(device.type).synchronize(device)
