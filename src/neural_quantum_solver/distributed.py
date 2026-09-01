from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class ParallelContext:
    """单卡或一进程一卡的分布式通信上下文。"""

    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str | None = None
    owns_process_group: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        device_type: str = "cuda",
        backend: str | None = None,
    ) -> "ParallelContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size == 1:
            device = torch.device(
                f"{device_type}:{local_rank}" if device_type != "cpu" else "cpu"
            )
            if device_type != "cpu":
                getattr(torch, device_type).set_device(device)
            return cls(device=device, local_rank=local_rank)
        if device_type != "cuda":
            raise ValueError("torchrun multi-process mode currently requires CUDA")
        selected_backend = backend or "nccl"
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if dist.is_initialized():
            raise RuntimeError("a process group is already initialized")
        dist.init_process_group(backend=selected_backend, init_method="env://")
        return cls(
            device=device,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=selected_backend,
            owns_process_group=True,
        )

    @classmethod
    def from_initialized(
        cls, device: torch.device | str, *, local_rank: int = 0
    ) -> "ParallelContext":
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed process group is not initialized")
        return cls(
            device=torch.device(device),
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=local_rank,
            backend=str(dist.get_backend()),
        )

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def parallel_mode(self) -> str:
        return self.backend or "distributed" if self.distributed else "single"

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def broadcast(self, tensor: torch.Tensor, *, source: int = 0) -> torch.Tensor:
        if self.distributed:
            dist.broadcast(tensor, src=source)
        return tensor

    def broadcast_model(self, model: nn.Module, *, source: int = 0) -> None:
        if not self.distributed:
            return
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=source)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=source)

    def all_gather_object(self, value: object) -> list[object]:
        if not self.distributed:
            return [value]
        gathered: list[object] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
