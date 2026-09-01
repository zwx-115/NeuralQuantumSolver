from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SpinHalfHilbert:
    """Unconstrained spin-1/2 product basis using Pauli-Z eigenvalues."""

    num_sites: int

    def __post_init__(self) -> None:
        if self.num_sites < 1:
            raise ValueError("num_sites must be positive")

    @property
    def size(self) -> int:
        return 1 << self.num_sites

    def all_states(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return states in legacy binary order: bit 0 maps to +1."""
        indices = torch.arange(self.size, device=device, dtype=torch.int64)
        shifts = torch.arange(self.num_sites - 1, -1, -1, device=device, dtype=torch.int64)
        bits = torch.bitwise_and(indices[:, None] >> shifts, 1)
        return (1 - 2 * bits).to(torch.int8)

    def state_to_index(self, states: torch.Tensor) -> torch.Tensor:
        self.validate(states)
        bits = (1 - states.to(torch.int64)) // 2
        weights = 1 << torch.arange(
            self.num_sites - 1, -1, -1, device=states.device, dtype=torch.int64
        )
        return torch.sum(bits * weights, dim=-1)

    def validate(self, states: torch.Tensor) -> None:
        if states.shape[-1] != self.num_sites:
            raise ValueError(f"expected last dimension {self.num_sites}, got {states.shape[-1]}")
        if not bool(torch.all((states == 1) | (states == -1))):
            raise ValueError("spin configurations must contain only +1 and -1")
