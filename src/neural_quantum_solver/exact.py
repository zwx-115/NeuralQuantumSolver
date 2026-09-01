from __future__ import annotations

from dataclasses import dataclass
import torch
from .systems import PhysicalSystem


@dataclass(frozen=True)
class EigenResult:
    energy: torch.Tensor
    state: torch.Tensor
    eigenvalues: torch.Tensor


class ExactDiagonalizer:
    """Dense Hermitian eigensolver for small reference systems."""

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu",
        max_sites: int = 14,
        hermitian_tolerance: float = 1e-11,
    ) -> None:
        if dtype not in (torch.complex64, torch.complex128):
            raise TypeError("exact diagonalization requires a complex dtype")
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_sites = max_sites
        self.hermitian_tolerance = hermitian_tolerance

    def matrix(self, system: PhysicalSystem) -> torch.Tensor:
        if system.hilbert.num_sites > self.max_sites:
            raise ValueError(f"dense exact diagonalization is limited to {self.max_sites} sites")
        matrix = system.hamiltonian.dense_matrix(
            system.hilbert, dtype=self.dtype, device=self.device
        )
        error = torch.max(torch.abs(matrix - matrix.mH)).item()
        if error > self.hermitian_tolerance:
            raise ValueError(f"Hamiltonian is not Hermitian; max error={error:.3e}")
        return matrix

    def diagonalize(self, system: PhysicalSystem) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(self.matrix(system))

    def ground_state(self, system: PhysicalSystem) -> EigenResult:
        eigenvalues, eigenvectors = self.diagonalize(system)
        return EigenResult(eigenvalues[0], eigenvectors[:, 0], eigenvalues)
