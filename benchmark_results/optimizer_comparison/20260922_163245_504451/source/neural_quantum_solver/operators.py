from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch

from .hilbert import SpinHalfHilbert


@dataclass(frozen=True)
class PauliTerm:
    coefficient: complex
    operators: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        sites = [site for site, _ in self.operators]
        if len(sites) != len(set(sites)):
            raise ValueError("a Pauli term cannot act twice on the same site")
        if any(op not in {"I", "X", "Y", "Z"} for _, op in self.operators):
            raise ValueError("supported Pauli labels are I, X, Y, and Z")


@dataclass(frozen=True)
class Connections:
    states: torch.Tensor
    matrix_elements: torch.Tensor
    sample_indices: torch.Tensor


class PauliHamiltonian:
    """Sum of Pauli strings with explicit coefficients."""

    def __init__(self, num_sites: int, terms: Iterable[PauliTerm]) -> None:
        self.num_sites = num_sites
        self.terms = tuple(terms)
        for term in self.terms:
            if any(site < 0 or site >= num_sites for site, _ in term.operators):
                raise ValueError("Pauli term site is outside the Hilbert space")

    def connections(
        self, configurations: torch.Tensor, *, dtype: torch.dtype = torch.complex128
    ) -> Connections:
        """Return packed H[row, column] connections for each input row state."""
        if configurations.ndim != 2 or configurations.shape[1] != self.num_sites:
            raise ValueError("configurations must have shape [batch, num_sites]")
        device = configurations.device
        states_out, elements_out, samples_out = [], [], []
        batch = configurations.shape[0]
        sample_ids = torch.arange(batch, device=device, dtype=torch.int64)
        for term in self.terms:
            states = configurations.clone()
            values = torch.full((batch,), complex(term.coefficient), device=device, dtype=dtype)
            for site, operator in term.operators:
                spin = configurations[:, site].to(dtype)
                if operator == "Z":
                    values = values * spin
                elif operator == "X":
                    states[:, site] = -states[:, site]
                elif operator == "Y":
                    # Row convention: <z|Y|-z> = -i*z.
                    values = values * (-1j * spin)
                    states[:, site] = -states[:, site]
            states_out.append(states)
            elements_out.append(values)
            samples_out.append(sample_ids)
        if not states_out:
            return Connections(
                configurations[:0],
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.int64),
            )
        return Connections(
            torch.cat(states_out), torch.cat(elements_out), torch.cat(samples_out)
        )

    def dense_matrix(
        self,
        hilbert: SpinHalfHilbert,
        *,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if hilbert.num_sites != self.num_sites:
            raise ValueError("Hamiltonian and Hilbert-space sizes differ")
        states = hilbert.all_states(device=device)
        matrix = torch.zeros((hilbert.size, hilbert.size), dtype=dtype, device=device)
        conn = self.connections(states, dtype=dtype)
        columns = hilbert.state_to_index(conn.states)
        matrix.index_put_((conn.sample_indices, columns), conn.matrix_elements, accumulate=True)
        return matrix
