from __future__ import annotations

from dataclasses import dataclass
from .hilbert import SpinHalfHilbert
from .lattice import Graph
from .operators import PauliHamiltonian, PauliTerm


@dataclass(frozen=True)
class PhysicalSystem:
    hilbert: SpinHalfHilbert
    lattice: Graph
    hamiltonian: PauliHamiltonian
    name: str


def _pair_terms(edges, coupling_x: float, coupling_y: float, coupling_z: float):
    terms: list[PauliTerm] = []
    for i, j in edges:
        if coupling_x:
            terms.append(PauliTerm(coupling_x, ((i, "X"), (j, "X"))))
        if coupling_y:
            terms.append(PauliTerm(coupling_y, ((i, "Y"), (j, "Y"))))
        if coupling_z:
            terms.append(PauliTerm(coupling_z, ((i, "Z"), (j, "Z"))))
    return terms


def tilted_field_ising(
    num_sites: int,
    *,
    coupling: float = 1.0,
    field_x: float = 0.5,
    field_z: float = 0.5,
    periodic: bool = False,
) -> PhysicalSystem:
    """H = J sum ZZ - hx sum X - hz sum Z, in Pauli convention."""
    lattice = Graph.chain(num_sites, periodic=periodic)
    terms = [PauliTerm(coupling, ((i, "Z"), (j, "Z"))) for i, j in lattice.edges]
    terms += [PauliTerm(-field_x, ((i, "X"),)) for i in range(num_sites)]
    terms += [PauliTerm(-field_z, ((i, "Z"),)) for i in range(num_sites)]
    return PhysicalSystem(
        SpinHalfHilbert(num_sites), lattice, PauliHamiltonian(num_sites, terms),
        "tilted_field_ising",
    )


def heisenberg(
    num_sites: int, *, coupling: float = 1.0, periodic: bool = False
) -> PhysicalSystem:
    lattice = Graph.chain(num_sites, periodic=periodic)
    terms = _pair_terms(lattice.edges, coupling, coupling, coupling)
    return PhysicalSystem(
        SpinHalfHilbert(num_sites), lattice, PauliHamiltonian(num_sites, terms), "heisenberg"
    )


def xxz(
    num_sites: int,
    *,
    coupling_xy: float = 1.0,
    anisotropy: float = 1.0,
    periodic: bool = False,
) -> PhysicalSystem:
    lattice = Graph.chain(num_sites, periodic=periodic)
    terms = _pair_terms(lattice.edges, coupling_xy, coupling_xy, anisotropy)
    return PhysicalSystem(
        SpinHalfHilbert(num_sites), lattice, PauliHamiltonian(num_sites, terms), "xxz"
    )


def j1j2_heisenberg(
    num_sites: int,
    *,
    j1: float = 1.0,
    j2: float = 0.5,
    periodic: bool = False,
) -> PhysicalSystem:
    lattice = Graph.chain(num_sites, periodic=periodic)
    nearest = list(lattice.edges)
    if periodic:
        next_nearest = sorted(
            {tuple(sorted((i, (i + 2) % num_sites))) for i in range(num_sites)}
        )
    else:
        next_nearest = [(i, i + 2) for i in range(num_sites - 2)]
    terms = _pair_terms(nearest, j1, j1, j1)
    terms += _pair_terms(next_nearest, j2, j2, j2)
    return PhysicalSystem(
        SpinHalfHilbert(num_sites), lattice, PauliHamiltonian(num_sites, terms),
        "j1j2_heisenberg",
    )
