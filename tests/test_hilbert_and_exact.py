import torch

from neural_quantum_solver import ExactDiagonalizer, SpinHalfHilbert
from neural_quantum_solver.systems import heisenberg, tilted_field_ising, xxz
from neural_quantum_solver.operators import PauliHamiltonian, PauliTerm


def test_legacy_basis_order_and_round_trip():
    hilbert = SpinHalfHilbert(2)
    states = hilbert.all_states()
    expected = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.int8)
    assert torch.equal(states, expected)
    assert torch.equal(hilbert.state_to_index(states), torch.arange(4))


def test_one_spin_ising_has_analytic_ground_energy():
    system = tilted_field_ising(1, coupling=0.0, field_x=0.3, field_z=0.4)
    result = ExactDiagonalizer().ground_state(system)
    assert torch.allclose(result.energy, torch.tensor(-0.5, dtype=result.energy.dtype))


def test_multiple_physical_systems_are_hermitian():
    solver = ExactDiagonalizer()
    for system in (tilted_field_ising(4), heisenberg(4), xxz(4, anisotropy=1.7)):
        matrix = solver.matrix(system)
        assert torch.allclose(matrix, matrix.mH, atol=1e-12, rtol=1e-12)
        result = solver.ground_state(system)
        assert torch.allclose(
            torch.linalg.vector_norm(result.state), torch.tensor(1.0, dtype=torch.float64)
        )


def test_two_spin_heisenberg_singlet_energy():
    result = ExactDiagonalizer().ground_state(heisenberg(2, coupling=1.0))
    assert torch.allclose(result.energy, torch.tensor(-3.0, dtype=result.energy.dtype))


def test_single_pauli_y_uses_dense_row_column_convention():
    hilbert = SpinHalfHilbert(1)
    operator = PauliHamiltonian(1, [PauliTerm(1.0, ((0, "Y"),))])
    expected = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    assert torch.equal(operator.dense_matrix(hilbert), expected)
