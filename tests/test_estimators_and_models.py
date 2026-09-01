import torch

from neural_quantum_solver import ComplexFNN, ComplexRBM, LogAmplitudeTable
from neural_quantum_solver.estimators import exact_energy, exact_state
from neural_quantum_solver.models import validate_nqs
from neural_quantum_solver.systems import tilted_field_ising


def test_all_models_implement_log_psi_contract():
    for model in (
        ComplexRBM(3, 6, seed=1), ComplexFNN([3, 8, 1], seed=2), LogAmplitudeTable(3)
    ):
        validate_nqs(model, 3)


def test_exact_energy_matches_dense_expectation_and_is_gauge_invariant():
    system = tilted_field_ising(3)
    model = LogAmplitudeTable(3)
    with torch.no_grad():
        model.log_amplitudes.copy_(torch.randn(8, dtype=torch.complex128))
    estimate = exact_energy(model, system)
    state = exact_state(model, system).amplitudes
    matrix = system.hamiltonian.dense_matrix(system.hilbert)
    reference = torch.vdot(state, matrix @ state).real
    assert torch.allclose(estimate.energy, reference, atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        model.log_amplitudes.add_(3.2 + 0.7j)
    shifted = exact_energy(model, system)
    assert torch.allclose(estimate.energy, shifted.energy, atol=1e-12, rtol=1e-12)


def test_ground_state_objective_has_gradients():
    model = ComplexRBM(3, 6, seed=3)
    estimate = exact_energy(model, tilted_field_ising(3))
    estimate.energy.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
