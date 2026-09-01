import torch

from neural_quantum_solver import ComplexRBM
from neural_quantum_solver.evolution import (
    exact_ground_state_target, freeze_model_target, nqs_ground_state_target,
)
from neural_quantum_solver.qgt import quantum_geometric_tensor, solve_sr
from neural_quantum_solver.systems import tilted_field_ising


def test_qgt_is_hermitian_positive_semidefinite():
    derivatives = torch.randn(12, 5, dtype=torch.complex128)
    probabilities = torch.softmax(torch.randn(12, dtype=torch.float64), dim=0)
    qgt = quantum_geometric_tensor(derivatives, probabilities)
    assert torch.allclose(qgt, qgt.mH, atol=1e-12, rtol=1e-12)
    assert torch.linalg.eigvalsh(qgt).min() > -1e-12
    update, diagnostics = solve_sr(
        qgt, torch.randn(5, dtype=torch.complex128),
        learning_rate=0.1, regularization=1e-3,
    )
    assert update.shape == (5,)
    assert diagnostics.effective_rank > 0


def test_ground_state_is_valid_evolution_target_and_model_target_is_frozen():
    system = tilted_field_ising(3)
    target = exact_ground_state_target(system)
    assert target.time == 0.0
    assert torch.allclose(
        torch.linalg.vector_norm(target.amplitudes), torch.tensor(1.0, dtype=torch.float64)
    )
    model = ComplexRBM(3, 6, seed=1)
    frozen = freeze_model_target(model, system)
    nqs_target = nqs_ground_state_target(model, system)
    assert nqs_target.amplitudes.shape == target.amplitudes.shape
    assert not nqs_target.amplitudes.requires_grad
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
