import torch

from neural_quantum_solver.qgt import quantum_geometric_tensor, solve_sr


def test_qgt_is_hermitian_positive_semidefinite():
    derivatives = torch.randn(12, 5, dtype=torch.complex128)
    probabilities = torch.softmax(torch.randn(12, dtype=torch.float64), dim=0)
    qgt = quantum_geometric_tensor(derivatives, probabilities)
    assert torch.allclose(qgt, qgt.mH, atol=1e-12, rtol=1e-12)
    assert torch.linalg.eigvalsh(qgt).min() > -1e-12
    update, diagnostics = solve_sr(
        qgt,
        torch.randn(5, dtype=torch.complex128),
        learning_rate=0.1,
        regularization=1e-3,
    )
    assert update.shape == (5,)
    assert diagnostics.effective_rank > 0


def test_sr_uses_direct_linear_solve(monkeypatch):
    qgt = torch.eye(3, dtype=torch.complex128)
    force = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex128)
    original_solve = torch.linalg.solve
    calls = 0

    def tracking_solve(matrix, right_hand_side):
        nonlocal calls
        calls += 1
        return original_solve(matrix, right_hand_side)

    monkeypatch.setattr(torch.linalg, "solve", tracking_solve)
    update, _ = solve_sr(
        qgt,
        force,
        learning_rate=0.1,
        regularization=1e-3,
    )

    expected = original_solve(qgt + 1e-3 * torch.eye(3), -0.1 * force)
    assert calls == 1
    assert torch.allclose(update, expected)
