import torch

from neural_quantum_solver.overlap import (
    direct_overlap_loss, legacy_ratio_loss, log_domain_ratio_loss,
)


def _random_state(size, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(size, generator=generator, dtype=torch.complex128) + (0.2 + 0.1j)


def test_direct_overlap_matches_ratio_exact_sum():
    phi, chi = _random_state(16, 1), _random_state(16, 2)
    hermitian = torch.randn(16, 16, dtype=torch.complex128)
    hermitian = hermitian + hermitian.mH
    unitary = torch.matrix_exp(-0.03j * hermitian)
    direct = direct_overlap_loss(phi, chi, unitary)
    ratio = legacy_ratio_loss(phi, chi, unitary)
    assert torch.allclose(direct.loss, ratio.loss, atol=2e-12, rtol=2e-12)
    assert direct.finite_flag and ratio.finite_flag
    stable = log_domain_ratio_loss(phi * 1e120, chi * 1e-120, unitary)
    assert torch.allclose(stable.loss, direct.loss, atol=2e-12, rtol=2e-12)


def test_loss_is_invariant_and_A_B_transform_reciprocally_under_gauge():
    phi, chi = _random_state(8, 4), _random_state(8, 5)
    unitary = torch.eye(8, dtype=torch.complex128)
    scale = 1e3 * torch.exp(torch.tensor(0.4j, dtype=torch.complex128))
    base = legacy_ratio_loss(phi, chi, unitary)
    shifted = legacy_ratio_loss(phi, scale * chi, unitary)
    assert torch.allclose(base.loss, shifted.loss, atol=1e-11, rtol=1e-11)
    assert torch.allclose(shifted.A, base.A / scale, atol=1e-11, rtol=1e-11)
    assert torch.allclose(shifted.B, base.B * scale, atol=1e-9, rtol=1e-11)
