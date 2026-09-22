"""二阶量子信赖域的独立数值检验。"""

from copy import deepcopy
import pytest
import torch

from neural_quantum_solver import ComplexRBM, LogAmplitudeTable, tilted_field_ising
from neural_quantum_solver.estimators import exact_state
from neural_quantum_solver.solvers import (
    CurvatureSettings, FunctionalWavefunction, OverlapCurvature,
    solve_trust_region, optimize_overlap_curvature,
)


def test_legacy_diagnostic_imports_forward_to_projection_solvers():
    from neural_quantum_solver.diagnostics.overlap_curvature import OverlapCurvature as LegacyCurvature
    from neural_quantum_solver.diagnostics.overlap_sr import OverlapSRSettings as LegacySRSettings
    from neural_quantum_solver.solvers import OverlapSRSettings
    assert LegacyCurvature is OverlapCurvature
    assert LegacySRSettings is OverlapSRSettings


def test_real_parameter_gradient_hessian_and_metric():
    system = tilted_field_ising(2)
    model = ComplexRBM(2, 2, seed=17)
    wave = FunctionalWavefunction(model, system)
    x = wave.pack().requires_grad_()
    target = exact_state(ComplexRBM(2, 2, seed=5), system).amplitudes.detach()
    def loss(y):
        psi = wave(y)
        r = psi - target * torch.vdot(target, psi)
        return torch.vdot(r, r).real
    g = torch.autograd.grad(loss(x), x, create_graph=True)[0]
    v = torch.linspace(-1, 1, x.numel(), dtype=x.dtype)
    v /= v.norm()
    hv = torch.autograd.grad(g @ v, x)[0]
    eps = 1e-4
    numeric = (loss(x + eps*v) - loss(x - eps*v)) / (2*eps)
    numeric2 = (loss(x + eps*v) - 2*loss(x) + loss(x - eps*v)) / eps**2
    torch.testing.assert_close(numeric, g @ v, atol=1e-8, rtol=1e-5)
    torch.testing.assert_close(numeric2, v @ hv, atol=1e-7, rtol=1e-4)
    psi, jv = torch.func.jvp(wave, (x.detach(),), (v,))
    horizontal = jv - psi * torch.vdot(psi, jv)
    distance = 1 - abs(torch.vdot(psi, wave(x.detach()+eps*v)))**2
    torch.testing.assert_close(distance/eps**2, torch.vdot(horizontal, horizontal).real,
                               atol=1e-5, rtol=1e-3)
    wave.assign(wave.pack())
    torch.testing.assert_close(wave(wave.pack()), exact_state(model, system).amplitudes)


def test_trust_region_negative_curvature_hard_case():
    h = torch.diag(torch.tensor([-2., 1.], dtype=torch.float64))
    g = torch.tensor([0., 0.2], dtype=torch.float64)
    z = solve_trust_region(g, h, torch.eye(2, dtype=h.dtype), 0.4)
    assert z.norm() <= 0.4 + 1e-12
    assert float(g @ z + .5*z @ h @ z) < -0.15


def test_rbm_negative_curvature_can_accept_nonlinear_step():
    system = tilted_field_ising(2, field_x=.5, field_z=.5)
    model = ComplexRBM(2, 8, seed=1)
    from neural_quantum_solver import ExactDiagonalizer
    ed = ExactDiagonalizer(dtype=torch.complex128)
    ground = tilted_field_ising(2, coupling=0, field_x=.5, field_z=0)
    values, vectors = ed.diagonalize(system)
    target = vectors @ (torch.exp(-.1j*values)*(vectors.mH @ ed.ground_state(ground).state))
    initial = 1-abs(torch.vdot(target, exact_state(model, system).amplitudes))**2
    opt = OverlapCurvature(model, system, target, CurvatureSettings())
    records = [opt.step() for _ in range(10)]
    assert any(r.min_reduced_hessian_eigenvalue < 0 for r in records)
    assert records[-1].loss_after < float(initial.detach())*.01
    assert any(r.accepted for r in records)
    assert all(r.quantum_distance_squared <= r.radius**2 for r in records if r.accepted)


@pytest.mark.parametrize('curvature', ['full', 'gauss_newton'])
def test_fit_decreases_loss_and_freezes_target(curvature):
    system = tilted_field_ising(2)
    model = LogAmplitudeTable(2)
    target = torch.tensor([1, 0.7j, -0.4, 0.2+0.3j], dtype=torch.complex128,
                          requires_grad=True)
    original_target = target.detach().clone()
    records = []
    result = optimize_overlap_curvature(model, system, target, max_steps=60,
        tolerance=1e-9, settings=CurvatureSettings(curvature=curvature),
        on_step=lambda r, i: records.append(r))
    assert result[1] < 1e-7
    assert result[2] <= result[0]
    assert target.grad is None
    torch.testing.assert_close(target.detach(), original_target)
    assert all(r.loss_after <= r.loss_before for r in records)
    assert all(r.update_norm <= 0.5 + 1e-10 for r in records)


def test_rejected_step_keeps_model_and_state_resume_is_exact():
    system = tilted_field_ising(2)
    a = ComplexRBM(2, 2, seed=9)
    target = exact_state(ComplexRBM(2, 2, seed=10), system).amplitudes.detach()
    opt = OverlapCurvature(a, system, target, CurvatureSettings())
    opt.step()
    b = deepcopy(a)
    resumed = OverlapCurvature(b, system, target, CurvatureSettings())
    resumed.load_state_dict(opt.state_dict())
    ra, rb = opt.step(), resumed.step()
    assert ra.loss_after == pytest.approx(rb.loss_after, abs=1e-14)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)
    before = deepcopy(a.state_dict())
    exact_target = exact_state(a, system).amplitudes.detach()
    stationary = OverlapCurvature(a, system, exact_target, CurvatureSettings())
    record = stationary.step()
    assert not record.accepted
    for name, p in a.state_dict().items():
        torch.testing.assert_close(p, before[name], atol=0, rtol=0)
