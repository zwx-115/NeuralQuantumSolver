import math
import copy
import torch

from neural_quantum_solver import (
    Adam,
    ComplexRBM,
    ExactSampler,
    FullSumState,
    GroundStateDriver,
    MetropolisSampler,
    SR,
    VariationalState,
    tilted_field_ising,
)
from neural_quantum_solver.estimators import exact_energy, sampled_energy


def _model(seed=3):
    return ComplexRBM(3, 6, dtype=torch.complex128, seed=seed)


def test_exact_sampler_can_be_combined_with_adam_or_sr():
    system = tilted_field_ising(3)
    for optimizer in (Adam(learning_rate=0.02), SR(learning_rate=0.03)):
        driver = GroundStateDriver(
            VariationalState(system, _model(), ExactSampler(), seed=4), optimizer
        )
        result = driver.advance()
        assert math.isfinite(result.energy)
        assert result.acceptance_rate is None
        assert result.update_norm > 0


def test_metropolis_sampler_can_be_combined_with_adam_or_sr():
    system = tilted_field_ising(3)
    for optimizer in (Adam(learning_rate=0.01), SR(learning_rate=0.02)):
        sampler = MetropolisSampler(16, thermal_sweeps=2, sweeps=1)
        driver = GroundStateDriver(
            VariationalState(system, _model(), sampler, seed=5), optimizer
        )
        result = driver.advance()
        assert math.isfinite(result.energy)
        assert 0.0 <= result.acceptance_rate <= 1.0


def test_full_sum_sr_lowers_energy_and_exposes_qgt_diagnostics():
    system = tilted_field_ising(3)
    driver = GroundStateDriver(
        FullSumState(system, _model(seed=7)),
        SR(learning_rate=0.05, regularization=1e-3),
    )
    result = driver.run(steps=15)
    assert result.history[-1].energy < result.history[0].energy
    assert result.history[-1].optimizer_metrics["effective_rank"] > 0


def test_driver_callback_can_stop_a_run():
    system = tilted_field_ising(3)
    driver = GroundStateDriver(FullSumState(system, _model()), Adam(0.01))
    result = driver.run(steps=20, callback=lambda step, _: step.step < 2)
    assert len(result.history) == 3


def test_exact_vmc_score_gradient_matches_differentiable_energy():
    system = tilted_field_ising(3)
    direct_model = _model(seed=11)
    score_model = copy.deepcopy(direct_model)

    exact_energy(direct_model, system).energy.backward()
    direct_gradient = torch.cat(
        [parameter.grad.reshape(-1) for parameter in direct_model.parameters()]
    )

    sample = ExactSampler().sample(score_model, system)
    statistics = sampled_energy(score_model, system, sample)
    log_values = score_model.log_psi(statistics.configurations)
    surrogate = 2 * torch.real(
        torch.sum(
            statistics.weights
            * (statistics.local_energies - statistics.energy).detach()
            * log_values.conj()
        )
    )
    surrogate.backward()
    score_gradient = torch.cat(
        [parameter.grad.reshape(-1) for parameter in score_model.parameters()]
    )
    assert torch.allclose(direct_gradient, score_gradient, atol=1e-12, rtol=1e-12)
