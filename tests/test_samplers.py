import pytest
import torch

from neural_quantum_solver import ComplexRBM, MetropolisSampler, tilted_field_ising


class CountingRBM(ComplexRBM):
    def __init__(self, num_visible: int, num_hidden: int, **kwargs):
        super().__init__(num_visible, num_hidden, **kwargs)
        self.log_psi_calls = 0

    def log_psi(self, configurations: torch.Tensor) -> torch.Tensor:
        self.log_psi_calls += 1
        return super().log_psi(configurations)


def test_metropolis_returns_chains_times_sweeps_samples():
    system = tilted_field_ising(3)
    model = ComplexRBM(3, 4, dtype=torch.complex128, seed=47)
    sampler = MetropolisSampler(
        num_chains=5,
        thermal_sweeps=1,
        sweeps=4,
        sweep_size=2,
    )

    batch = sampler.sample(model, system, generator=torch.Generator().manual_seed(3))

    assert batch.configurations.shape == (20, 3)
    assert torch.all((batch.configurations == 1) | (batch.configurations == -1))
    assert batch.weights is None
    assert batch.exact is False
    assert 0.0 <= batch.acceptance_rate <= 1.0


def test_metropolis_uses_configured_updates_between_samples():
    system = tilted_field_ising(3)
    model = CountingRBM(3, 4, dtype=torch.complex128, seed=53)
    sampler = MetropolisSampler(
        num_chains=4,
        thermal_sweeps=2,
        sweeps=3,
        sweep_size=2,
    )

    sampler.sample(model, system, generator=torch.Generator().manual_seed(5))

    # One initial evaluation + 2*3 thermal updates + 3*2 sampling updates.
    assert model.log_psi_calls == 13


def test_default_sweep_size_is_number_of_sites():
    system = tilted_field_ising(3)
    model = CountingRBM(3, 4, dtype=torch.complex128, seed=59)
    sampler = MetropolisSampler(
        num_chains=2,
        thermal_sweeps=0,
        sweeps=4,
    )

    batch = sampler.sample(model, system, generator=torch.Generator().manual_seed(7))

    assert batch.configurations.shape == (8, 3)
    # One initial evaluation + 4 retained samples * 3 local updates.
    assert model.log_psi_calls == 13


@pytest.mark.parametrize("sweep_size", [0, -1, True, 1.5])
def test_metropolis_rejects_invalid_sweep_size(sweep_size):
    with pytest.raises(ValueError, match="sweep_size"):
        MetropolisSampler(4, sweep_size=sweep_size)

