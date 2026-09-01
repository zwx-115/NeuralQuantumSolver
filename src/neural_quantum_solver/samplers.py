from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import torch
from .models import NeuralQuantumState
from .systems import PhysicalSystem


@dataclass(frozen=True)
class SampleBatch:
    configurations: torch.Tensor
    weights: torch.Tensor | None
    acceptance_rate: float | None
    exact: bool
    accepted: int | None = None
    proposed: int | None = None


class Sampler(Protocol):
    def sample(
        self,
        model: NeuralQuantumState,
        system: PhysicalSystem,
        *,
        generator: torch.Generator | None = None,
    ) -> SampleBatch: ...


class ExactSampler:
    def sample(
        self,
        model: NeuralQuantumState,
        system: PhysicalSystem,
        *,
        generator: torch.Generator | None = None,
    ) -> SampleBatch:
        device = next(model.parameters()).device
        configurations = system.hilbert.all_states(device=device)
        with torch.no_grad():
            probabilities = torch.softmax(2 * model.log_psi(configurations).real, dim=0)
        return SampleBatch(configurations, probabilities, None, True)


class MetropolisSampler:
    """Parallel-chain Metropolis sampler with trajectory collection.

    ``sweeps`` is the number of retained samples per chain.  ``sweep_size`` is
    the number of local single-spin proposals between two retained samples;
    when omitted it is the number of sites, i.e. one full lattice sweep.
    Consequently, each call returns ``num_chains * sweeps`` configurations.
    """

    def __init__(
        self,
        num_chains: int,
        *,
        thermal_sweeps: int = 100,
        sweeps: int = 1,
        sweep_size: int | None = None,
    ):
        integer_settings = (num_chains, thermal_sweeps, sweeps)
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in integer_settings)
            or num_chains < 1
            or thermal_sweeps < 0
            or sweeps < 1
        ):
            raise ValueError("invalid Metropolis sampler settings")
        if (
            sweep_size is not None
            and (
                isinstance(sweep_size, bool)
                or not isinstance(sweep_size, int)
                or sweep_size < 1
            )
        ):
            raise ValueError("sweep_size must be a positive integer or None")
        self.num_chains = num_chains
        self.thermal_sweeps = thermal_sweeps
        self.sweeps = sweeps
        self.sweep_size = sweep_size

    def shard(self, num_shards: int, shard_index: int) -> "MetropolisSampler":
        """Return one balanced chain shard while preserving total chain count."""
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("invalid sampler shard")
        quotient, remainder = divmod(self.num_chains, num_shards)
        local_chains = quotient + int(shard_index < remainder)
        if local_chains == 0:
            raise ValueError("num_chains must be at least num_gpus")
        return MetropolisSampler(
            local_chains,
            thermal_sweeps=self.thermal_sweeps,
            sweeps=self.sweeps,
            sweep_size=self.sweep_size,
        )

    def sample(
        self, model: NeuralQuantumState, system: PhysicalSystem, *,
        generator: torch.Generator | None = None,
    ) -> SampleBatch:
        device = next(model.parameters()).device
        num_sites = system.hilbert.num_sites
        sweep_size = self.sweep_size or num_sites
        states = 2 * torch.randint(
            0, 2, (self.num_chains, num_sites),
            device=device, generator=generator,
        ) - 1
        accepted = proposed = 0
        site_cursor = 0

        with torch.no_grad():
            log_psi = model.log_psi(states)

            def advance(num_updates: int, *, measure_acceptance: bool) -> None:
                nonlocal accepted, proposed, site_cursor, log_psi
                for _ in range(num_updates):
                    site = site_cursor % num_sites
                    site_cursor += 1
                    trial = states.clone()
                    trial[:, site] *= -1
                    trial_log = model.log_psi(trial)
                    probability = torch.exp(2 * (trial_log.real - log_psi.real))
                    random = torch.rand(
                        self.num_chains, device=device, generator=generator
                    )
                    take = random < torch.minimum(
                        probability, torch.ones_like(probability)
                    )
                    if measure_acceptance:
                        accepted += int(take.sum())
                        proposed += self.num_chains
                    states[take] = trial[take]
                    log_psi[take] = trial_log[take]

            # Thermal sweeps retain their conventional size of num_sites.
            advance(self.thermal_sweeps * num_sites, measure_acceptance=False)

            samples = []
            for _ in range(self.sweeps):
                advance(sweep_size, measure_acceptance=True)
                samples.append(states.clone())

        configurations = torch.stack(samples, dim=0).reshape(
            self.sweeps * self.num_chains, num_sites
        )
        return SampleBatch(
            configurations, None, accepted / proposed, False, accepted, proposed
        )
