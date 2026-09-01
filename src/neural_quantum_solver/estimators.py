from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import torch
from .models import NeuralQuantumState
from .samplers import SampleBatch
from .systems import PhysicalSystem


@dataclass(frozen=True)
class ExactState:
    configurations: torch.Tensor
    log_amplitudes: torch.Tensor
    amplitudes: torch.Tensor
    probabilities: torch.Tensor
    log_norm: torch.Tensor


@dataclass(frozen=True)
class EnergyEstimate:
    energy: torch.Tensor
    variance: torch.Tensor
    local_energies: torch.Tensor
    imaginary_residual: torch.Tensor


@dataclass(frozen=True)
class SampledEnergy:
    configurations: torch.Tensor
    weights: torch.Tensor
    local_energies: torch.Tensor
    energy: torch.Tensor
    variance: torch.Tensor
    imaginary_residual: torch.Tensor
    acceptance_rate: float | None
    exact: bool


@dataclass(frozen=True)
class ShardedSampledEnergy:
    """Global statistics plus device-local shards for data-parallel updates."""

    models: tuple[NeuralQuantumState, ...]
    shards: tuple[SampledEnergy, ...]
    energy: torch.Tensor
    variance: torch.Tensor
    imaginary_residual: torch.Tensor
    acceptance_rate: float | None
    exact: bool

    @property
    def num_samples(self) -> int:
        return sum(shard.configurations.shape[0] for shard in self.shards)


def exact_state(model: NeuralQuantumState, system: PhysicalSystem) -> ExactState:
    device = next(model.parameters()).device
    configurations = system.hilbert.all_states(device=device)
    log_amplitudes = model.log_psi(configurations)
    log_norm = torch.logsumexp(2 * log_amplitudes.real, dim=0)
    amplitudes = torch.exp(log_amplitudes - 0.5 * log_norm)
    probabilities = torch.abs(amplitudes) ** 2
    return ExactState(configurations, log_amplitudes, amplitudes, probabilities, log_norm)


def exact_energy(
    model: NeuralQuantumState, system: PhysicalSystem, *, dtype: torch.dtype | None = None
) -> EnergyEstimate:
    state = exact_state(model, system)
    dtype = dtype or state.amplitudes.dtype
    connections = system.hamiltonian.connections(state.configurations, dtype=dtype)
    connected_log_psi = model.log_psi(connections.states)
    terms = connections.matrix_elements * torch.exp(
        connected_log_psi - state.log_amplitudes[connections.sample_indices]
    )
    local = torch.zeros(
        system.hilbert.size, dtype=dtype, device=state.amplitudes.device
    )
    local.index_add_(0, connections.sample_indices, terms)
    probabilities = state.probabilities.to(local.real.dtype)
    energy_complex = torch.sum(probabilities * local)
    energy = energy_complex.real
    variance_raw = torch.sum(probabilities * torch.abs(local) ** 2) - torch.abs(energy_complex) ** 2
    variance = torch.where(variance_raw >= 0, variance_raw, torch.zeros_like(variance_raw))
    return EnergyEstimate(energy, variance, local, energy_complex.imag.abs())


def expectation(state: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    return torch.vdot(state, operator @ state) / torch.vdot(state, state)


@torch.no_grad()
def local_energies(
    model: NeuralQuantumState,
    system: PhysicalSystem,
    configurations: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the generic packed-connection local-energy estimator."""
    parameter = next(model.parameters())
    connections = system.hamiltonian.connections(
        configurations, dtype=parameter.dtype
    )
    base = model.log_psi(configurations)
    connected = model.log_psi(connections.states)
    terms = connections.matrix_elements * torch.exp(
        connected - base[connections.sample_indices]
    )
    result = torch.zeros(
        configurations.shape[0], dtype=parameter.dtype, device=parameter.device
    )
    result.index_add_(0, connections.sample_indices, terms)
    return result


def sample_weights(sample: SampleBatch, model: NeuralQuantumState) -> torch.Tensor:
    if sample.exact:
        if sample.weights is None:
            raise ValueError("an exact sample batch must contain Born weights")
        return sample.weights
    parameter = next(model.parameters())
    return torch.full(
        (sample.configurations.shape[0],),
        1.0 / sample.configurations.shape[0],
        dtype=parameter.real.dtype,
        device=parameter.device,
    )


def sampled_energy(
    model: NeuralQuantumState,
    system: PhysicalSystem,
    sample: SampleBatch,
) -> SampledEnergy:
    weights = sample_weights(sample, model)
    local = local_energies(model, system, sample.configurations)
    energy = torch.sum(weights * local)
    variance = torch.sum(weights * torch.abs(local - energy) ** 2).real
    return SampledEnergy(
        sample.configurations,
        weights,
        local,
        energy,
        variance,
        energy.imag.abs(),
        sample.acceptance_rate,
        sample.exact,
    )


def sharded_sampled_energy(
    models: tuple[NeuralQuantumState, ...],
    system: PhysicalSystem,
    samples: tuple[SampleBatch, ...],
) -> ShardedSampledEnergy:
    """Compute global VMC statistics while retaining data on each device."""
    if len(models) != len(samples) or not models:
        raise ValueError("models and samples must contain equally many shards")
    if len({sample.exact for sample in samples}) != 1:
        raise ValueError("sample shards disagree on exact/Monte Carlo mode")

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        local_values = tuple(
            executor.map(
                lambda pair: local_energies(pair[0], system, pair[1].configurations),
                zip(models, samples),
            )
        )

    primary = next(models[0].parameters()).device
    if samples[0].exact:
        if any(sample.weights is None for sample in samples):
            raise ValueError("exact shards require globally normalized weights")
        local_weights = tuple(sample.weights for sample in samples)
    else:
        total = sum(sample.configurations.shape[0] for sample in samples)
        local_weights = tuple(
            torch.full(
                (sample.configurations.shape[0],),
                1.0 / total,
                dtype=next(model.parameters()).real.dtype,
                device=next(model.parameters()).device,
            )
            for model, sample in zip(models, samples)
        )

    energy = sum(
        (
            torch.sum(weights * local).to(primary)
            for weights, local in zip(local_weights, local_values)
        ),
        torch.zeros((), dtype=local_values[0].dtype, device=primary),
    )
    variance = sum(
        (
            torch.sum(
                weights * torch.abs(local - energy.to(local.device)) ** 2
            ).real.to(primary)
            for weights, local in zip(local_weights, local_values)
        ),
        torch.zeros((), dtype=energy.real.dtype, device=primary),
    )

    shards = tuple(
        SampledEnergy(
            sample.configurations,
            weights,
            local,
            energy.to(local.device),
            variance.to(local.device),
            energy.imag.abs().to(local.device),
            sample.acceptance_rate,
            sample.exact,
        )
        for sample, weights, local in zip(samples, local_weights, local_values)
    )

    if samples[0].exact:
        acceptance_rate = None
    elif all(sample.accepted is not None and sample.proposed for sample in samples):
        accepted = sum(int(sample.accepted) for sample in samples)
        proposed = sum(int(sample.proposed) for sample in samples)
        acceptance_rate = accepted / proposed
    else:
        counts = [sample.configurations.shape[0] for sample in samples]
        acceptance_rate = sum(
            float(sample.acceptance_rate) * count
            for sample, count in zip(samples, counts)
        ) / sum(counts)

    return ShardedSampledEnergy(
        models,
        shards,
        energy,
        variance,
        energy.imag.abs(),
        acceptance_rate,
        samples[0].exact,
    )
