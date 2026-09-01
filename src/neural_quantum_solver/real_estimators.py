from __future__ import annotations

from dataclasses import dataclass

import torch

from .devices import run_on_device, transfer_tensor
from .distributed import ParallelContext
from .models import AmplitudePhaseNQS, LogPsiParts
from .samplers import SampleBatch
from .systems import PhysicalSystem


@dataclass(frozen=True)
class RealPairExactState:
    configurations: torch.Tensor
    log_psi: LogPsiParts
    amplitude_real: torch.Tensor
    amplitude_imag: torch.Tensor
    probabilities: torch.Tensor
    log_norm: torch.Tensor


@dataclass(frozen=True)
class RealPairEnergyEstimate:
    energy: torch.Tensor
    variance: torch.Tensor
    local_energy_real: torch.Tensor
    local_energy_imag: torch.Tensor
    imaginary_residual: torch.Tensor


@dataclass(frozen=True)
class RealPairSampledEnergy:
    configurations: torch.Tensor
    weights: torch.Tensor
    local_energy_real: torch.Tensor
    local_energy_imag: torch.Tensor
    energy: torch.Tensor
    energy_imag: torch.Tensor
    variance: torch.Tensor
    imaginary_residual: torch.Tensor
    acceptance_rate: float | None
    exact: bool


@dataclass(frozen=True)
class RealPairShardedSampledEnergy:
    models: tuple[AmplitudePhaseNQS, ...]
    shards: tuple[RealPairSampledEnergy, ...]
    energy: torch.Tensor
    energy_imag: torch.Tensor
    variance: torch.Tensor
    imaginary_residual: torch.Tensor
    acceptance_rate: float | None
    exact: bool

    @property
    def num_samples(self) -> int:
        return sum(shard.configurations.shape[0] for shard in self.shards)


@dataclass(frozen=True)
class DistributedRealPairSampledEnergy:
    """每个进程仅保留本地样本、全局统计量由集合通信得到。"""

    model: AmplitudePhaseNQS
    shard: RealPairSampledEnergy
    context: ParallelContext
    energy: torch.Tensor
    energy_imag: torch.Tensor
    variance: torch.Tensor
    imaginary_residual: torch.Tensor
    acceptance_rate: float | None
    exact: bool
    num_samples: int


def real_pair_exact_state(
    model: AmplitudePhaseNQS, system: PhysicalSystem
) -> RealPairExactState:
    parameter = next(model.parameters())
    configurations = system.hilbert.all_states(device=parameter.device)
    parts = model.log_psi_parts(configurations)
    log_norm = torch.logsumexp(2 * parts.log_amplitude, dim=0)
    magnitude = torch.exp(parts.log_amplitude - 0.5 * log_norm)
    amplitude_real = magnitude * torch.cos(parts.phase)
    amplitude_imag = magnitude * torch.sin(parts.phase)
    probabilities = torch.exp(2 * parts.log_amplitude - log_norm)
    return RealPairExactState(
        configurations,
        parts,
        amplitude_real,
        amplitude_imag,
        probabilities,
        log_norm,
    )


@torch.no_grad()
def real_pair_local_energies(
    model: AmplitudePhaseNQS,
    system: PhysicalSystem,
    configurations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameter = next(model.parameters())
    connections = system.hamiltonian.real_connections(
        configurations, dtype=parameter.dtype
    )
    base = model.log_psi_parts(configurations)
    connected = model.log_psi_parts(connections.states)
    sample_indices = connections.sample_indices
    delta_log_amplitude = (
        connected.log_amplitude - base.log_amplitude[sample_indices]
    )
    delta_phase = connected.phase - base.phase[sample_indices]
    magnitude = torch.exp(delta_log_amplitude)
    ratio_real = magnitude * torch.cos(delta_phase)
    ratio_imag = magnitude * torch.sin(delta_phase)
    terms_real = (
        connections.matrix_elements_real * ratio_real
        - connections.matrix_elements_imag * ratio_imag
    )
    terms_imag = (
        connections.matrix_elements_real * ratio_imag
        + connections.matrix_elements_imag * ratio_real
    )
    local_real = torch.zeros(
        configurations.shape[0], dtype=parameter.dtype, device=parameter.device
    )
    local_imag = torch.zeros_like(local_real)
    local_real.index_add_(0, sample_indices, terms_real)
    local_imag.index_add_(0, sample_indices, terms_imag)
    return local_real, local_imag


def real_pair_exact_energy(
    model: AmplitudePhaseNQS, system: PhysicalSystem
) -> RealPairEnergyEstimate:
    state = real_pair_exact_state(model, system)
    local_real, local_imag = real_pair_local_energies(
        model, system, state.configurations
    )
    energy = torch.sum(state.probabilities * local_real)
    energy_imag = torch.sum(state.probabilities * local_imag)
    variance_raw = torch.sum(
        state.probabilities
        * ((local_real - energy).square() + (local_imag - energy_imag).square())
    )
    variance = torch.clamp_min(variance_raw, 0)
    return RealPairEnergyEstimate(
        energy, variance, local_real, local_imag, torch.abs(energy_imag)
    )


def _sample_weights(
    sample: SampleBatch, model: AmplitudePhaseNQS
) -> torch.Tensor:
    if sample.exact:
        if sample.weights is None:
            raise ValueError("an exact sample batch must contain Born weights")
        return sample.weights
    parameter = next(model.parameters())
    return torch.full(
        (sample.configurations.shape[0],),
        1.0 / sample.configurations.shape[0],
        dtype=parameter.dtype,
        device=parameter.device,
    )


def real_pair_sampled_energy(
    model: AmplitudePhaseNQS,
    system: PhysicalSystem,
    sample: SampleBatch,
) -> RealPairSampledEnergy:
    weights = _sample_weights(sample, model)
    local_real, local_imag = real_pair_local_energies(
        model, system, sample.configurations
    )
    energy = torch.sum(weights * local_real)
    energy_imag = torch.sum(weights * local_imag)
    variance = torch.sum(
        weights
        * ((local_real - energy).square() + (local_imag - energy_imag).square())
    )
    return RealPairSampledEnergy(
        sample.configurations,
        weights,
        local_real,
        local_imag,
        energy,
        energy_imag,
        variance,
        torch.abs(energy_imag),
        sample.acceptance_rate,
        sample.exact,
    )


def distributed_real_pair_sampled_energy(
    model: AmplitudePhaseNQS,
    system: PhysicalSystem,
    sample: SampleBatch,
    context: ParallelContext,
) -> DistributedRealPairSampledEnergy:
    if not context.distributed:
        raise ValueError("distributed sampled energy requires world_size > 1")
    if sample.exact:
        raise NotImplementedError(
            "distributed ExactSampler is not supported by the NVIDIA benchmark"
        )
    local_real, local_imag = real_pair_local_energies(
        model, system, sample.configurations
    )
    device = next(model.parameters()).device
    local_count = torch.tensor(
        sample.configurations.shape[0], dtype=torch.int64, device=device
    )
    total_count = context.all_reduce(local_count.clone())
    num_samples = int(total_count.item())
    weights = torch.full(
        (sample.configurations.shape[0],),
        1.0 / num_samples,
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    energy = context.all_reduce(torch.sum(weights * local_real))
    energy_imag = context.all_reduce(torch.sum(weights * local_imag))
    variance = context.all_reduce(
        torch.sum(
            weights
            * (
                (local_real - energy).square()
                + (local_imag - energy_imag).square()
            )
        )
    )
    if sample.accepted is not None and sample.proposed is not None:
        counts = torch.tensor(
            [int(sample.accepted), sample.proposed],
            dtype=torch.int64,
            device=device,
        )
        context.all_reduce(counts)
        acceptance_rate = float(counts[0].item() / counts[1].item())
    else:
        acceptance_rate = sample.acceptance_rate
    shard = RealPairSampledEnergy(
        sample.configurations,
        weights,
        local_real,
        local_imag,
        energy,
        energy_imag,
        variance,
        torch.abs(energy_imag),
        acceptance_rate,
        False,
    )
    return DistributedRealPairSampledEnergy(
        model,
        shard,
        context,
        energy,
        energy_imag,
        variance,
        torch.abs(energy_imag),
        acceptance_rate,
        False,
        num_samples,
    )


def real_pair_sharded_sampled_energy(
    models: tuple[AmplitudePhaseNQS, ...],
    system: PhysicalSystem,
    samples: tuple[SampleBatch, ...],
) -> RealPairShardedSampledEnergy:
    if len(models) != len(samples) or not models:
        raise ValueError("models and samples must contain equally many shards")
    if len({sample.exact for sample in samples}) != 1:
        raise ValueError("sample shards disagree on exact/Monte Carlo mode")

    def evaluate_local_energy(pair):
        model, sample = pair
        device = next(model.parameters()).device
        return run_on_device(
            device,
            real_pair_local_energies,
            model,
            system,
            sample.configurations,
        )

    local_values = tuple(
        evaluate_local_energy(pair)
        for pair in zip(models, samples)
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
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
            for model, sample in zip(models, samples)
        )
    energy = sum(
        (
            transfer_tensor(torch.sum(weights * local[0]), primary)
            for weights, local in zip(local_weights, local_values)
        ),
        torch.zeros((), dtype=local_values[0][0].dtype, device=primary),
    )
    energy_imag = sum(
        (
            transfer_tensor(torch.sum(weights * local[1]), primary)
            for weights, local in zip(local_weights, local_values)
        ),
        torch.zeros((), dtype=local_values[0][1].dtype, device=primary),
    )
    variance = sum(
        (
            transfer_tensor(
                torch.sum(
                    weights
                    * (
                        (
                            local[0]
                            - transfer_tensor(energy, local[0].device)
                        ).square()
                        + (
                            local[1]
                            - transfer_tensor(energy_imag, local[1].device)
                        ).square()
                    )
                ),
                primary,
            )
            for weights, local in zip(local_weights, local_values)
        ),
        torch.zeros((), dtype=energy.dtype, device=primary),
    )
    shards = tuple(
        RealPairSampledEnergy(
            sample.configurations,
            weights,
            local[0],
            local[1],
            transfer_tensor(energy, local[0].device),
            transfer_tensor(energy_imag, local[0].device),
            transfer_tensor(variance, local[0].device),
            transfer_tensor(torch.abs(energy_imag), local[0].device),
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
    return RealPairShardedSampledEnergy(
        models,
        shards,
        energy,
        energy_imag,
        variance,
        torch.abs(energy_imag),
        acceptance_rate,
        samples[0].exact,
    )
