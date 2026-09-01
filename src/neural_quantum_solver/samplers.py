from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import torch
from .models import AmplitudePhaseNQS, NeuralQuantumState
from .systems import PhysicalSystem


@dataclass(frozen=True)
class SampleBatch:
    configurations: torch.Tensor
    weights: torch.Tensor | None
    acceptance_rate: float | None
    exact: bool
    accepted: int | torch.Tensor | None = None
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
            if isinstance(model, AmplitudePhaseNQS):
                log_amplitude = model.log_amplitude(configurations)
            else:
                log_amplitude = model.log_psi(configurations).real
            probabilities = torch.softmax(2 * log_amplitude, dim=0)
        return SampleBatch(configurations, probabilities, None, True)


class MetropolisSampler:
    """能够保留演化轨迹的并行链 Metropolis 采样器。

    ``sweeps`` 表示每条链保留的样本数。``sweep_size`` 表示两个保留样本
    之间执行的局域单自旋提议次数；省略时取格点数，即完整扫描一次晶格。
    因此每次调用返回 ``num_chains * sweeps`` 个构型。
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
        """在保持总链数不变的前提下返回一个负载均衡的链分片。"""
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
        defer_scalar_results: bool = False,
    ) -> SampleBatch:
        device = next(model.parameters()).device
        num_sites = system.hilbert.num_sites
        sweep_size = self.sweep_size or num_sites
        states = 2 * torch.randint(
            0, 2, (self.num_chains, num_sites),
            device=device, generator=generator,
        ) - 1
        accepted = torch.zeros((), dtype=torch.int64, device=device)
        proposed = 0
        site_cursor = 0

        with torch.no_grad():
            if isinstance(model, AmplitudePhaseNQS):
                log_amplitude = model.log_amplitude(states)
            else:
                log_amplitude = model.log_psi(states).real

            def advance(num_updates: int, *, measure_acceptance: bool) -> None:
                nonlocal accepted, proposed, site_cursor, states, log_amplitude
                for _ in range(num_updates):
                    site = site_cursor % num_sites
                    site_cursor += 1
                    trial = states.clone()
                    trial[:, site] *= -1
                    if isinstance(model, AmplitudePhaseNQS):
                        trial_log_amplitude = model.log_amplitude(trial)
                    else:
                        trial_log_amplitude = model.log_psi(trial).real
                    probability = torch.exp(
                        2 * (trial_log_amplitude - log_amplitude)
                    )
                    random = torch.rand(
                        self.num_chains, device=device, generator=generator
                    )
                    take = random < torch.minimum(
                        probability, torch.ones_like(probability)
                    )
                    if measure_acceptance:
                        accepted += take.sum()
                        proposed += self.num_chains
                    states = torch.where(take[:, None], trial, states)
                    log_amplitude = torch.where(
                        take, trial_log_amplitude, log_amplitude
                    )

            # 热化阶段仍按传统定义，每次 sweep 执行 num_sites 次更新。
            advance(self.thermal_sweeps * num_sites, measure_acceptance=False)

            samples = []
            for _ in range(self.sweeps):
                advance(sweep_size, measure_acceptance=True)
                samples.append(states.clone())

        configurations = torch.stack(samples, dim=0).reshape(
            self.sweeps * self.num_chains, num_sites
        )
        if defer_scalar_results:
            return SampleBatch(
                configurations, None, None, False, accepted, proposed
            )
        accepted_value = int(accepted)
        return SampleBatch(
            configurations,
            None,
            accepted_value / proposed,
            False,
            accepted_value,
            proposed,
        )
