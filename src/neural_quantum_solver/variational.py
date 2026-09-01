from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import torch

from .devices import DeviceMesh, make_generator
from .estimators import (
    SampledEnergy,
    ShardedSampledEnergy,
    sampled_energy,
    sharded_sampled_energy,
)
from .models import AmplitudePhaseNQS, NeuralQuantumState
from .real_estimators import (
    RealPairSampledEnergy,
    RealPairShardedSampledEnergy,
    real_pair_sampled_energy,
    real_pair_sharded_sampled_energy,
)
from .samplers import ExactSampler, SampleBatch, Sampler
from .systems import PhysicalSystem


@dataclass
class VariationalState:
    """将模型、物理系统和采样策略组合成一个变分态。"""

    system: PhysicalSystem
    model: NeuralQuantumState
    sampler: Sampler = field(default_factory=ExactSampler)
    seed: int = 0
    num_gpus: int = 1
    _last_sample: SampleBatch | tuple[SampleBatch, ...] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        device = next(self.model.parameters()).device
        self.device_mesh = DeviceMesh.create(device, self.num_gpus)
        if device != self.device_mesh.primary:
            raise ValueError("model must be located on the first requested device")
        if isinstance(self.sampler, ExactSampler):
            if self.num_gpus > self.system.hilbert.size:
                raise ValueError("num_gpus cannot exceed the exact Hilbert-space size")
        elif hasattr(self.sampler, "num_chains"):
            if self.sampler.num_chains < self.num_gpus:
                raise ValueError("num_chains must be at least num_gpus")
        self.replicas = (self.model,) + tuple(
            deepcopy(self.model).to(replica_device)
            for replica_device in self.device_mesh.devices[1:]
        )
        self.generators = tuple(
            make_generator(replica_device, self.seed + index)
            for index, replica_device in enumerate(self.device_mesh.devices)
        )

    def _sync_replicas(self) -> None:
        state_dict = self.model.state_dict()
        for replica in self.replicas[1:]:
            replica.load_state_dict(state_dict)

    def _exact_sample_shards(self) -> tuple[SampleBatch, ...]:
        configurations = self.system.hilbert.all_states(device="cpu")
        configuration_shards = tuple(
            chunk.to(device)
            for chunk, device in zip(
                torch.tensor_split(configurations, self.num_gpus),
                self.device_mesh.devices,
            )
        )
        def evaluate(pair):
            model, shard = pair
            with torch.no_grad():
                if isinstance(model, AmplitudePhaseNQS):
                    return model.log_amplitude(shard)
                return model.log_psi(shard).real

        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            log_amplitudes = tuple(
                executor.map(
                    evaluate,
                    zip(self.replicas, configuration_shards),
                )
            )
            local_log_norms = torch.stack(
                [
                    torch.logsumexp(2 * values, dim=0).to(
                        self.device_mesh.primary
                    )
                    for values in log_amplitudes
                ]
            )
            log_norm = torch.logsumexp(local_log_norms, dim=0)
            return tuple(
                SampleBatch(
                    shard,
                    torch.exp(2 * values - log_norm.to(values.device)),
                    None,
                    True,
                )
                for shard, values in zip(configuration_shards, log_amplitudes)
            )

    def _metropolis_sample_shards(self) -> tuple[SampleBatch, ...]:
        if not hasattr(self.sampler, "shard"):
            raise TypeError(
                f"{type(self.sampler).__name__} does not support multi-device sharding"
            )
        samplers = tuple(
            self.sampler.shard(self.num_gpus, index)
            for index in range(self.num_gpus)
        )
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            return tuple(
                executor.map(
                    lambda values: values[0].sample(
                        values[1], self.system, generator=values[2]
                    ),
                    zip(samplers, self.replicas, self.generators),
                )
            )

    @property
    def samples(self) -> SampleBatch | tuple[SampleBatch, ...]:
        if self._last_sample is None:
            return self.sample()
        return self._last_sample

    def sample(self) -> SampleBatch | tuple[SampleBatch, ...]:
        if self.num_gpus == 1:
            self._last_sample = self.sampler.sample(
                self.model, self.system, generator=self.generators[0]
            )
            return self._last_sample
        self._sync_replicas()
        if isinstance(self.sampler, ExactSampler):
            self._last_sample = self._exact_sample_shards()
        else:
            self._last_sample = self._metropolis_sample_shards()
        return self._last_sample

    def expect_energy(
        self, *, resample: bool = False
    ) -> (
        SampledEnergy
        | ShardedSampledEnergy
        | RealPairSampledEnergy
        | RealPairShardedSampledEnergy
    ):
        sample = self.sample() if resample or self._last_sample is None else self._last_sample
        if isinstance(self.model, AmplitudePhaseNQS):
            if isinstance(sample, tuple):
                return real_pair_sharded_sampled_energy(
                    self.replicas, self.system, sample
                )
            return real_pair_sampled_energy(self.model, self.system, sample)
        if isinstance(sample, tuple):
            return sharded_sampled_energy(
                self.replicas, self.system, sample
            )
        return sampled_energy(self.model, self.system, sample)

    def reset(self) -> None:
        self._last_sample = None


class FullSumState(VariationalState):
    def __init__(
        self,
        system: PhysicalSystem,
        model: NeuralQuantumState,
        *,
        seed: int = 0,
        num_gpus: int = 1,
    ) -> None:
        super().__init__(system, model, ExactSampler(), seed, num_gpus)
