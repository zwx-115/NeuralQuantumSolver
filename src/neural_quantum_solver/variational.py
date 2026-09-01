from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import torch

from .devices import DeviceMesh, make_generator, run_on_device, transfer_tensor
from .distributed import ParallelContext
from .estimators import (
    SampledEnergy,
    ShardedSampledEnergy,
    sampled_energy,
    sharded_sampled_energy,
)
from .models import AmplitudePhaseNQS, NeuralQuantumState
from .real_estimators import (
    DistributedRealPairSampledEnergy,
    RealPairSampledEnergy,
    RealPairShardedSampledEnergy,
    distributed_real_pair_sampled_energy,
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
    parallel_context: ParallelContext | None = None
    _last_sample: SampleBatch | tuple[SampleBatch, ...] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        device = next(self.model.parameters()).device
        if self.parallel_context is not None and self.parallel_context.distributed:
            context = self.parallel_context
            if device != context.device:
                raise ValueError("model must be located on the local-rank device")
            if not isinstance(self.model, AmplitudePhaseNQS):
                raise TypeError(
                    "NCCL benchmark mode currently requires AmplitudePhaseNQS"
                )
            if isinstance(self.sampler, ExactSampler):
                raise NotImplementedError(
                    "NCCL benchmark mode currently requires MetropolisSampler"
                )
            if not hasattr(self.sampler, "shard"):
                raise TypeError("distributed sampler must provide shard()")
            if self.sampler.num_chains < context.world_size:
                raise ValueError("num_chains must be at least world_size")
            self.device_mesh = DeviceMesh((device,))
            self.replicas = (self.model,)
            self.generators = (
                make_generator(device, self.seed + context.rank),
            )
            self._distributed_sampler = self.sampler.shard(
                context.world_size, context.rank
            )
            context.broadcast_model(self.model)
            return
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
            deepcopy(self.model).cpu().to(replica_device)
            for replica_device in self.device_mesh.devices[1:]
        )
        self.generators = tuple(
            make_generator(replica_device, self.seed + index)
            for index, replica_device in enumerate(self.device_mesh.devices)
        )

    def _sync_replicas(self) -> None:
        state_dict = {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
        }
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
            device = next(model.parameters()).device

            def evaluate_on_device():
                with torch.no_grad():
                    if isinstance(model, AmplitudePhaseNQS):
                        return model.log_amplitude(shard)
                    return model.log_psi(shard).real

            return run_on_device(device, evaluate_on_device)

        log_amplitudes = tuple(
            evaluate(pair)
            for pair in zip(self.replicas, configuration_shards)
        )
        local_log_norms = torch.stack(
            [
                transfer_tensor(
                    torch.logsumexp(2 * values, dim=0),
                    self.device_mesh.primary,
                )
                for values in log_amplitudes
            ]
        )
        log_norm = torch.logsumexp(local_log_norms, dim=0)
        return tuple(
            SampleBatch(
                shard,
                torch.exp(
                    2 * values - transfer_tensor(log_norm, values.device)
                ),
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

        def sample_shard(values):
            sampler, model, generator = values
            device = next(model.parameters()).device
            return run_on_device(
                device,
                sampler.sample,
                model,
                self.system,
                generator=generator,
                defer_scalar_results=True,
            )

        batches = tuple(
            sample_shard(values)
            for values in zip(samplers, self.replicas, self.generators)
        )
        normalized = []
        for batch in batches:
            if batch.accepted is None or batch.proposed is None:
                raise RuntimeError("Metropolis shard did not report acceptance counts")
            accepted = int(batch.accepted)
            normalized.append(
                replace(
                    batch,
                    acceptance_rate=accepted / batch.proposed,
                    accepted=accepted,
                )
            )
        return tuple(normalized)

    @property
    def samples(self) -> SampleBatch | tuple[SampleBatch, ...]:
        if self._last_sample is None:
            return self.sample()
        return self._last_sample

    def sample(self) -> SampleBatch | tuple[SampleBatch, ...]:
        if self.parallel_context is not None and self.parallel_context.distributed:
            self.parallel_context.broadcast_model(self.model)
            self._last_sample = self._distributed_sampler.sample(
                self.model,
                self.system,
                generator=self.generators[0],
            )
            return self._last_sample
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
        | DistributedRealPairSampledEnergy
        | RealPairSampledEnergy
        | RealPairShardedSampledEnergy
    ):
        sample = self.sample() if resample or self._last_sample is None else self._last_sample
        if isinstance(self.model, AmplitudePhaseNQS):
            if self.parallel_context is not None and self.parallel_context.distributed:
                if not isinstance(sample, SampleBatch):
                    raise RuntimeError("distributed sampling returned multiple shards")
                return distributed_real_pair_sampled_energy(
                    self.model,
                    self.system,
                    sample,
                    self.parallel_context,
                )
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
