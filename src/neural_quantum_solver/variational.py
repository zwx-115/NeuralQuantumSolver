from __future__ import annotations

from dataclasses import dataclass, field
import torch

from .estimators import SampledEnergy, sampled_energy
from .models import NeuralQuantumState
from .samplers import ExactSampler, SampleBatch, Sampler
from .systems import PhysicalSystem


@dataclass
class VariationalState:
    """A model, physical system, and sampling strategy treated as one state."""

    system: PhysicalSystem
    model: NeuralQuantumState
    sampler: Sampler = field(default_factory=ExactSampler)
    seed: int = 0
    _last_sample: SampleBatch | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        device = next(self.model.parameters()).device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)

    @property
    def samples(self) -> SampleBatch:
        if self._last_sample is None:
            return self.sample()
        return self._last_sample

    def sample(self) -> SampleBatch:
        self._last_sample = self.sampler.sample(
            self.model, self.system, generator=self.generator
        )
        return self._last_sample

    def expect_energy(self, *, resample: bool = False) -> SampledEnergy:
        sample = self.sample() if resample or self._last_sample is None else self._last_sample
        return sampled_energy(self.model, self.system, sample)

    def reset(self) -> None:
        self._last_sample = None


class FullSumState(VariationalState):
    def __init__(
        self, system: PhysicalSystem, model: NeuralQuantumState, *, seed: int = 0
    ) -> None:
        super().__init__(system, model, ExactSampler(), seed)
