from __future__ import annotations

from dataclasses import dataclass
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .models import NeuralQuantumState
from .drivers import GroundStateDriver
from .optimizers import Adam
from .systems import PhysicalSystem
from .variational import FullSumState


@dataclass(frozen=True)
class GroundStateStepResult:
    step: int
    energy: float
    variance: float
    grad_norm: float


@dataclass
class GroundStateRunResult:
    history: list[GroundStateStepResult]
    best_energy: float
    best_state_dict: dict[str, torch.Tensor]


class ExactGroundStateRunner:
    """Differentiable full-summation ground-state optimization."""

    def __init__(
        self, *, learning_rate: float = 1e-2, steps: int = 1000,
        num_gpus: int = 1,
    ) -> None:
        if learning_rate <= 0 or steps < 1:
            raise ValueError("learning_rate and steps must be positive")
        self.learning_rate, self.steps = learning_rate, steps
        self.num_gpus = num_gpus

    def run(self, model: NeuralQuantumState, system: PhysicalSystem) -> GroundStateRunResult:
        driver = GroundStateDriver(
            FullSumState(system, model, num_gpus=self.num_gpus),
            Adam(learning_rate=self.learning_rate),
        )
        result = driver.run(self.steps)
        history = [
            GroundStateStepResult(
                step.step, step.energy, step.variance, step.gradient_norm
            )
            for step in result.history
        ]
        return GroundStateRunResult(history, result.best_energy, result.best_state_dict)
