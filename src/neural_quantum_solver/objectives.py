from dataclasses import dataclass
import torch
from .estimators import exact_energy
from .models import NeuralQuantumState
from .systems import PhysicalSystem


@dataclass(frozen=True)
class ObjectiveResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | float | bool]


class ExactGroundStateObjective:
    def __call__(self, model: NeuralQuantumState, system: PhysicalSystem) -> ObjectiveResult:
        estimate = exact_energy(model, system)
        return ObjectiveResult(
            estimate.energy,
            {"energy": estimate.energy.detach(), "variance": estimate.variance.detach(),
             "imaginary_residual": estimate.imaginary_residual.detach()},
        )
