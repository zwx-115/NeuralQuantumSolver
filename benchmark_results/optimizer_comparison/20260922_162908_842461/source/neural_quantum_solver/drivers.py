from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import torch

from .optimizers import GroundStateOptimizer, OptimizerStep
from .checkpoint import save_checkpoint
from .variational import VariationalState


@dataclass(frozen=True)
class GroundStateStep:
    step: int
    energy: float
    imaginary_residual: float
    variance: float
    acceptance_rate: float | None
    gradient_norm: float
    update_norm: float
    optimizer_metrics: dict[str, float | int | torch.Tensor] = field(default_factory=dict)


@dataclass
class GroundStateResult:
    history: list[GroundStateStep]
    best_energy: float
    best_state_dict: dict[str, torch.Tensor]


class GroundStateDriver:
    """Sampler- and optimizer-independent VMC ground-state driver."""

    def __init__(
        self,
        variational_state: VariationalState,
        optimizer: GroundStateOptimizer,
    ) -> None:
        self.state = variational_state
        self.optimizer = optimizer
        self.step_count = 0

    def advance(self) -> GroundStateStep:
        statistics = self.state.expect_energy(resample=True)
        optimizer_step: OptimizerStep = self.optimizer.step(
            self.state.model, statistics
        )
        result = GroundStateStep(
            self.step_count,
            float(statistics.energy.real),
            float(statistics.imaginary_residual),
            float(statistics.variance),
            statistics.acceptance_rate,
            optimizer_step.gradient_norm,
            optimizer_step.update_norm,
            optimizer_step.metrics,
        )
        self.step_count += 1
        self.state.reset()
        return result

    def run(
        self,
        steps: int,
        *,
        callback: Callable[[GroundStateStep, "GroundStateDriver"], bool | None] | None = None,
        report_every: int | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> GroundStateResult:
        if steps < 1:
            raise ValueError("steps must be positive")
        history: list[GroundStateStep] = []
        best_energy = float("inf")
        best_state: dict[str, torch.Tensor] = {}
        for _ in range(steps):
            state_before_update = {
                key: value.detach().cpu().clone()
                for key, value in self.state.model.state_dict().items()
            }
            result = self.advance()
            history.append(result)
            if result.energy < best_energy:
                best_energy = result.energy
                best_state = state_before_update
            if report_every and (
                result.step % report_every == 0 or len(history) == steps
            ):
                print(format_step(result))
            if callback is not None and callback(result, self) is False:
                break
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                path,
                self.state.model,
                model_config={"class": type(self.state.model).__name__},
                run_config={
                    "sampler": type(self.state.sampler).__name__,
                    "optimizer": type(self.optimizer).__name__,
                },
                step=self.step_count,
            )
        return GroundStateResult(history, best_energy, best_state)


def format_step(result: GroundStateStep) -> str:
    sampling = (
        "exact"
        if result.acceptance_rate is None
        else f"accept={result.acceptance_rate:.3f}"
    )
    extra = ""
    if "effective_rank" in result.optimizer_metrics:
        extra = (
            f" rank={result.optimizer_metrics['effective_rank']}"
            f" cond={result.optimizer_metrics['condition_number']:.3e}"
        )
    return (
        f"step={result.step:5d} energy={result.energy:.12f} "
        f"imag={result.imaginary_residual:.2e} var={result.variance:.3e} "
        f"{sampling} |update|={result.update_norm:.3e}{extra}"
    )
