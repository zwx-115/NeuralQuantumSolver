"""A：相互独立的晚时刻精确时间切片拟合实验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median, pstdev
from typing import Callable
import torch

from ..diagnostics.snapshots import ExactSnapshot, SnapshotFitResult, fit_snapshot
from ..models import NeuralQuantumState
from ..systems import PhysicalSystem


ModelFactory = Callable[[int, ExactSnapshot], NeuralQuantumState]
ResultCallback = Callable[[dict[str, float | int | str | bool]], None]


@dataclass(frozen=True)
class SnapshotFittingSettings:
    max_steps: int = 1_000
    tolerance: float = 1e-10
    learning_rate: float = 1e-3
    seeds: tuple[int, ...] = (1, 2, 3)


class SnapshotFittingExperiment:
    """将每个模型族独立拟合到每个冻结精确态。

    工厂函数接收 ``(seed, snapshot)``，轨迹 NQS 因而可以恢复与
    ``snapshot.time`` 匹配的 checkpoint；拟合结果不会写回源 checkpoint。
    """

    def __init__(
        self,
        system: PhysicalSystem,
        snapshots: list[ExactSnapshot],
        *,
        families: dict[str, ModelFactory],
        settings: SnapshotFittingSettings = SnapshotFittingSettings(),
        family_seeds: dict[str, tuple[int, ...]] | None = None,
        family_learning_rates: dict[str, float] | None = None,
    ) -> None:
        if not families:
            raise ValueError("at least one model family is required")
        self.system = system
        self.snapshots = snapshots
        self.families = families
        self.settings = settings
        self.family_seeds = family_seeds or {}
        self.family_learning_rates = family_learning_rates or {}
        unknown = set(self.family_seeds) - set(self.families)
        if unknown:
            raise ValueError(f"family_seeds contains unknown families: {sorted(unknown)}")
        unknown_rates = set(self.family_learning_rates) - set(self.families)
        if unknown_rates:
            raise ValueError(
                f"family_learning_rates contains unknown families: {sorted(unknown_rates)}"
            )
        if any(rate <= 0 for rate in self.family_learning_rates.values()):
            raise ValueError("family learning rates must be positive")

    def run(
        self, *, on_result: ResultCallback | None = None
    ) -> tuple[list[dict[str, float | int | str | bool]], dict[str, dict[str, torch.Tensor]]]:
        rows: list[dict[str, float | int | str | bool]] = []
        checkpoints: dict[str, dict[str, torch.Tensor]] = {}
        for family, factory in self.families.items():
            learning_rate = self.family_learning_rates.get(
                family, self.settings.learning_rate
            )
            for snapshot in self.snapshots:
                for seed in self.family_seeds.get(family, self.settings.seeds):
                    torch.manual_seed(seed)
                    result = fit_snapshot(
                        factory(seed, snapshot), self.system, snapshot,
                        optimizer_factory=lambda parameters: torch.optim.Adam(
                            parameters, lr=learning_rate
                        ),
                        max_steps=self.settings.max_steps,
                        tolerance=self.settings.tolerance,
                    )
                    key = f"{family}_t{snapshot.time:g}_seed{seed}"
                    checkpoints[key] = result.best_state_dict
                    row = _row(family, seed, result)
                    rows.append(row)
                    if on_result is not None:
                        on_result(row)
        return rows, checkpoints

    def config(self) -> dict[str, object]:
        return {
            "experiment": "late_time_snapshot_fitting",
            "settings": asdict(self.settings),
            "families": sorted(self.families),
            "family_seeds": self.family_seeds,
            "family_learning_rates": self.family_learning_rates,
            "times": [snapshot.time for snapshot in self.snapshots],
        }


def _row(family: str, seed: int, result: SnapshotFitResult) -> dict[str, float | int | str | bool]:
    return {
        "family": family,
        "seed": seed,
        "time": result.time,
        "initial_loss": result.initial_loss,
        "best_loss": result.best_loss,
        "final_loss": result.final_loss,
        "iterations": result.iterations,
        "converged": result.converged,
    }


def summarize_snapshot_fits(
    rows: list[dict[str, float | int | str | bool]],
) -> list[dict[str, float | int | str]]:
    """按模型族和时间汇总各随机种子的最佳拟合 loss。"""
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        family = row["family"]
        time = row["time"]
        best_loss = row["best_loss"]
        if not isinstance(family, str) or not isinstance(time, float):
            raise TypeError("snapshot fitting rows must contain string family and float time")
        if not isinstance(best_loss, float):
            raise TypeError("snapshot fitting rows must contain float best_loss")
        grouped.setdefault((family, time), []).append(best_loss)
    return [
        {
            "family": family,
            "time": time,
            "num_seeds": len(losses),
            "best_loss_min": min(losses),
            "best_loss_median": median(losses),
            "best_loss_mean": mean(losses),
            "best_loss_std": pstdev(losses),
        }
        for (family, time), losses in sorted(grouped.items())
    ]
