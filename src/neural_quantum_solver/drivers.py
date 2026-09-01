from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
import torch

from .devices import synchronize_devices
from .estimators import ShardedSampledEnergy
from .models import AmplitudePhaseNQS
from .real_estimators import RealPairShardedSampledEnergy
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
    num_samples: int
    sampling_seconds: float
    energy_seconds: float
    optimization_seconds: float
    step_seconds: float
    optimizer_metrics: dict[str, float | int | torch.Tensor] = field(default_factory=dict)


@dataclass
class GroundStateResult:
    history: list[GroundStateStep]
    best_energy: float
    best_state_dict: dict[str, torch.Tensor]
    total_seconds: float


class GroundStateDriver:
    """与采样器和优化器解耦的 VMC 基态驱动器。"""

    def __init__(
        self,
        variational_state: VariationalState,
        optimizer: GroundStateOptimizer,
    ) -> None:
        self.state = variational_state
        self.optimizer = optimizer
        self.step_count = 0

    def advance(self) -> GroundStateStep:
        devices = self.state.device_mesh.devices
        synchronize_devices(devices)
        step_started = perf_counter()
        sampling_started = perf_counter()
        self.state.sample()
        synchronize_devices(devices)
        sampling_seconds = perf_counter() - sampling_started

        energy_started = perf_counter()
        statistics = self.state.expect_energy()
        synchronize_devices(devices)
        energy_seconds = perf_counter() - energy_started

        optimization_started = perf_counter()
        optimizer_step: OptimizerStep = self.optimizer.step(
            self.state.model, statistics
        )
        synchronize_devices(devices)
        optimization_seconds = perf_counter() - optimization_started
        step_seconds = perf_counter() - step_started
        num_samples = (
            statistics.num_samples
            if isinstance(
                statistics, (ShardedSampledEnergy, RealPairShardedSampledEnergy)
            )
            else statistics.configurations.shape[0]
        )
        result = GroundStateStep(
            self.step_count,
            float(statistics.energy.real),
            float(statistics.imaginary_residual),
            float(statistics.variance),
            statistics.acceptance_rate,
            optimizer_step.gradient_norm,
            optimizer_step.update_norm,
            num_samples,
            sampling_seconds,
            energy_seconds,
            optimization_seconds,
            step_seconds,
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
        history_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        experiment_label: str | None = None,
        run_metadata: dict[str, object] | None = None,
    ) -> GroundStateResult:
        if steps < 1:
            raise ValueError("steps must be positive")
        history: list[GroundStateStep] = []
        best_energy = float("inf")
        best_state: dict[str, torch.Tensor] = {}
        history_file = Path(history_path) if history_path is not None else None
        if history_file is not None:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            with history_file.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=_HISTORY_FIELDS).writeheader()
        metadata_file = Path(metadata_path) if metadata_path is not None else None
        metadata = _run_metadata(self, experiment_label)
        if run_metadata is not None:
            metadata["run_config"] = run_metadata
        if metadata_file is not None:
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            _write_json(metadata_file, metadata)
        synchronize_devices(self.state.device_mesh.devices)
        run_started = perf_counter()
        for _ in range(steps):
            state_before_update = {
                key: value.detach().cpu().clone()
                for key, value in self.state.model.state_dict().items()
            }
            result = self.advance()
            history.append(result)
            if history_file is not None:
                _append_history(history_file, result)
            if result.energy < best_energy:
                best_energy = result.energy
                best_state = state_before_update
            if report_every and (
                result.step % report_every == 0 or len(history) == steps
            ):
                print(format_step(result))
            if callback is not None and callback(result, self) is False:
                break
        synchronize_devices(self.state.device_mesh.devices)
        total_seconds = perf_counter() - run_started
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
                    "device": str(self.state.device_mesh.primary),
                    "num_gpus": self.state.device_mesh.num_devices,
                    "representation": (
                        "real_pair"
                        if isinstance(self.state.model, AmplitudePhaseNQS)
                        else "complex"
                    ),
                },
                step=self.step_count,
            )
        metadata.update(
            {
                "completed_steps": len(history),
                "best_energy": best_energy,
                "total_seconds": total_seconds,
            }
        )
        if metadata_file is not None:
            _write_json(metadata_file, metadata)
        print(f"total optimization time = {total_seconds:.6f} s")
        return GroundStateResult(history, best_energy, best_state, total_seconds)


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
    if "solve_seconds" in result.optimizer_metrics:
        extra += f" solve={result.optimizer_metrics['solve_seconds']:.4f}s"
    return (
        f"step={result.step:5d} energy={result.energy:.12f} "
        f"imag={result.imaginary_residual:.2e} var={result.variance:.3e} "
        f"{sampling} samples={result.num_samples} |update|={result.update_norm:.3e} "
        f"step_time={result.step_seconds:.4f}s{extra}"
    )


_HISTORY_FIELDS = (
    "step", "energy", "variance", "imaginary_residual", "acceptance_rate",
    "num_samples", "gradient_norm", "update_norm", "sampling_seconds",
    "energy_seconds", "optimization_seconds", "step_seconds",
    "vmc_samples_per_second",
    "backward_seconds", "jacobian_seconds", "qgt_force_seconds",
    "solve_seconds", "qgt_dimension", "effective_rank", "condition_number",
    "fs_step_norm",
)


def _history_row(result: GroundStateStep) -> dict[str, object]:
    row = {
        "step": result.step,
        "energy": result.energy,
        "variance": result.variance,
        "imaginary_residual": result.imaginary_residual,
        "acceptance_rate": result.acceptance_rate,
        "num_samples": result.num_samples,
        "gradient_norm": result.gradient_norm,
        "update_norm": result.update_norm,
        "sampling_seconds": result.sampling_seconds,
        "energy_seconds": result.energy_seconds,
        "optimization_seconds": result.optimization_seconds,
        "step_seconds": result.step_seconds,
        "vmc_samples_per_second": result.num_samples
        / (result.sampling_seconds + result.energy_seconds),
    }
    for name in _HISTORY_FIELDS:
        if name in result.optimizer_metrics:
            value = result.optimizer_metrics[name]
            row[name] = value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value
    return row


def _append_history(path: Path, result: GroundStateStep) -> None:
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_HISTORY_FIELDS)
        writer.writerow(_history_row(result))


def _run_metadata(driver: GroundStateDriver, label: str | None) -> dict[str, object]:
    parameter = next(driver.state.model.parameters())
    return {
        "experiment_label": label,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": getattr(torch.version, "cuda", None),
        "musa_runtime_version": getattr(torch.version, "musa", None),
        "devices": [str(device) for device in driver.state.device_mesh.devices],
        "device_details": _device_details(driver.state.device_mesh.devices),
        "backend": driver.state.device_mesh.backend,
        "num_gpus": driver.state.device_mesh.num_devices,
        "dtype": str(parameter.dtype),
        "model": type(driver.state.model).__name__,
        "representation": (
            "real_pair"
            if isinstance(driver.state.model, AmplitudePhaseNQS)
            else "complex"
        ),
        "parameter_count": sum(value.numel() for value in driver.state.model.parameters()),
        "sampler": type(driver.state.sampler).__name__,
        "sampler_config": _public_scalar_settings(driver.state.sampler),
        "optimizer": type(driver.optimizer).__name__,
        "optimizer_config": _public_scalar_settings(driver.optimizer),
        "num_sites": driver.state.system.hilbert.num_sites,
        "seed": driver.state.seed,
        "hilbert_size": driver.state.system.hilbert.size,
        "hamiltonian_terms": len(driver.state.system.hamiltonian.terms),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def update_run_metadata(path: str | Path, **values: object) -> None:
    """向已有运行记录 JSON 中追加最终参考值和评估结果。"""
    target = Path(path)
    with target.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload.update(values)
    _write_json(target, payload)


def _public_scalar_settings(value: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, setting in vars(value).items():
        if name.startswith("_"):
            continue
        if setting is None or isinstance(setting, (bool, int, float, str)):
            result[name] = setting
        elif isinstance(setting, torch.dtype):
            result[name] = str(setting)
        elif name == "jacobian":
            result[name] = {
                "class": type(setting).__name__,
                **_public_scalar_settings(setting),
            }
    return result


def _device_details(devices: tuple[torch.device, ...]) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for device in devices:
        item: dict[str, object] = {"device": str(device)}
        if device.type == "cpu":
            item["name"] = platform.processor() or "CPU"
        else:
            backend = getattr(torch, device.type)
            try:
                properties = backend.get_device_properties(device)
                item["name"] = getattr(properties, "name", str(properties))
                item["total_memory_bytes"] = getattr(
                    properties, "total_memory", None
                )
            except (AttributeError, RuntimeError) as error:
                item["metadata_error"] = str(error)
        details.append(item)
    return details
