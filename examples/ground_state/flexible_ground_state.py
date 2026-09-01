"""可配置的 CUDA/MUSA 基态计算示例。"""

import gc
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import (  # noqa: E402
    Adam,
    AmplitudePhaseFNN,
    AmplitudePhaseRBM,
    AmplitudePhaseTable,
    ExactSampler,
    GroundStateDriver,
    LogJacobian,
    MetropolisSampler,
    SR,
    VariationalState,
    expand_device_names,
    tilted_field_ising,
    update_run_metadata,
)


# ---------------------------------------------------------------------------
# 设备设置：切换硬件时只需要修改下面两行。
# ---------------------------------------------------------------------------
DEVICE = "musa:0"  # NVIDIA 使用 "cuda:0"；摩尔线程使用 "musa:0"
NUM_GPUS = 4      # 可设为 1、2、4……；从 DEVICE 开始连续使用显卡


# 物理系统参数。
NUM_SITES = 14
COUPLING = 1.0
FIELD_X = 0.5
FIELD_Z = 0.5
PERIODIC = False

# 完整的加速器训练路径只使用这个实数数据类型。
DTYPE = torch.float32
SEED = 666
OPTIMIZATION_STEPS = 100
REPORT_EVERY = 10
REPETITIONS = 5
RUN_NAME = f"real_pair_sr_{DEVICE.replace(':', '')}_{NUM_GPUS}gpu"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_results"
SUMMARY_PATH = BENCHMARK_DIR / f"{RUN_NAME}_summary.json"
ALPHA_NUM = 5

DEVICE_NAMES = expand_device_names(DEVICE, NUM_GPUS)


def percentile(values: list[float], percent: float) -> float:
    """使用线性插值计算百分位数。"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def distribution(values: list[float]) -> dict[str, float | int]:
    """返回可直接写入 JSON 的重复测量统计。"""
    if not values:
        raise ValueError("cannot summarize an empty measurement")
    return {
        "count": len(values),
        "mean": mean(values),
        "stddev": pstdev(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def reset_peak_device_memory() -> None:
    gc.collect()
    for device_name in DEVICE_NAMES:
        device = torch.device(device_name)
        if device.type == "cpu":
            continue
        backend = getattr(torch, device.type)
        with backend.device(device):
            backend.empty_cache()
            backend.reset_peak_memory_stats()


def peak_device_memory() -> list[dict[str, str | int]]:
    peaks: list[dict[str, str | int]] = []
    for device_name in DEVICE_NAMES:
        device = torch.device(device_name)
        if device.type == "cpu":
            continue
        backend = getattr(torch, device.type)
        with backend.device(device):
            backend.synchronize()
            peaks.append(
                {
                    "device": device_name,
                    "max_memory_allocated_bytes": int(
                        backend.max_memory_allocated()
                    ),
                    "max_memory_reserved_bytes": int(
                        backend.max_memory_reserved()
                    ),
                }
            )
    return peaks


def run_benchmark(run_index: int) -> dict[str, object]:
    reset_peak_device_memory()
    run_label = f"{RUN_NAME}_run{run_index:02d}"
    history_path = BENCHMARK_DIR / f"{run_label}_steps.csv"
    metadata_path = BENCHMARK_DIR / f"{run_label}_metadata.json"

    system = tilted_field_ising(
        num_sites=NUM_SITES,
        coupling=COUPLING,
        field_x=FIELD_X,
        field_z=FIELD_Z,
        periodic=PERIODIC,
    )

    # 选择一个模型。下面两个注释掉的模型可使用相同的状态和驱动器。
    model = AmplitudePhaseRBM(
        num_visible=NUM_SITES,
        num_hidden=ALPHA_NUM * NUM_SITES,
        dtype=DTYPE,
        device=DEVICE,
        seed=SEED,
    )
    # model = AmplitudePhaseFNN(
    #     [NUM_SITES, 4 * NUM_SITES, 1],
    #     dtype=DTYPE, device=DEVICE, seed=SEED,
    # )
    # model = AmplitudePhaseTable(NUM_SITES, dtype=DTYPE, device=DEVICE)

    sampler = MetropolisSampler(
        num_chains=10000,
        thermal_sweeps=20,
        sweeps=1,
        sweep_size=None,
    )
    # sampler = ExactSampler()

    state = VariationalState(
        system=system,
        model=model,
        sampler=sampler,
        seed=SEED,
        num_gpus=NUM_GPUS,
    )
    optimizer = SR(
        learning_rate=0.05,
        regularization=1e-3,
        rcond=1e-12,
        jacobian=LogJacobian(method="auto"),
        solver_device="auto",
    )
    # optimizer = Adam(learning_rate=0.001)

    print(f"\nbenchmark run           = {run_index}/{REPETITIONS}")
    print(f"primary device          = {DEVICE}")
    print(f"number of GPUs          = {NUM_GPUS}")
    print(f"model                    = {type(model).__name__}")
    print(f"sampler                  = {type(sampler).__name__}")
    print(f"optimizer                = {type(optimizer).__name__}")

    result = GroundStateDriver(state, optimizer).run(
        steps=OPTIMIZATION_STEPS,
        report_every=REPORT_EVERY,
        checkpoint_path=None,
        history_path=history_path,
        metadata_path=metadata_path,
        experiment_label=run_label,
        run_metadata={
            "coupling": COUPLING,
            "field_x": FIELD_X,
            "field_z": FIELD_Z,
            "periodic": PERIODIC,
            "optimization_steps": OPTIMIZATION_STEPS,
            "benchmark_repetitions": REPETITIONS,
            "benchmark_run_index": run_index,
        },
    )

    # 第 0 步包含初始化和预热，不纳入稳态性能统计。
    steady_steps = result.history[1:] or result.history
    metric_values = {
        "steady_step_seconds": [step.step_seconds for step in steady_steps],
        "sampling_seconds": [step.sampling_seconds for step in steady_steps],
        "energy_seconds": [step.energy_seconds for step in steady_steps],
        "optimization_seconds": [
            step.optimization_seconds for step in steady_steps
        ],
        "vmc_samples_per_second": [
            step.num_samples / (step.sampling_seconds + step.energy_seconds)
            for step in steady_steps
        ],
    }
    solve_times = [
        float(step.optimizer_metrics["solve_seconds"])
        for step in steady_steps
        if "solve_seconds" in step.optimizer_metrics
    ]
    if solve_times:
        metric_values["solve_seconds"] = solve_times

    performance_statistics = {
        name: distribution(values) for name, values in metric_values.items()
    }
    memory_peaks = peak_device_memory()
    update_run_metadata(
        metadata_path,
        benchmark_run_index=run_index,
        benchmark_repetitions=REPETITIONS,
        timing_scope="GroundStateDriver.run optimization only; excludes ED and full-sum evaluation",
        steady_step_warmup_excluded=1,
        percentile_method="linear_interpolation",
        performance_statistics=performance_statistics,
        peak_device_memory=memory_peaks,
    )

    record = {
        "run_index": run_index,
        "history_file": history_path.name,
        "metadata_file": metadata_path.name,
        "total_seconds": result.total_seconds,
        "best_energy": result.best_energy,
        "mean_steady_step_seconds": performance_statistics[
            "steady_step_seconds"
        ]["mean"],
        "mean_sampling_seconds": performance_statistics["sampling_seconds"][
            "mean"
        ],
        "mean_energy_seconds": performance_statistics["energy_seconds"]["mean"],
        "mean_optimization_seconds": performance_statistics[
            "optimization_seconds"
        ]["mean"],
        "mean_vmc_samples_per_second": performance_statistics[
            "vmc_samples_per_second"
        ]["mean"],
        "mean_solve_seconds": (
            performance_statistics["solve_seconds"]["mean"]
            if "solve_seconds" in performance_statistics
            else None
        ),
        "max_device_peak_memory_allocated_bytes": max(
            (
                int(item["max_memory_allocated_bytes"])
                for item in memory_peaks
            ),
            default=0,
        ),
        "max_device_peak_memory_reserved_bytes": max(
            (
                int(item["max_memory_reserved_bytes"])
                for item in memory_peaks
            ),
            default=0,
        ),
    }
    print(f"best sampled energy      = {result.best_energy:.12f}")
    print(f"step history             = {history_path}")
    print(f"run metadata             = {metadata_path}")
    return record


run_records = [
    run_benchmark(run_index)
    for run_index in range(1, REPETITIONS + 1)
]

aggregate_metric_names = (
    "total_seconds",
    "mean_steady_step_seconds",
    "mean_sampling_seconds",
    "mean_energy_seconds",
    "mean_optimization_seconds",
    "mean_vmc_samples_per_second",
    "mean_solve_seconds",
    "max_device_peak_memory_allocated_bytes",
    "max_device_peak_memory_reserved_bytes",
)
aggregate_statistics = {}
for metric_name in aggregate_metric_names:
    values = [
        float(record[metric_name])
        for record in run_records
        if record[metric_name] is not None
    ]
    if values:
        aggregate_statistics[metric_name] = distribution(values)

summary = {
    "experiment_label": RUN_NAME,
    "repetitions": REPETITIONS,
    "seed_per_run": SEED,
    "timing_scope": (
        "GroundStateDriver.run optimization only; excludes ED and "
        "full-sum evaluation"
    ),
    "steady_step_warmup_excluded": 1,
    "aggregate_basis": "one mean steady-state value per independent run",
    "percentile_method": "linear_interpolation",
    "runs": run_records,
    "aggregate_statistics": aggregate_statistics,
}
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, ensure_ascii=False)

print(f"\nbenchmark summary        = {SUMMARY_PATH}")
