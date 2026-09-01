"""NVIDIA 单卡与 NCCL 多卡基态性能基准。"""

import gc
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import (  # noqa: E402
    AmplitudePhaseRBM,
    GroundStateDriver,
    LogJacobian,
    MetropolisSampler,
    ParallelContext,
    SR,
    VariationalState,
    tilted_field_ising,
    update_run_metadata,
)


NUM_SITES = 14
COUPLING = 1.0
FIELD_X = 0.5
FIELD_Z = 0.5
PERIODIC = False

DTYPE = torch.float32
SEED = 666
NUM_CHAINS = 10_000
THERMAL_SWEEPS = 20
SWEEPS = 1
OPTIMIZATION_STEPS = 100
REPORT_EVERY = 10
REPETITIONS = 5
ALPHA_NUM = 5
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_results"


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def distribution(values: list[float]) -> dict[str, float | int]:
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


def reset_peak_memory(context: ParallelContext) -> None:
    gc.collect()
    with torch.cuda.device(context.device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gather_peak_memory(context: ParallelContext) -> list[dict[str, str | int]]:
    with torch.cuda.device(context.device):
        torch.cuda.synchronize()
        local_peak = {
            "device": f"cuda:{context.local_rank}",
            "rank": context.rank,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    gathered = context.all_gather_object(local_peak)
    return sorted(
        (dict(item) for item in gathered),
        key=lambda item: int(item["rank"]),
    )


def run_benchmark(
    context: ParallelContext, run_name: str, run_index: int
) -> dict[str, object] | None:
    reset_peak_memory(context)
    run_label = f"{run_name}_run{run_index:02d}"
    history_path = BENCHMARK_DIR / f"{run_label}_steps.csv"
    metadata_path = BENCHMARK_DIR / f"{run_label}_metadata.json"

    system = tilted_field_ising(
        num_sites=NUM_SITES,
        coupling=COUPLING,
        field_x=FIELD_X,
        field_z=FIELD_Z,
        periodic=PERIODIC,
    )
    model = AmplitudePhaseRBM(
        num_visible=NUM_SITES,
        num_hidden=ALPHA_NUM * NUM_SITES,
        dtype=DTYPE,
        device=context.device,
        seed=SEED,
    )
    sampler = MetropolisSampler(
        num_chains=NUM_CHAINS,
        thermal_sweeps=THERMAL_SWEEPS,
        sweeps=SWEEPS,
        sweep_size=None,
    )
    state = VariationalState(
        system=system,
        model=model,
        sampler=sampler,
        seed=SEED,
        num_gpus=context.world_size,
        parallel_context=context,
    )
    optimizer = SR(
        learning_rate=0.05,
        regularization=1e-3,
        rcond=1e-12,
        jacobian=LogJacobian(method="auto"),
        solver_device="auto",
    )

    if context.is_main:
        print(f"\nbenchmark run           = {run_index}/{REPETITIONS}")
        print(f"primary device          = cuda:0")
        print(f"number of GPUs          = {context.world_size}")
        print(f"parallel mode           = {context.parallel_mode}")
        print(f"communication backend   = {context.backend or 'none'}")
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
            "num_chains_global": NUM_CHAINS,
            "launch_command": (
                "python examples/ground_state/nvidia_ground_state.py"
                if context.world_size == 1
                else (
                    "torchrun --standalone "
                    f"--nproc_per_node={context.world_size} "
                    "examples/ground_state/nvidia_ground_state.py"
                )
            ),
        },
    )

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
    memory_peaks = gather_peak_memory(context)
    context.barrier()
    if not context.is_main:
        return None

    update_run_metadata(
        metadata_path,
        benchmark_run_index=run_index,
        benchmark_repetitions=REPETITIONS,
        timing_scope=(
            "GroundStateDriver.run optimization only; "
            "excludes ED and full-sum evaluation"
        ),
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
            int(item["max_memory_allocated_bytes"]) for item in memory_peaks
        ),
        "max_device_peak_memory_reserved_bytes": max(
            int(item["max_memory_reserved_bytes"]) for item in memory_peaks
        ),
    }
    print(f"best sampled energy      = {result.best_energy:.12f}")
    print(f"step history             = {history_path}")
    print(f"run metadata             = {metadata_path}")
    return record


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA benchmark requires a CUDA-enabled PyTorch build")
    context = ParallelContext.from_env(device_type="cuda")
    run_name = f"real_pair_sr_cuda0_{context.world_size}gpu"
    summary_path = BENCHMARK_DIR / f"{run_name}_summary.json"
    try:
        records = [
            run_benchmark(context, run_name, run_index)
            for run_index in range(1, REPETITIONS + 1)
        ]
        context.barrier()
        if not context.is_main:
            return
        run_records = [record for record in records if record is not None]
        metric_names = (
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
        for metric_name in metric_names:
            values = [
                float(record[metric_name])
                for record in run_records
                if record[metric_name] is not None
            ]
            aggregate_statistics[metric_name] = distribution(values)
        summary = {
            "experiment_label": run_name,
            "repetitions": REPETITIONS,
            "seed_per_run": SEED,
            "parallel_mode": context.parallel_mode,
            "communication_backend": context.backend,
            "world_size": context.world_size,
            "timing_scope": (
                "GroundStateDriver.run optimization only; "
                "excludes ED and full-sum evaluation"
            ),
            "steady_step_warmup_excluded": 1,
            "aggregate_basis": "one mean steady-state value per measurement run",
            "percentile_method": "linear_interpolation",
            "runs": run_records,
            "aggregate_statistics": aggregate_statistics,
        }
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
        print(f"\nbenchmark summary        = {summary_path}")
    finally:
        context.close()


if __name__ == "__main__":
    main()
