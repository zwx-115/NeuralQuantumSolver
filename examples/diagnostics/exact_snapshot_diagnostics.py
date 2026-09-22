"""可直接运行的小系统 A/B/C 诊断示例。

此示例以 ED 时间切片为目标，并将每个时间点的真实 p-tVMC checkpoint、
同尺寸随机重启和更大随机重启分别拟合及比较。
"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import ComplexRBM, ExactDiagonalizer, tilted_field_ising  # noqa: E402
from neural_quantum_solver.diagnostics import evolve_exact_snapshots  # noqa: E402
from neural_quantum_solver.experiments import (  # noqa: E402
    SnapshotFittingExperiment,
    SnapshotFittingSettings,
    TrajectoryPoint,
    diagnose_trajectory,
    save_diagnostic_artifacts,
    summarize_snapshot_fits,
)


NUM_SITES = 12
HIDDEN_MULTIPLIER = 4
LARGER_HIDDEN_MULTIPLIER = 8
TIMES = [round(0.1 * index, 10) for index in range(21)]
FIT_STEPS = 1_000
LEARNING_RATE = 1e-2
LARGER_LEARNING_RATE = 3e-3
SEEDS = (1, 2, 3, 4, 5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.complex128
TRAJECTORY_DT = 0.05
OUTPUT_DIR = PROJECT_ROOT / "benchmark_results" / "diagnostics" / "late_time_snapshot_adam_1000_seed5"
TRAJECTORY_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "projected_ptvmc_trajectory"


# 淬火前 Hamiltonian：其基态定义 t=0 的精确初态。
initial_system = tilted_field_ising(NUM_SITES, coupling=0.0, field_x=0.5, field_z=0.0)
# 淬火后 Hamiltonian：ED 和后续 p-tVMC 都应使用它进行实时演化。
evolution_system = tilted_field_ising(NUM_SITES, coupling=1.0, field_x=0.5, field_z=0.5)
initial_exact = ExactDiagonalizer(
    dtype=DTYPE, device=DEVICE
).ground_state(initial_system).state
snapshots = evolve_exact_snapshots(
    evolution_system, initial_exact, TIMES, dtype=DTYPE, device=DEVICE
)
print("开始 ED 时间切片诊断")
print(f"num_sites (L)          = {NUM_SITES}")
print(f"dtype / device         = {DTYPE} / {DEVICE}")
print(f"时间切片                = {TIMES}")
print(f"同尺寸 / 更大 hidden   = {HIDDEN_MULTIPLIER * NUM_SITES} / {LARGER_HIDDEN_MULTIPLIER * NUM_SITES}")


def same_size(seed: int, _snapshot) -> ComplexRBM:
    return ComplexRBM(
        NUM_SITES, HIDDEN_MULTIPLIER * NUM_SITES,
        dtype=DTYPE, device=DEVICE, seed=seed,
    )


def larger(seed: int, _snapshot) -> ComplexRBM:
    return ComplexRBM(
        NUM_SITES, LARGER_HIDDEN_MULTIPLIER * NUM_SITES,
        dtype=DTYPE, device=DEVICE, seed=seed,
    )


def trajectory_model(_seed: int, snapshot) -> ComplexRBM:
    """恢复与 ED 时间切片相同时间点的 projected p-tVMC RBM。"""
    path = TRAJECTORY_DIRECTORY / f"ptvmc_t{snapshot.time:.2f}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少轨迹 checkpoint：{path}；请先运行 projected_time_evolution.py"
        )
    payload = torch.load(path, map_location=DEVICE, weights_only=True)
    config = payload["model_config"]
    if config["num_visible"] != NUM_SITES:
        raise ValueError(
            f"checkpoint 的 L={config['num_visible']}，与诊断的 L={NUM_SITES} 不一致"
        )
    model = ComplexRBM(
        config["num_visible"], config["num_hidden"],
        dtype=DTYPE, device=DEVICE,
    )
    model.load_state_dict(payload["model_state_dict"])
    return model


experiment = SnapshotFittingExperiment(
    evolution_system,
    snapshots,
    families={
        "trajectory": trajectory_model,
        "same_size_restart": same_size,
        "larger_restart": larger,
    },
    settings=SnapshotFittingSettings(
        max_steps=FIT_STEPS, learning_rate=LEARNING_RATE, seeds=SEEDS,
    ),
    family_seeds={"trajectory": (SEEDS[0],)},
    family_learning_rates={"larger_restart": LARGER_LEARNING_RATE},
)


def report_fit(row) -> None:
    """打印一个冻结时间切片拟合完成后的关键指标。"""
    print(
        f"拟合完成：t={row['time']:.2f}，{row['family']}，seed={row['seed']}，"
        f"初始 loss={row['initial_loss']:.3e}，最佳 loss={row['best_loss']:.3e}",
        flush=True,
    )


print("开始 A：同尺寸与更大 RBM 的独立拟合", flush=True)
fit_rows, checkpoints = experiment.run(on_result=report_fit)

trajectory_rows, spectra = diagnose_trajectory(
    evolution_system,
    [TrajectoryPoint(snapshot.time, trajectory_model(SEEDS[0], snapshot)) for snapshot in snapshots],
)
print("完成 B/C 占位诊断，正在写入结果文件", flush=True)
destination = save_diagnostic_artifacts(
    OUTPUT_DIR,
    config=experiment.config() | {
        "dtype": str(DTYPE),
        "device": DEVICE,
        "initial_system": {
            "coupling": 0.0, "field_x": 0.5, "field_z": 0.0, "periodic": False,
        },
        "evolution_system": {
            "coupling": 1.0, "field_x": 0.5, "field_z": 0.5, "periodic": False,
        },
        "trajectory": {"dt": TRAJECTORY_DT, "checkpoint_directory": str(TRAJECTORY_DIRECTORY)},
    },
    rows=fit_rows + trajectory_rows,
    summary={
        "snapshot_fit_seed_statistics": summarize_snapshot_fits(fit_rows),
        "qgt_spectra": spectra,
        "trajectory_rows": trajectory_rows,
    },
    checkpoints=checkpoints,
)
print(f"Wrote diagnostics to {destination}")
