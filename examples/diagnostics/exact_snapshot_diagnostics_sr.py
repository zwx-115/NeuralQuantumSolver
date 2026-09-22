"""对 SR p-tVMC 轨迹运行 A/B/C 全 Hilbert 空间诊断。"""

import csv
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import ComplexRBM, ExactDiagonalizer, tilted_field_ising  # noqa: E402
from neural_quantum_solver.diagnostics import evolve_exact_snapshots, fit_snapshot_sr  # noqa: E402
from neural_quantum_solver.solvers import OverlapSRSettings  # noqa: E402
from neural_quantum_solver.experiments import (  # noqa: E402
    TrajectoryPoint,
    diagnose_trajectory,
    save_diagnostic_artifacts,
    summarize_snapshot_fits,
)


NUM_SITES = 12
HIDDEN_MULTIPLIER = 4
LARGER_HIDDEN_MULTIPLIER = 8
# TIMES = [round(0.1 * index, 10) for index in range(21)]
TIMES = [0.0, 0.5, 1.0, 1.3, 1.5, 1.6, 2.0]
FIT_STEPS = 100
SEEDS = (1, 2, 3, 4, 5)
DTYPE = torch.complex128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAJECTORY_DT = 0.05
TRAJECTORY_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "projected_ptvmc_trajectory_sr"
OUTPUT_DIR = PROJECT_ROOT / "benchmark_results" / "diagnostics" / "late_time_snapshot_sr_1000_seed5"
SR_SETTINGS = OverlapSRSettings(
    learning_rate=0.05,
    regularization=1e-2,
    rcond=1e-12,
    jacobian_chunk_size=1024,
    max_backtracks=8,
    backtrack_factor=0.5,
)


initial_system = tilted_field_ising(NUM_SITES, coupling=0.0, field_x=0.5, field_z=0.0)
evolution_system = tilted_field_ising(NUM_SITES, coupling=1.0, field_x=0.5, field_z=0.5)
initial_exact = ExactDiagonalizer(dtype=DTYPE, device=DEVICE).ground_state(initial_system).state
snapshots = evolve_exact_snapshots(evolution_system, initial_exact, TIMES, dtype=DTYPE, device=DEVICE)


def same_size(seed: int) -> ComplexRBM:
    """构造同尺寸的独立随机重启 RBM。"""
    return ComplexRBM(NUM_SITES, HIDDEN_MULTIPLIER * NUM_SITES, dtype=DTYPE, device=DEVICE, seed=seed)


def larger(seed: int) -> ComplexRBM:
    """构造更大尺寸的独立随机重启 RBM。"""
    return ComplexRBM(NUM_SITES, LARGER_HIDDEN_MULTIPLIER * NUM_SITES, dtype=DTYPE, device=DEVICE, seed=seed)


def trajectory_model(snapshot_time: float) -> ComplexRBM:
    """恢复与 ED 时间切片相同时间的 SR p-tVMC checkpoint。"""
    path = TRAJECTORY_DIRECTORY / f"ptvmc_t{snapshot_time:.2f}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 SR trajectory checkpoint：{path}；请先运行 projected_time_evolution_sr.py")
    payload = torch.load(path, map_location=DEVICE, weights_only=True)
    optimizer = payload.get("run_config", {}).get("projection_optimizer", {}).get("name")
    if optimizer != "overlap_sr":
        raise ValueError(f"{path} 不是 overlap-SR trajectory checkpoint")
    config = payload["model_config"]
    model = ComplexRBM(config["num_visible"], config["num_hidden"], dtype=DTYPE, device=DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    return model


def run_fit(
    family: str,
    seed: int,
    snapshot,
    factory,
    history_writer: csv.DictWriter,
) -> tuple[dict[str, float | int | str | bool], dict[str, torch.Tensor]]:
    """执行一个 SR 快照拟合，并逐步写入 SR 数值历史。"""
    def on_step(record, iteration: int) -> None:
        history_writer.writerow({
            "family": family, "seed": seed, "time": snapshot.time,
        } | record.row(iteration))
        # 每十步报告一次进度，便于长时间 SR 拟合时观察当前任务是否仍在推进。
        if iteration % 10 == 0:
            print(
                f"拟合进度：t={snapshot.time:.2f}，{family}，seed={seed}，"
                f"step={iteration}/{FIT_STEPS}，loss={record.loss_after:.3e}",
                flush=True,
            )

    result = fit_snapshot_sr(
        factory(), evolution_system, snapshot,
        settings=SR_SETTINGS, max_steps=FIT_STEPS, on_step=on_step,
    )
    return ({
        "family": family, "seed": seed, "time": result.time,
        "initial_loss": result.initial_loss, "best_loss": result.best_loss,
        "final_loss": result.final_loss, "iterations": result.iterations,
        "converged": result.converged,
    }, result.best_state_dict)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
history_fields = [
    "family", "seed", "time", "iteration", "loss_before", "loss_after", "accepted",
    "backtracks", "step_scale", "gradient_norm", "update_norm", "fs_step_norm",
    "qgt_rank", "qgt_condition_number",
]
fit_rows: list[dict[str, float | int | str | bool]] = []
checkpoints: dict[str, dict[str, torch.Tensor]] = {}
print("开始 SR A/B/C 诊断", flush=True)
print(f"L={NUM_SITES}, times={TIMES}, SR={SR_SETTINGS.config()}", flush=True)
with (OUTPUT_DIR / "sr_fit_history.csv").open("w", newline="") as handle:
    history_writer = csv.DictWriter(handle, fieldnames=history_fields)
    history_writer.writeheader()
    families = (
        ("trajectory", (SEEDS[0],), lambda _seed, snapshot: lambda: trajectory_model(snapshot.time)),
        ("same_size_restart", SEEDS, lambda seed, _snapshot: lambda: same_size(seed)),
        ("larger_restart", SEEDS, lambda seed, _snapshot: lambda: larger(seed)),
    )
    for family, family_seeds, model_factory in families:
        for snapshot in snapshots:
            for seed in family_seeds:
                torch.manual_seed(seed)
                row, state = run_fit(
                    family, seed, snapshot, model_factory(seed, snapshot), history_writer,
                )
                fit_rows.append(row)
                checkpoints[f"{family}_t{snapshot.time:g}_seed{seed}"] = state
                print(
                    f"拟合完成：t={snapshot.time:.2f}，{family}，seed={seed}，"
                    f"初始 loss={row['initial_loss']:.3e}，最佳 loss={row['best_loss']:.3e}",
                    flush=True,
                )

trajectory_models = [TrajectoryPoint(snapshot.time, trajectory_model(snapshot.time)) for snapshot in snapshots]
trajectory_rows, spectra = diagnose_trajectory(evolution_system, trajectory_models)
destination = save_diagnostic_artifacts(
    OUTPUT_DIR,
    config={
        "experiment": "late_time_snapshot_fitting",
        "optimizer": {"name": "overlap_sr"} | SR_SETTINGS.config(),
        "fit_steps": FIT_STEPS,
        "restart_seeds": SEEDS,
        "family_seeds": {"trajectory": [SEEDS[0]]},
        "times": TIMES,
        "dtype": str(DTYPE), "device": DEVICE,
        "initial_system": {"coupling": 0.0, "field_x": 0.5, "field_z": 0.0, "periodic": False},
        "evolution_system": {"coupling": 1.0, "field_x": 0.5, "field_z": 0.5, "periodic": False},
        "trajectory": {"dt": TRAJECTORY_DT, "checkpoint_directory": str(TRAJECTORY_DIRECTORY)},
        "families": {
            "trajectory": "matching overlap-SR p-tVMC checkpoint",
            "same_size_restart": {"ansatz": "ComplexRBM", "hidden_multiplier": 4},
            "larger_restart": {"ansatz": "ComplexRBM", "hidden_multiplier": 8},
        },
    },
    rows=fit_rows + trajectory_rows,
    summary={
        "snapshot_fit_seed_statistics": summarize_snapshot_fits(fit_rows),
        "qgt_spectra": spectra,
        "trajectory_rows": trajectory_rows,
    },
    checkpoints=checkpoints,
)
print(f"SR A/B/C 诊断已写入 {destination}", flush=True)
