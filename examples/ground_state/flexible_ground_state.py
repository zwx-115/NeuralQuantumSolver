"""可配置的 CUDA/MUSA 基态计算示例。"""

from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

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
    ExactDiagonalizer,
    ExactSampler,
    GroundStateDriver,
    LogJacobian,
    MetropolisSampler,
    SR,
    VariationalState,
    tilted_field_ising,
    update_run_metadata,
)
from neural_quantum_solver.estimators import exact_energy  # noqa: E402


# ---------------------------------------------------------------------------
# 设备设置：切换硬件时只需要修改下面两行。
# ---------------------------------------------------------------------------
DEVICE = "cuda:0"  # NVIDIA 使用 "cuda:0"；摩尔线程使用 "musa:0"
NUM_GPUS = 1       # 可设为 1、2、4……；从 DEVICE 开始连续使用显卡


# 物理系统参数。
NUM_SITES = 10
COUPLING = 1.0
FIELD_X = 0.5
FIELD_Z = 0.5
PERIODIC = False

# 完整的加速器训练路径只使用这个实数数据类型。
DTYPE = torch.float32
SEED = 666
OPTIMIZATION_STEPS = 100
REPORT_EVERY = 10
RUN_NAME = f"real_pair_sr_{DEVICE.replace(':', '')}_{NUM_GPUS}gpu"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_results"
HISTORY_PATH = BENCHMARK_DIR / f"{RUN_NAME}_steps.csv"
METADATA_PATH = BENCHMARK_DIR / f"{RUN_NAME}_metadata.json"
ALPHA_NUM = 5

SCRIPT_STARTED = perf_counter()


system = tilted_field_ising(
    num_sites=NUM_SITES,
    coupling=COUPLING,
    field_x=FIELD_X,
    field_z=FIELD_Z,
    periodic=PERIODIC,
)

# 选择一个模型。下面两个注释掉的模型可以使用相同的变分态和驱动器。
model = AmplitudePhaseRBM(
    num_visible=NUM_SITES,
    num_hidden= ALPHA_NUM* NUM_SITES,
    dtype=DTYPE,
    device=DEVICE,
    seed=SEED,
)
# model = AmplitudePhaseFNN([NUM_SITES, 4 * NUM_SITES, 1], dtype=DTYPE,
#                           device=DEVICE, seed=SEED)
# model = AmplitudePhaseTable(NUM_SITES, dtype=DTYPE, device=DEVICE)

# 选择一个采样器。MC 保留样本总数始终为 num_chains * sweeps，
# 不会随 NUM_GPUS 改变。
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

# 选择一个优化器。LogJacobian 仅供 SR 使用，Adam 不会计算它。
# solver_device="auto" 表示在第一张 CUDA 或 MUSA 显卡上求解。
optimizer = SR(
    learning_rate=0.05,
    regularization=1e-3,
    rcond=1e-12,
    # auto：AmplitudePhaseRBM 使用解析导数，通用实数网络使用 vmap。
    jacobian=LogJacobian(method="auto"),
    solver_device="auto",
)
# optimizer = Adam(learning_rate=0.001)

print(f"primary device          = {DEVICE}")
print(f"number of GPUs          = {NUM_GPUS}")
print(f"model                    = {type(model).__name__}")
print(f"sampler                  = {type(sampler).__name__}")
print(f"optimizer                = {type(optimizer).__name__}")

result = GroundStateDriver(state, optimizer).run(
    steps=OPTIMIZATION_STEPS,
    report_every=REPORT_EVERY,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "ground_state.pt",
    history_path=HISTORY_PATH,
    metadata_path=METADATA_PATH,
    experiment_label=RUN_NAME,
    run_metadata={
        "coupling": COUPLING,
        "field_x": FIELD_X,
        "field_z": FIELD_Z,
        "periodic": PERIODIC,
        "optimization_steps": OPTIMIZATION_STEPS,
    },
)

# NQS 的 full-sum 评估仍在 DEVICE 上运行，因此后端错误会直接显示，
# 不会被 CPU 回退掩盖。稠密 ED 是独立的 CPU complex128 参考计算，
# 不参与 NQS 优化。
exact = ExactDiagonalizer(dtype=torch.complex128, device="cpu").ground_state(system)
final_full_sum_energy = exact_energy(model, system).energy.item()
steady_steps = result.history[1:] or result.history
solve_times = [
    step.optimizer_metrics["solve_seconds"]
    for step in steady_steps
    if "solve_seconds" in step.optimizer_metrics
]
update_run_metadata(
    METADATA_PATH,
    exact_ground_energy=exact.energy.item(),
    final_nqs_full_sum_energy=final_full_sum_energy,
    final_energy_error=final_full_sum_energy - exact.energy.item(),
    mean_steady_step_seconds=mean(step.step_seconds for step in steady_steps),
    median_steady_step_seconds=median(step.step_seconds for step in steady_steps),
    mean_steady_solve_seconds=mean(solve_times) if solve_times else None,
    end_to_end_seconds=perf_counter() - SCRIPT_STARTED,
)

print(f"exact ground energy      = {exact.energy.item():.12f}")
print(f"best sampled energy      = {result.best_energy:.12f}")
print(f"final NQS full sum       = {final_full_sum_energy:.12f}")
print(f"full-sum error           = {final_full_sum_energy - exact.energy.item():.3e}")
print(f"step history             = {HISTORY_PATH}")
print(f"run metadata             = {METADATA_PATH}")
