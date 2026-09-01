"""使用 Adam 和完整 Hilbert 空间求和优化 NQS 基态。"""

from pathlib import Path
import sys

import torch


# 无需先执行 `pip install -e .`，也可以直接运行这个示例。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import (  # noqa: E402
    Adam,
    AmplitudePhaseRBM,
    ExactDiagonalizer,
    FullSumState,
    GroundStateDriver,
    tilted_field_ising,
)
from neural_quantum_solver.estimators import exact_energy  # noqa: E402


# ---------------------------------------------------------------------------
# 可编辑的实验参数
# ---------------------------------------------------------------------------
NUM_SITES = 14
COUPLING = 1.0
FIELD_X = 0.5
FIELD_Z = 0.5
PERIODIC = False

HIDDEN_DENSITY = 4
LEARNING_RATE = 0.01
OPTIMIZATION_STEPS = 300
REPORT_EVERY = 10

DTYPE = torch.float32
DEVICE = "cuda:0"  # NVIDIA: "cuda:0"; Moore Threads: "musa"
NUM_GPUS = 1
SEED = 7


# ED 与 NQS 使用完全相同的物理系统对象。
system = tilted_field_ising(
    num_sites=NUM_SITES,
    coupling=COUPLING,
    field_x=FIELD_X,
    field_z=FIELD_Z,
    periodic=PERIODIC,
)

model = AmplitudePhaseRBM(
    num_visible=NUM_SITES,
    num_hidden=HIDDEN_DENSITY * NUM_SITES,
    dtype=DTYPE,
    device=DEVICE,
    seed=SEED,
)

# FullSumState 枚举全部 2**NUM_SITES 个构型及其精确 Born 概率，
# 因此不存在 Markov 链、热化 sweep 或 MC 接受率。
variational_state = FullSumState(
    system=system,
    model=model,
    seed=SEED,
    num_gpus=NUM_GPUS,
)

optimizer = Adam(
    learning_rate=LEARNING_RATE,
)

print(f"device                 = {DEVICE}")
print(f"number of GPUs         = {NUM_GPUS}")
print(f"Hilbert-space size     = {system.hilbert.size}")
print("sampling               = exact full summation")

result = GroundStateDriver(variational_state, optimizer).run(
    steps=OPTIMIZATION_STEPS,
    report_every=REPORT_EVERY,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "ground_state_adam_full_sum.pt",
)

# 下面两个量都没有采样噪声：第一个是精确本征值，第二个是最终 NQS
# 波函数的精确能量期望值。
exact = ExactDiagonalizer(dtype=torch.complex128, device="cpu").ground_state(system)
final_nqs_energy = exact_energy(model, system).energy.item()

print(f"exact ground energy    = {exact.energy.item():.12f}")
print(f"best NQS energy        = {result.best_energy:.12f}")
print(f"final NQS energy       = {final_nqs_energy:.12f}")
print(f"final energy error     = {final_nqs_energy - exact.energy.item():.3e}")
