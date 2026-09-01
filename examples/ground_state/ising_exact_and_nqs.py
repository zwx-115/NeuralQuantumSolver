from pathlib import Path
import sys

import torch

# 无需预先安装软件包，即可从任意工作目录直接运行本示例。
# 常规使用仍建议先执行 `pip install -e .`。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import (
    AmplitudePhaseRBM,
    ExactDiagonalizer,
    tilted_field_ising,
)
from neural_quantum_solver.runners import ExactGroundStateRunner

# 在 NVIDIA 和摩尔线程之间切换时修改下面两个参数。
DEVICE = "cuda:0"  # NVIDIA: "cuda:0"; Moore Threads: "musa"
NUM_GPUS = 1

# 周期边界条件
periodic = True
n_sites = 8
alpha = 5

system = tilted_field_ising(n_sites, coupling=1.0, field_x=0.5, field_z=0.0, periodic=periodic)
exact = ExactDiagonalizer(dtype=torch.complex128).ground_state(system)
model = AmplitudePhaseRBM(
    n_sites, alpha * n_sites, dtype=torch.float32, device=DEVICE, seed=7
)
fit = ExactGroundStateRunner(
    learning_rate=0.02, steps=500, num_gpus=NUM_GPUS
).run(model, system)

print(f"exact ground energy: {exact.energy.item():.12f}")
print(f"best NQS energy:     {fit.best_energy:.12f}")
