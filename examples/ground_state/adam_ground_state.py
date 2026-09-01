"""使用 Adam 和 Monte Carlo 采样优化 NQS 基态。"""

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
    GroundStateDriver,
    MetropolisSampler,
    VariationalState,
    tilted_field_ising,
)
from neural_quantum_solver.estimators import exact_energy  # noqa: E402


# 在 NVIDIA 和摩尔线程之间切换时修改下面两个参数。
DEVICE = "cuda:0"  # NVIDIA: "cuda:0"; Moore Threads: "musa"
NUM_GPUS = 1

# ED 与 NQS 共用同一个物理系统。
system = tilted_field_ising(
    num_sites=10,
    coupling=1.0,
    field_x=0.5,
    field_z=0.5,
    periodic=False,
)

model = AmplitudePhaseRBM(
    num_visible=system.hilbert.num_sites,
    num_hidden=4 * system.hilbert.num_sites,
    dtype=torch.float32,
    device=DEVICE,
    seed=7,
)

# 每个优化步保留的样本总数为 num_chains * sweeps。
# sweep_size=None 表示两个保留样本之间执行 num_sites 次局域 MC 更新。
sampler = MetropolisSampler(
    num_chains=100,
    thermal_sweeps=20,
    sweeps=100,
    sweep_size=None,
)

variational_state = VariationalState(
    system=system,
    model=model,
    sampler=sampler,
    seed=10,
    num_gpus=NUM_GPUS,
)

# Adam 使用一个标量 VMC 代理损失和一次反向传播，不会构造 SR 所需的
# 逐样本 LogJacobian。
optimizer = Adam(
    learning_rate=0.001,
)

driver = GroundStateDriver(variational_state, optimizer)

print(f"primary device         = {DEVICE}")
print(f"number of GPUs         = {NUM_GPUS}")
print(f"samples per step       = {sampler.num_chains * sampler.sweeps}")
result = driver.run(
    steps=1000,
    report_every=50,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "ground_state_adam.pt",
)

# 消除 Monte Carlo 噪声后评估最终 NQS，并与 ED 对比。
exact = ExactDiagonalizer(dtype=torch.complex128, device="cpu").ground_state(system)
final_full_sum_energy = exact_energy(model, system).energy.item()
print(f"exact ground energy    = {exact.energy.item():.12f}")
print(f"best sampled energy    = {result.best_energy:.12f}")
print(f"final NQS full sum     = {final_full_sum_energy:.12f}")
print(f"full-sum error         = {final_full_sum_energy - exact.energy.item():.3e}")
