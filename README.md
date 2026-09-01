# NeuralQuantumSolver：双实数振幅—相位基态求解器

`NeuralQuantumSolver` 是一个用于量子多体系统基态计算的 PyTorch 框架，同时支持 NVIDIA CUDA 与摩尔线程 MUSA。当前分支采用两个实数神经网络分别表示波函数的对数振幅和相位：

```text
log ψ(s) = A(s) + i Φ(s)
```

其中 `A(s)` 和 `Φ(s)` 都是实数。采样、局域能量、Adam、逐样本 Jacobian、量子几何张量（QGT）、随机重配置（SR）、多卡统计聚合以及 SR 线性方程求解均使用 `float32` 或 `float64` 实数张量，因此训练过程不依赖 GPU 的复数算子支持。

精确对角化（ED）用于小系统的精确参考计算，仍在 CPU 上采用 `complex128`，并与神经网络求解器共用同一套物理系统和哈密顿量定义。原有复数神经网络接口予以保留，用于回归测试和兼容旧代码。

## 安装与测试

在项目根目录执行：

```powershell
python -m pip install -e . --no-deps
python -m pytest
```

安装为可编辑包后，可以从任意示例目录导入 `neural_quantum_solver`，不会再出现因当前工作目录不同而导致的 `ModuleNotFoundError`。

## 切换 GPU 后端

计算设备只需在运行脚本中修改：

```python
DEVICE = "cuda:0"  # NVIDIA GPU
DEVICE = "musa:0"  # 摩尔线程 GPU
```

多卡数量通过以下参数设置：

```python
NUM_GPUS = 1  # 也可以设为 2、4 等
```

多卡模式要求使用同一种后端，并从主卡开始连续选择设备，例如 `cuda:0` 与 `NUM_GPUS = 4` 对应 `cuda:0` 到 `cuda:3`。程序不会在某个 GPU 算子失败时自动转移到 CPU；这样可以真实记录 CUDA 或 MUSA 后端的兼容性和性能问题。

## 灵活的基态计算脚本

主要示例位于：

```powershell
python examples/ground_state/flexible_ground_state.py
```

脚本顶部集中放置了物理系统、神经网络、采样器、优化器、设备、多卡数量和训练步数等参数。典型的可组合调用方式如下：

```python
import torch

from neural_quantum_solver import (
    AmplitudePhaseRBM,
    GroundStateDriver,
    LogJacobian,
    MetropolisSampler,
    SR,
    VariationalState,
    tilted_field_ising,
)

device = "cuda:0"  # 改为 "musa:0" 即可切换后端
num_gpus = 1
num_sites = 10

system = tilted_field_ising(
    num_sites=num_sites,
    coupling=1.0,
    field_x=1.0,
    field_z=0.0,
    periodic=True,
)

model = AmplitudePhaseRBM(
    num_visible=num_sites,
    num_hidden=20,
    dtype=torch.float32,
    device=device,
)

sampler = MetropolisSampler(
    num_chains=1_000,
    thermal_sweeps=20,
    sweeps=10,
    sweep_size=num_sites,
)

state = VariationalState(
    system=system,
    model=model,
    sampler=sampler,
    seed=0,
    num_gpus=num_gpus,
)

optimizer = SR(
    learning_rate=0.05,
    regularization=1.0e-3,
    jacobian=LogJacobian(method="auto"),
)

result = GroundStateDriver(state, optimizer).run(steps=200)
print(result.history[-1].energy)
```

## 可选神经网络模型

- `AmplitudePhaseRBM`：对数振幅和相位分别使用一个实数 RBM；其逐样本导数可使用解析公式。
- `AmplitudePhaseFNN`：对数振幅和相位分别使用一个通用实数前馈网络；适合配合 `vmap` 自动计算逐样本导数。
- `AmplitudePhaseTable`：对小系统中每个计算基矢的对数振幅和相位直接参数化，适合测试和验证。
- `RealRBM`：单个实数 RBM，可作为自定义振幅或相位子网络使用。

所有振幅—相位模型都继承 `AmplitudePhaseNQS`，并通过 `log_psi_parts` 返回两个实数张量：

```text
LogPsiParts(log_amplitude=A, phase=Φ)
```

## 采样与优化方法

采样器：

- `ExactSampler`：枚举全部基矢，适合小系统和无蒙特卡洛误差的基态优化。
- `MetropolisSampler`：使用多条马尔可夫链进行采样；接受率只由实数对数振幅决定。

优化器：

- `Adam`：使用能量梯度的一阶优化，不需要构造逐样本 Jacobian 或 QGT。
- `SR`：使用逐样本对数振幅和相位 Jacobian 构造实数 QGT，然后用 `torch.linalg.solve` 解正则化线性方程。

`LogJacobian` 支持：

- `method="auto"`：对 `AmplitudePhaseRBM` 自动采用解析导数，其他模型采用 `vmap`。
- `method="analytic"`：强制使用解析导数，仅适用于已经实现解析公式的模型。
- `method="vmap"`：通过 PyTorch `torch.func.vmap` 批量计算任意兼容模型的逐样本导数。
- `method="sequential"`：保留逐样本求导实现，主要用于验证和回归测试。

## 物理系统与边界条件

哈密顿量由 `PauliHamiltonian` 和 Pauli 项组合，因此 ED 与神经网络计算使用完全相同的物理系统定义。内置系统包括横场/倾斜场 Ising 模型，也可以使用 `PauliTerm` 自行构造包含 `X`、`Y`、`Z` 及多体乘积项的哈密顿量。

边界条件通常在系统构造函数中通过 `periodic` 设置：

```python
system = tilted_field_ising(
    num_sites=10,
    coupling=1.0,
    field_x=1.0,
    field_z=0.0,
    periodic=False,  # False：开放边界；True：周期边界
)
```

在 GPU 训练路径中，哈密顿量连接系数和局域能量均拆分为实部与虚部两个实数张量，因此包含 Pauli `Y` 的物理系统也不要求设备支持复数运算。

## 双实数 SR 公式

设

```text
log ψ(s) = A(s) + i Φ(s)，
J_A = ∂A/∂θ，
J_Φ = ∂Φ/∂θ。
```

对于实数参数，QGT 的实部为：

```text
S = Cov(J_A) + Cov(J_Φ)
```

能量梯度对应的实数力为：

```text
F = ⟨ centered(J_A) centered(E_real)
    + centered(J_Φ) centered(E_imag) ⟩
```

SR 在主 GPU 上求解：

```text
(S + λI) Δθ = -ηF
```

实现使用 `torch.linalg.solve`，不显式计算矩阵逆，也不会因 MUSA 算子不支持而静默退回 CPU。

## 蒙特卡洛样本约定

`MetropolisSampler` 返回的总样本数为：

```text
N_sample = num_chains × sweeps
```

相邻两个保存样本之间默认执行 `num_sites` 次 Metropolis 更新尝试，也可以通过 `sweep_size` 修改。因此总的更新尝试次数约为：

```text
N_update = num_chains × sweeps × sweep_size
```

当 `sweep_size=num_sites` 时，这与“每取得一个新样本，先演化一个完整晶格扫描”的习惯一致。

## 多卡计算方式

每张 GPU 保存一个模型副本，并独立完成：

1. 采样；
2. 局域能量计算；
3. Adam 梯度或 SR Jacobian/QGT 统计量计算。

随后将梯度或 SR 充分统计量聚合到主 GPU。SR 的线性方程只在主 GPU 上求解，因为一次稠密线性方程求解本身并不会自动分布到多卡；其余高开销的样本相关计算可以并行分摊到多卡。

该实现不依赖 NCCL/MCCL，适合在 CUDA 和 MUSA 上使用相同的程序流程进行基准测试。

## 性能记录

基态驱动器可以记录每一步的：

- 能量、方差和蒙特卡洛接受率；
- 采样、局域能量、求导、统计量聚合、SR `solve`、参数更新和整步耗时；
- 总运行时间、最佳能量和最终能量；
- 后端、设备名称、GPU 数量、PyTorch 版本、模型、采样器、优化器和关键参数。

逐步数据可写入 CSV，运行配置及最终摘要可写入 JSON。计时前后会同步所有选中的 GPU，减少异步执行对 CUDA/MUSA 对比结果的干扰。建议先进行若干预热步骤，再使用相同模型、样本数、精度和随机种子进行公平比较。

## 精确对角化

小系统可以使用 `ExactDiagonalizer` 得到精确基态能量和波函数：

```python
from neural_quantum_solver import ExactDiagonalizer, tilted_field_ising

system = tilted_field_ising(
    num_sites=10,
    coupling=1.0,
    field_x=1.0,
    field_z=0.0,
    periodic=True,
)

exact = ExactDiagonalizer(system).ground_state()
print(exact.energy)
```

ED 在 CPU 上使用 `complex128` 稠密矩阵，只适用于 Hilbert 空间维数较小的系统。它是精确参考路径，不属于 CUDA/MUSA 神经网络性能基准。

## 全空间能量检查

对于较小系统，可以在训练结束后枚举全部基矢，用当前 NQS 参数计算无采样误差的能量。该过程仍在所选 GPU 上使用双实数表示；只有保存检查点或最佳模型快照时才把数据复制到 CPU，以保证文件可以在 NVIDIA 与 MUSA 环境之间移植。

## 进一步说明

- [架构说明](docs/ARCHITECTURE.md)
- [MUSA 兼容性记录](docs/MUSA_COMPATIBILITY.md)
