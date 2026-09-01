# 双实数振幅—相位基态架构

## 表示层边界

`AmplitudePhaseNQS.log_psi_parts` 返回两个实数张量：

```text
LogPsiParts(log_amplitude, phase)
```

GPU 计算路径不需要把 `psi` 或 `log(psi)` 构造成复数张量。原有复数模型被隔离在旧接口之后，仅用于兼容已有代码和回归测试。

## 依赖关系

```text
SpinHalfHilbert + Graph
        -> PauliHamiltonian.real_connections
        -> PhysicalSystem

AmplitudePhaseNQS
        -> Exact/Metropolis 采样器（只使用 log_amplitude）
        -> 双实数局域能量（实部数组 + 虚部数组）
        -> Adam 或实数 QGT/SR
        -> GroundStateDriver
```

稠密精确对角化采用独立的 CPU `complex128` 路径，但与神经网络求解器使用同一个 `PhysicalSystem`。

## 优化器

Adam 对以下实数代理目标求导：

```text
2 * < (E_real-Ebar_real) * log_amplitude
    + (E_imag-Ebar_imag) * phase >.
```

SR 分别构造对数振幅与相位的逐样本 Jacobian。对于实数参数，其度量矩阵和力都是实数：

```text
S = Cov(J_amplitude) + Cov(J_phase)
F = < centered(J_amplitude) * centered(E_real)
    + centered(J_phase) * centered(E_imag) >.
```

主 GPU 使用 `torch.linalg.solve` 求解：

```text
(S + lambda I) delta = -eta F
```

## 多 GPU

`DeviceMesh` 创建一组后端相同且编号连续的 `cuda` 或 `musa` 设备。每张设备保存一个模型副本，并独立执行采样、局域能量计算和求导。Adam 梯度或实数 SR 的充分统计量随后归并到主模型副本。

当前实现不依赖 NCCL/MCCL，也不会把不受支持的 GPU 计算自动转移到 CPU。SR 的稠密线性方程只在主 GPU 上求解；与样本数量相关的 Jacobian、QGT 和力统计可以分布到多卡计算。

## 计时与可移植性

所有 GPU 阶段的计时都会同步每一张已选择的设备。每完成一个优化步骤，就把数据追加到 CSV；JSON 用于保存环境、配置和最终摘要。

检查点与最佳模型快照使用 CPU 存储，使文件能够在 NVIDIA 和 MUSA 环境之间移植。该存储传输发生在被测步骤计算之外，不计入 GPU 核心计算耗时。
