# MUSA 兼容性记录

本文记录基态性能测试中发现的后端限制。兼容处理必须保持 CUDA/MUSA 的 GPU 计算流程，不得在算子失败时静默地改用 CPU 重试。

## 双实数分支的处理方案

`real-amplitude-phase-nqs` 训练路径不再依赖 GPU 复数张量。神经网络参数与输出、哈密顿量连接值、局域能量、Jacobian、QGT、力和 SR 求解全部表示为 `float32` 或 `float64` 实数张量。

下面记录的限制来自保留的旧复数路径，也是采用当前双实数架构的直接原因。

## 2026-09-01：`torch.linalg.pinv` 失败

- 环境信息：muDNN v3300。
- 输入：位于 `musa` 上、经过正则化的 `complex64` SR 矩阵。
- 报错：`NOT_SUPPORTED`，内部进入了一元 `MAX`/`DOUBLE` 路径。
- 对照测试：`torch.linalg.inv`、`torch.linalg.solve`、`torch.linalg.eigvalsh` 和 `torch.linalg.svdvals` 均可成功执行。
- 处理方案：SR 直接在主 CUDA 或 MUSA GPU 上使用 `torch.linalg.solve` 求解正则化线性方程，不再使用伪逆，也不回退到 CPU。

服务器原始错误信息：

```text
muDNN(v3300) ... ERROR# NOT_SUPPORTED in Binary::Run, Reason:
Unsupported unary mode: MAX, or data type: DOUBLE
RuntimeError: BinaryCall MUDNN failed in: Run MaximumTensor
```

## 2026-09-01：复数矩阵—向量运算（`mv`）失败

- 环境信息：muDNN v3300。
- 失败表达式：`ComplexRBM` 中使用 `complex64` 操作数执行 `x @ visible_bias`。
- 报错：`SetMUTensorDType Unsupported tensor dtype: ComplexFloat`。
- 观察结果：在此之前执行的复数矩阵—矩阵表达式 `x @ weight.mT` 可以成功运行。
- 旧复数路径的处理方案：把向量表示成单列矩阵，在 CUDA 和 MUSA 上使用数学等价的 `mm` 路径；QGT 诊断乘法也采用同样处理。该方案不回退到 CPU。
- 状态：CUDA 回归测试以及 MC+SR 冒烟测试已通过；MUSA 仍需在目标服务器上复测。

## 2026-09-01：复数 `cosh`/`log` 路径失败

- 失败表达式：`torch.log(2 * torch.cosh(theta)).sum(dim=-1)`，其中 `theta` 为 `complex64` MUSA 张量。
- 报错：`SetMUTensorDType Unsupported tensor dtype: ComplexFloat`。
- 影响：即使绕过复数矩阵—向量运算，复数 RBM 的激活函数路径仍无法在当前 MUSA 环境运行。
- 处理方案：当前分支改用两个完全实数的网络分别表示对数振幅与相位；采样、局域能量、梯度和 SR 全程使用实数 GPU 算子。
- 状态：CPU 与 CUDA 测试已通过；MUSA 需要在安装 torch_musa 的目标服务器上完成最终复测。
