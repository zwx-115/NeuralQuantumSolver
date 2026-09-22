# Implemented architecture and migration boundary

This branch is a clean implementation, while `nqs-gs-time-python-origin` remains
the immutable behavioral reference.

## Current dependency flow

`SpinHalfHilbert + Graph -> PauliHamiltonian -> PhysicalSystem`

`NeuralQuantumState -> exact/Metropolis sampling -> estimator -> objective -> runner`

Exact diagonalization consumes only `PhysicalSystem`. It therefore works with
every system assembled from explicit Pauli terms, independently of the NQS.

## Implemented first milestone

- Legacy-compatible spin enumeration and explicit encoding conversion.
- Generic packed connected-state generation for Pauli Hamiltonians.
- Dense `complex128` exact diagonalization and exact real-time propagation.
- Ising, Heisenberg, XXZ, and J1-J2 system constructors.
- Unified complex `log_psi` contract with RBM, FNN, and lookup-table ansatzes.
- Differentiable full-Hilbert-space ground-state energy and Adam runner.
- Exact and Metropolis samplers with distinct result metadata.
- `VariationalState` composition of a model, physical system, and sampler.
- Sampler-independent `Adam` and dense `SR` optimization strategies.
- `GroundStateDriver` orchestration with callbacks, diagnostics, and checkpointing.
- Legacy/direct/normalized/gauge-fixed/log-domain overlap entry points and
  overlap diagnostics.
- QGT construction, regularized SR solve diagnostics, frozen evolution targets,
  and structured checkpoints.

## Deliberately not hidden behind the baseline

The stabilized overlap paths are separate functions. They do not replace
`legacy_ratio_loss`. The present `log_domain_ratio_loss` uses normalized dense
matrix products as the small-system stability reference; a sparse local-gate
complex-log-sum implementation belongs to the next evolution milestone.

The full-sum projected rollout example is a small-system numerical baseline.
Monte Carlo overlap projection, a reusable time-evolution driver, minSR,
persistent-chain/autoregressive samplers, Trotter block scheduling, CUDA
performance benchmarks, and legacy checkpoint adapters remain future migration
phases. Their absence does not affect exact diagonalization or the composable
exact/Monte-Carlo ground-state paths.

## 投影求解器研究扩展（2026-09）

当前源码将“只读诊断”和“会更新波函数参数的求解器”分开：

```text
diagnostics/
  qgt.py, tangent.py       只计算谱、秩和切空间残差
  snapshots.py             快照与历史兼容入口

solvers/
  projected_sr.py          固定目标的全求和 SR 投影
  projected_curvature.py   固定目标的子空间二阶量子信赖域投影
```

`optimizers.py` 仍只服务基态 VMC 的 `GroundStateOptimizer` 接口；它接收
采样能量统计量，不能在不改变语义的前提下直接承载固定目标 infidelity 求解器。
投影求解器直接接收 `(model, system, frozen_target)`，与 `evolution.py` 构造的
`FrozenTarget` 语义相容，但不依赖具体 Hamiltonian 或 RBM。

`diagnostics/overlap_sr.py` 与 `diagnostics/overlap_curvature.py` 是临时兼容入口，
只转发到 `solvers/`，不再包含实现。旧实验脚本和已启动的 Python 进程可继续使用；
所有新代码应从 `neural_quantum_solver.solvers` 导入。

`examples/diagnostics/compare_overlap_optimizers.py` 编排同初值 Adam/SR/二阶优化对照，
负责快照、rollout、独立输出目录和复现元数据；绘图由单独脚本负责。
决策与数值契约详见 `docs/CURVATURE_OPTIMIZER.md`。目前只验证小系统正确性，
不把候选优化器作为默认基线。
