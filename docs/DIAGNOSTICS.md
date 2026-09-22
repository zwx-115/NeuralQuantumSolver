# 长时 p-tVMC 诊断

诊断包与 p-tVMC 执行器、重叠估计器和 SR 更新路径解耦。它只针对小系统的全 Hilbert
空间求和，并优先使用 `complex128`。

`diagnostics.snapshots.evolve_exact_snapshots` 在指定时刻生成冻结的 ED 精确态。
`experiments.SnapshotFittingExperiment` 将各模型族分别拟合到每个精确态，并通过
`save_diagnostic_artifacts` 写入 `config.json`、`results.csv`、`summary.json` 和
`checkpoints.pt`。`trajectory` 模型族恢复原始 rollout 中对应时刻的 p-tVMC checkpoint；
同尺寸和更大 RBM 均从新的随机初始化开始。起始配置模板是
[`configs/diagnostics/late_time_snapshot_fitting.json`](../configs/diagnostics/late_time_snapshot_fitting.json)。
可运行的 L=12 全求和实验位于 `examples/diagnostics/exact_snapshot_diagnostics.py`。

`diagnostics.qgt.qgt_spectrum` 计算

\[
S_{ij}=\langle O_i^*O_j\rangle-\langle O_i^*\rangle\langle O_j\rangle.
\]

它记录原始谱、以 `max(atol, rtol * lambda_max)` 为阈值的数值秩、参与率有效秩
`Tr(S)^2 / Tr(S^2)`、保留条件数和有效秩比例。它不会施加 SR 正则化，也不会更新参数。

`diagnostics.tangent.tangent_space_residual` 计算

\[
R_{tan}=1-\frac{F^\dagger S^+F}{\operatorname{Var}(H)}.
\]

同时它会显式求解 `-i(H-<H>)|psi>` 在复 Hilbert 空间切向量上的最小二乘投影，独立报告
归一化残差。零方差本征态的归一化残差没有定义（`NaN`），绝对残差为零。

`experiments.diagnose_trajectory` 接收 rollout 中冻结的 `(time, model)` 点，返回写入 CSV
的标量行，以及写入 JSON 的原始 QGT 本征值和奇异值。函数会深拷贝模型，因此不会修改正在
运行的动力学轨迹。

## SR 对比轮次

`examples/time_evolution/projected_time_evolution_sr.py` 保持与 Adam 基线相同的基态
制备、哈密顿量、`dt` 和 checkpoint 时间点，仅将每一步的 overlap 投影改为全求和 SR。
它写入 `benchmark_results/projected_ptvmc_trajectory_sr`，其中
`projection_sr_history.csv` 记录每一次 SR 更新的 loss、QGT 数值秩、条件数、更新范数和
Fubini--Study 步长。

随后运行 `examples/diagnostics/exact_snapshot_diagnostics_sr.py`。该脚本将 A 的冻结 ED
快照拟合也改为 SR，并在 SR trajectory checkpoint 上运行 B/C；结果写入
`benchmark_results/diagnostics/late_time_snapshot_sr_1000_seed5`。原有 Adam 目录不会被
读取或写入。`examples/diagnostics/plot_adam_vs_sr.py` 在两轮完成后生成并排比较图。
