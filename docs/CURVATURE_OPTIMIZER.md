# 子空间二阶量子信赖域实验

这是实验优化器，不保证比 Adam/SR 更快或达到更低 loss，也不宣称算法创新性。
原始 Adam/SR 脚本、结果目录、DynamicsRunner 和基础结果对象均不修改。
核心实现：`src/neural_quantum_solver/solvers/projected_curvature.py`。
它与 `src/neural_quantum_solver/solvers/projected_sr.py` 同属固定目标投影求解器层；
旧的 `diagnostics/overlap_curvature.py` 仅保留导入兼容。

## 数学与实现

复参数逐张量按实虚交错展开为实坐标 x。通过 functional_call 构造
归一化态 ψ(x)，只要求 NQS 的 forward 调用 log_psi。
目标 φ 在一次投影中 detach、归一化并冻结。

`r=(I-|φ><φ|)ψ`，`L=||r||²=1-|<φ|ψ>|²`。
残差形式减少接近 1 的 fidelity 相减造成的浮点消减；不是更换 loss。

实量子度量为 `G_ij=Re<∂iψ|(I-|ψ><ψ|)|∂jψ>`。
JVP/VJP 计算 Gv，不构造完整 Jacobian 或 QGT。CG 近似解
`(G+regularization I)p=-∇L`，日志记录迭代次数和相对残差。
CG 步数有限，因此这个搜索方向不等于原稠密复数 SR 的精确更新。

子空间 V 的候选方向：CG 自然方向、负梯度、Hessian 乘自然方向、
上一接受步；经两次实正交化去除近共线项，最多四维。
不存在搜索到全空间最负曲率方向的保证。

完整曲率为 `H_L=2 Re(J_r†J_r)+2 Re Σ r* ∂²r`。
只用 Hessian–向量乘积计算 `VᵀH_LV`。
`--methods gauss_newton` 使用相同方向构造，但将小矩阵替换为
`2 Re[(J_rV)†(J_rV)]`，用于移除残差二阶项的消融；它不是旧 SR。
该消融仍计算用于子空间方向的一个 Hessian–向量乘积。

小型二次模型 `m(z)=L+gᵀVz+zᵀVᵀH_LVz/2` 在约束
`zᵀ[VᵀGV+(radius/parameter_radius)² I]z <= (0.9 radius)²`
下通过 Cholesky 白化、谱分解和 secular equation 求解，包含负曲率 hard case。
该合并约束同时限制局部量子距离和欧氏参数更新；0.9 留出非线性余量。

接受条件：实际原始 loss 严格下降、rho=实际下降/预测下降 >= 0.1，且
实际态间 infidelity <= radius²。拒绝缩小半径 0.5 倍；rho>0.75 且
接近子问题边界时扩大 2 倍。不接受的试探不写入模型。
日志记录非有限试探数，不用 clamp 掩盖 loss。
信赖域的局部平方距离与实际 infidelity 在小步极限一致，但不是有限步恒等式。

## 默认参数

| 参数 | 值 |
|---|---:|
| 初始 / 最小 / 最大量子半径 | 0.1 / 1e-7 / 0.5 |
| 欧氏参数半径 | 0.5 |
| CG 度量正则化 | 0.01 |
| CG 最大迭代 / 相对容差 | 12 / 1e-6 |
| 每轮最多试探 | 6 |
| 接受比率 / 内边界比例 | 0.1 / 0.9 |
| 拟合步数 / loss 停止阈值 | 100 / 1e-10 |
| dtype | complex128（实坐标 float64） |

原始 SR 对照保持 lr=0.05、regularization=0.01、rcond=1e-12、
chunk_size=512、max_backtracks=8。Adam 为 lr=0.01，大模型 lr=0.003。

## 先做两个晚时刻的同初值对照

在已经安装 PyTorch 的环境、仓库根目录执行：

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$PWD\src"
python .\examples\diagnostics\compare_overlap_optimizers.py --times 1.5 2.0 --steps 100
```

默认从已有 SR 目录 `projected_ptvmc_trajectory_sr` 读取这两个时间的 checkpoint，
每个 checkpoint 复制三份，分别交给 Adam、SR、curvature，目标均为同一 ED 快照。
系统 L=12，初态 J=0,hx=0.5,hz=0，演化 J=1,hx=0.5,hz=0.5，OBC。
这里的时间点是快照时间，不是积分 dt。
这是受控的新比较，不把之前不同初值/步数的数据冒充同初值对照。

增加消融或仅运行新方法：

```powershell
python .\examples\diagnostics\compare_overlap_optimizers.py --methods curvature gauss_newton --times 1.5 2.0 --steps 100
```

完整 A 的模型家族和 seeds 可以显式开启，但建议先完成上述小规模对照：

```powershell
python .\examples\diagnostics\compare_overlap_optimizers.py --families trajectory same_size_restart larger_restart --seeds 1 2 3 4 5 --times 0 0.5 1 1.3 1.5 1.6 2 --steps 100
```

每十步打印进度；每步 CSV 立即 flush，每次拟合结束保存最佳模型、初值、最终模型和目标。
checkpoint 内的 optimizer_state 对应 final_state_dict，不一定对应最佳模型。
核心 OverlapCurvature 支持同目标下的状态恢复；脚本没有自动续跑功能。
中途停止时已完成的拟合和当前已刷新的 CSV 保留，未完成拟合没有中间 checkpoint。

## 独立 p-tVMC 轨迹

```powershell
python .\examples\diagnostics\compare_overlap_optimizers.py --mode rollout --methods curvature --steps 40 --dt 0.05 --final-time 2.0
```

复用源 SR 目录的 t=0 RBM（源基态制备可能包含 MC，本脚本不重新做基态训练）。
本脚本的全部投影和 ED 参考均为 full summation，无 MC。
同一次运行也可使用 `--methods adam sr curvature`，三种方法从完全相同 t=0 参数出发。
每个时间步新建内层优化器，使用本步最佳参数继续演化，故与原 Adam 脚本
使用最后参数的策略有显式差异；不要把这个差异归因于优化器。
物理演化使用 ED 谱分解施加 exp(-i H dt)，不加入新的积分器。
`--diagnose` 可在 rollout 后调用已有 B/C 诊断；这会增加计算量和模型副本内存。
每个时间步保存 `ptvmc_stepXXXX.pt`，避免两位小数命名造成不同 dt 的覆盖。

## 输出与绘图

默认新建 `benchmark_results/optimizer_comparison/日期_时间_微秒/`。
`--output 路径` 可以指定名称，但该路径必须尚不存在，防止覆盖任何旧结果。

- run_config.json：完整设置、设备/版本、git 状态、输入 checkpoint SHA256。
- source/：本次实际使用的 Python 包和实验脚本副本。
- summary.csv：最佳/最终 loss、迭代、停止原因、同步计时。
- histories/：逐步 CSV 和每次拟合 checkpoint。
- rollout 时各方法子目录：trajectory.csv、每步模型；可选 diagnostics/。

```powershell
python .\examples\diagnostics\plot_optimizer_comparison.py .\benchmark_results\optimizer_comparison\你的运行目录
```

绘图需要 matplotlib；核心计算只依赖 PyTorch 和 Python 标准库。
图比较最佳 loss 对时间点、最佳 loss 对实际耗时和 rollout 的 ED infidelity。
绘图的 log 显示下限为 1e-16，不修改 CSV 中的数值。

## 验收与边界

测试命令：`python -m pytest tests/test_overlap_curvature.py -q`。
检查复参数梯度/二阶方向导数有限差分、量子度量小步极限、
负曲率 hard case、RBM 的非线性接受行为、目标冻结、单调接受及同目标状态恢复。
小系统测试只能验证实现，不能证明 L=12 的科学结论或性能优势。
当前实现使用全 Hilbert 态，适用于小系统，不是大系统 MC 替代方案。
停在 min_radius 或 stationary_direction 时，converged 仍由 loss 阈值判断，
不能把没有可接受更新解释为表达能力极限。
