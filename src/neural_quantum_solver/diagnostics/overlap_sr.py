"""兼容入口：投影 SR 求解器已迁移至 :mod:`neural_quantum_solver.solvers`。

保留该模块，以便旧的实验脚本、已启动 Python 进程和外部用户在迁移期继续
导入。新代码应从 ``neural_quantum_solver.solvers`` 导入。
"""

from ..solvers.projected_sr import (
    OverlapSRSettings,
    OverlapSRStep,
    optimize_overlap_sr,
    overlap_infidelity,
    overlap_sr_step,
)

__all__ = [
    "OverlapSRSettings",
    "OverlapSRStep",
    "optimize_overlap_sr",
    "overlap_infidelity",
    "overlap_sr_step",
]
