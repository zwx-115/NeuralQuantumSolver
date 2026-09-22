"""将冻结目标态投影回 NQS 流形的求解器。

求解器会更新模型参数；它们和只读的 QGT、切空间诊断分开保存。
"""

from .projected_curvature import (
    CurvatureSettings,
    CurvatureStep,
    FunctionalWavefunction,
    OverlapCurvature,
    optimize_overlap_curvature,
    solve_trust_region,
)
from .projected_sr import (
    OverlapSRSettings,
    OverlapSRStep,
    optimize_overlap_sr,
    overlap_infidelity,
    overlap_sr_step,
)

__all__ = [
    "CurvatureSettings",
    "CurvatureStep",
    "FunctionalWavefunction",
    "OverlapCurvature",
    "OverlapSRSettings",
    "OverlapSRStep",
    "optimize_overlap_curvature",
    "optimize_overlap_sr",
    "overlap_infidelity",
    "overlap_sr_step",
    "solve_trust_region",
]
