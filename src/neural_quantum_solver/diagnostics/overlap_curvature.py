"""兼容入口：二阶投影求解器已迁移至 :mod:`neural_quantum_solver.solvers`。"""

from ..solvers.projected_curvature import (
    CurvatureSettings,
    CurvatureStep,
    FunctionalWavefunction,
    OverlapCurvature,
    optimize_overlap_curvature,
    solve_trust_region,
)

__all__ = [
    "CurvatureSettings",
    "CurvatureStep",
    "FunctionalWavefunction",
    "OverlapCurvature",
    "optimize_overlap_curvature",
    "solve_trust_region",
]
