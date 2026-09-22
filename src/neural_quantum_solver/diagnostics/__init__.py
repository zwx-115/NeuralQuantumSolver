"""面向小型 NQS 系统的全 Hilbert 空间诊断及历史导入兼容层。"""

from .qgt import QGTSpectrum, qgt_spectrum
from ..solvers import (
    CurvatureSettings,
    CurvatureStep,
    OverlapCurvature,
    OverlapSRSettings,
    OverlapSRStep,
    optimize_overlap_curvature,
    optimize_overlap_sr,
    overlap_sr_step,
)
from .snapshots import ExactSnapshot, SnapshotFitResult, evolve_exact_snapshots, fit_snapshot, fit_snapshot_sr
from .tangent import TangentResidual, tangent_space_residual

__all__ = [
    "ExactSnapshot", "OverlapSRSettings", "OverlapSRStep", "QGTSpectrum",
    "SnapshotFitResult", "TangentResidual", "evolve_exact_snapshots", "fit_snapshot",
    "fit_snapshot_sr", "optimize_overlap_sr", "overlap_sr_step", "qgt_spectrum",
    "tangent_space_residual",
    # 历史兼容导出；新代码应从 neural_quantum_solver.solvers 导入求解器。
    "CurvatureSettings", "CurvatureStep", "OverlapCurvature", "optimize_overlap_curvature",
]
