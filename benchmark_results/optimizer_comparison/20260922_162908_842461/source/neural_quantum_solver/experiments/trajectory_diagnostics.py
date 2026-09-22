"""B/C：对冻结轨迹模型运行全求和 QGT 与切空间诊断。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import torch

from ..diagnostics import qgt_spectrum, tangent_space_residual
from ..models import NeuralQuantumState
from ..systems import PhysicalSystem


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    model: NeuralQuantumState


def diagnose_trajectory(
    system: PhysicalSystem,
    points: list[TrajectoryPoint],
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> tuple[list[dict[str, float | int | bool]], dict[str, dict[str, torch.Tensor]]]:
    """返回可写入 CSV 的标量行和可写入 JSON 的谱数组。

    每个模型都会在评估前复制，因此诊断不会修改动力学 rollout 持有的 NQS
    实例。
    """
    rows: list[dict[str, float | int | bool]] = []
    spectra: dict[str, dict[str, torch.Tensor]] = {}
    for point in points:
        model = deepcopy(point.model).eval()
        qgt = qgt_spectrum(model, system, rtol=rtol, atol=atol)
        tangent = tangent_space_residual(model, system, rtol=rtol, atol=atol)
        key = f"t{point.time:g}"
        rows.append({
            "time": point.time,
            "num_parameters": qgt.num_parameters,
            "numerical_rank": qgt.numerical_rank,
            "effective_rank": float(qgt.effective_rank.item()),
            "effective_rank_fraction": float(qgt.effective_rank_fraction.item()),
            "retained_condition_number": float(qgt.retained_condition_number.item()),
            "qgt_threshold": float(qgt.threshold.item()),
            "tangent_residual": float(tangent.normalized_residual.item()),
            "tangent_absolute_residual": float(tangent.absolute_residual.item()),
            "hamiltonian_variance": float(tangent.variance.item()),
            "tangent_ls_residual": float(tangent.least_squares_residual.item()),
            "tangent_defined": tangent.defined,
        })
        spectra[key] = {
            "eigenvalues": qgt.eigenvalues,
            "singular_values": qgt.singular_values,
        }
    return rows, spectra
