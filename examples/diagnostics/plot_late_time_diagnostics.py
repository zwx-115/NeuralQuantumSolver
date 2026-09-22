"""绘制长时 p-tVMC 的 A/B/C 全 Hilbert 空间诊断结果。

默认读取 ``benchmark_results/diagnostics/late_time_snapshot_adam_1000_seed5``，
并将 PNG 写入其 ``figures`` 子目录。可用 ``--show`` 在生成后显示图像。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "benchmark_results"
    / "diagnostics"
    / "late_time_snapshot_adam_1000_seed5"
)
DEFAULT_TRAJECTORY = PROJECT_ROOT / "benchmark_results" / "projected_ptvmc_trajectory"

COLORS = {
    "trajectory": "#202020",
    "same_size_restart": "#1f77b4",
    "larger_restart": "#d62728",
}
LABELS = {
    "trajectory": "p-tVMC checkpoint",
    "same_size_restart": "随机初始化，4L hidden",
    "larger_restart": "随机初始化，8L hidden",
}


def configure_plot_font() -> None:
    """优先使用 Windows 已安装的中文字体，保证图中的中文标签可读。"""
    for name in ("Microsoft YaHei", "SimHei", "SimSun"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.sans-serif"] = [name]
        break
    plt.rcParams["axes.unicode_minus"] = False


def parse_arguments() -> argparse.Namespace:
    """解析输入目录、轨迹目录、输出目录和显示选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="诊断结果目录")
    parser.add_argument(
        "--trajectory", type=Path, default=DEFAULT_TRAJECTORY, help="p-tVMC trajectory.csv 所在目录"
    )
    parser.add_argument("--output", type=Path, default=None, help="PNG 输出目录，默认为 <input>/figures")
    parser.add_argument("--show", action="store_true", help="保存后显示全部图像")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV；缺少文件时给出可操作的错误信息。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_or_none(value: str | None) -> float | None:
    """将空 CSV 单元格保留为 None。"""
    return None if value in (None, "") else float(value)


def grouped_fits(rows: list[dict[str, str]]) -> dict[str, dict[float, list[dict[str, str]]]]:
    """按模型族和时间组织 A 实验的逐 seed 记录。"""
    groups: dict[str, dict[float, list[dict[str, str]]]] = {}
    for row in rows:
        family = row["family"]
        if not family:
            continue
        groups.setdefault(family, {}).setdefault(float(row["time"]), []).append(row)
    return groups


def plot_snapshot_losses(rows: list[dict[str, str]], output: Path) -> None:
    """绘制 A：原始轨迹误差、重优化结果及随机 seed 离散度。"""
    groups = grouped_fits(rows)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    trajectory = groups["trajectory"]
    times = sorted(trajectory)
    actual = [float(trajectory[time][0]["initial_loss"]) for time in times]
    refitted = [float(trajectory[time][0]["best_loss"]) for time in times]
    axes[0].plot(times, actual, "o-", color=COLORS["trajectory"], label="原始 p-tVMC checkpoint")
    axes[0].plot(times, refitted, "o--", color="#777777", label="checkpoint 再优化后的最佳值")

    for family in ("same_size_restart", "larger_restart"):
        family_times = sorted(groups[family])
        minima = []
        medians = []
        for time in family_times:
            losses = [float(row["best_loss"]) for row in groups[family][time]]
            minima.append(min(losses))
            medians.append(median(losses))
        axes[0].plot(family_times, minima, "o-", color=COLORS[family], label=f"{LABELS[family]}：seed 最小值")
        axes[0].plot(family_times, medians, "--", color=COLORS[family], alpha=0.8, label=f"{LABELS[family]}：seed 中位数")

    axes[0].set_yscale("log")
    axes[0].set_xlabel("时间 t")
    axes[0].set_ylabel(r"infidelity $1-|\langle\psi_{ED}|\psi_{RBM}\rangle|^2$")
    axes[0].set_title("A：每个时间切片的最佳可达 infidelity")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    for family in ("same_size_restart", "larger_restart"):
        for time, points in groups[family].items():
            losses = [float(row["best_loss"]) for row in points]
            axes[1].scatter([time] * len(losses), losses, color=COLORS[family], alpha=0.45, s=18)
        family_times = sorted(groups[family])
        medians = [median(float(row["best_loss"]) for row in groups[family][time]) for time in family_times]
        axes[1].plot(family_times, medians, color=COLORS[family], label=f"{LABELS[family]}：中位数")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("时间 t")
    axes[1].set_ylabel("各 seed 的最佳 infidelity")
    axes[1].set_title("A：随机初始化的 seed 离散度")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.savefig(output / "a_snapshot_infidelity.png", dpi=180)


def trajectory_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """提取 B/C 的轨迹行；这些行在 CSV 中没有模型族名称。"""
    return sorted((row for row in rows if not row["family"]), key=lambda row: float(row["time"]))


def plot_qgt(rows: list[dict[str, str]], summary: dict[str, object], output: Path) -> None:
    """绘制 B：秩、有效秩比例、条件数和若干时刻的 QGT 谱。"""
    points = trajectory_rows(rows)
    times = [float(row["time"]) for row in points]
    numerical_rank = [float(row["numerical_rank"]) for row in points]
    parameter_count = [float(row["num_parameters"]) for row in points]
    effective_rank = [float(row["effective_rank"]) for row in points]
    effective_fraction = [float(row["effective_rank_fraction"]) for row in points]
    condition = [float(row["retained_condition_number"]) for row in points]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].plot(times, numerical_rank, "o-", label="数值秩")
    axes[0].plot(times, parameter_count, "--", color="#555555", label="参数数目")
    axes[0].set_xlabel("时间 t")
    axes[0].set_ylabel("维数")
    axes[0].set_title("B：QGT 数值秩")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    effective_line = axes[1].plot(times, effective_rank, "o-", label="effective rank")
    fraction_axis = axes[1].twinx()
    fraction_line = fraction_axis.plot(
        times, [100.0 * value for value in effective_fraction], "s--", color="#ff7f0e",
        label="effective rank / 参数数目",
    )
    axes[1].set_xlabel("时间 t")
    axes[1].set_ylabel("effective rank")
    fraction_axis.set_ylabel("effective rank / 参数数目 (%)")
    axes[1].set_title("B：QGT 有效秩")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(effective_line + fraction_line, [line.get_label() for line in effective_line + fraction_line])

    axes[2].semilogy(times, condition, "o-", color="#9467bd")
    axes[2].set_xlabel("时间 t")
    axes[2].set_ylabel("保留条件数")
    axes[2].set_title("B：QGT 保留条件数")
    axes[2].grid(True, which="both", alpha=0.25)
    figure.savefig(output / "b_qgt_scalars.png", dpi=180)

    spectra = summary.get("qgt_spectra", {})
    figure, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    requested_times = (0.0, 0.5, 1.0, 1.5, 2.0)
    for time in requested_times:
        key = f"t{time:g}"
        point = spectra.get(key) if isinstance(spectra, dict) else None
        if not isinstance(point, dict):
            continue
        eigenvalues = point.get("eigenvalues")
        if not isinstance(eigenvalues, list):
            continue
        axis.semilogy(range(1, len(eigenvalues) + 1), eigenvalues, ".-", ms=3, label=f"t={time:g}")
    axis.set_xlabel("按升序排列的本征值编号")
    axis.set_ylabel("QGT 本征值")
    axis.set_title("B：QGT 谱")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(output / "b_qgt_spectra.png", dpi=180)


def plot_tangent(rows: list[dict[str, str]], output: Path) -> None:
    """绘制 C：切空间残差及与显式最小二乘交叉验证的差异。"""
    points = trajectory_rows(rows)
    times = [float(row["time"]) for row in points]
    residual = [float(row["tangent_residual"]) for row in points]
    least_squares = [float(row["tangent_ls_residual"]) for row in points]
    mismatch = [abs(left - right) for left, right in zip(residual, least_squares, strict=True)]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].semilogy(times, residual, "o-", label=r"$R_{tan}$：QGT 公式")
    axes[0].semilogy(times, least_squares, "s--", label="显式 Jacobian 最小二乘")
    axes[0].set_xlabel("时间 t")
    axes[0].set_ylabel("归一化残差")
    axes[0].set_title("C：Schrödinger 方向离开切空间的比例")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(times, mismatch, "o-", color="#ff7f0e")
    axes[1].set_xlabel("时间 t")
    axes[1].set_ylabel(r"$|R_{tan}-R_{LS}|$")
    axes[1].set_title("C：两种残差计算的差异")
    axes[1].grid(True, which="both", alpha=0.25)
    figure.savefig(output / "c_tangent_residual.png", dpi=180)


def plot_rollout_quality(trajectory_directory: Path, output: Path) -> None:
    """绘制原始 p-tVMC rollout 的累计 ED 误差和单步投影误差。"""
    rows = load_csv(trajectory_directory / "trajectory.csv")
    times = [float(row["time"]) for row in rows]
    ed_infidelity = [1.0 - float(row["ed_fidelity"]) for row in rows]
    projection_rows = [row for row in rows if row["projection_loss"]]
    projection_times = [float(row["time"]) for row in projection_rows]
    projection_loss = [float(row["projection_loss"]) for row in projection_rows]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].semilogy(times, ed_infidelity, "o-", color=COLORS["trajectory"])
    axes[0].set_xlabel("时间 t")
    axes[0].set_ylabel(r"$1-F(\psi_{ED}(t),\psi_{p\mathrm{-}tVMC}(t))$")
    axes[0].set_title("原始 p-tVMC 的累计 ED infidelity")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].semilogy(projection_times, projection_loss, "o-", color="#2ca02c")
    axes[1].set_xlabel("时间 t")
    axes[1].set_ylabel("单步 projection loss")
    axes[1].set_title("每个 dt=0.05 步的局部投影误差")
    axes[1].grid(True, which="both", alpha=0.25)
    figure.savefig(output / "rollout_quality.png", dpi=180)


def main() -> None:
    """读取结果、生成四张 PNG，并输出图像目录。"""
    arguments = parse_arguments()
    configure_plot_font()
    output = arguments.output or arguments.input / "figures"
    output.mkdir(parents=True, exist_ok=True)
    rows = load_csv(arguments.input / "results.csv")
    with (arguments.input / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    plot_snapshot_losses(rows, output)
    plot_qgt(rows, summary, output)
    plot_tangent(rows, output)
    plot_rollout_quality(arguments.trajectory, output)
    print(f"已写入图像：{output}")
    if arguments.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
