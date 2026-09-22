"""比较已完成的 Adam 与 overlap-SR 长时 p-tVMC A/B/C 实验。"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAM_DIAGNOSTICS = PROJECT_ROOT / "benchmark_results" / "diagnostics" / "late_time_snapshot_adam_1000_seed5"
SR_DIAGNOSTICS = PROJECT_ROOT / "benchmark_results" / "diagnostics" / "late_time_snapshot_sr_1000_seed5"
ADAM_TRAJECTORY = PROJECT_ROOT / "benchmark_results" / "projected_ptvmc_trajectory"
SR_TRAJECTORY = PROJECT_ROOT / "benchmark_results" / "projected_ptvmc_trajectory_sr"
OUTPUT = PROJECT_ROOT / "benchmark_results" / "comparisons" / "adam_vs_sr"


def configure_font() -> None:
    """选择可用中文字体。"""
    for name in ("Microsoft YaHei", "SimHei", "SimSun"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.sans-serif"] = [name]
        break
    plt.rcParams["axes.unicode_minus"] = False


def read_rows(path: Path) -> list[dict[str, str]]:
    """读取一个结果 CSV。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少比较输入：{path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def family_median(rows: list[dict[str, str]], family: str) -> tuple[list[float], list[float]]:
    """提取模型族每个时间点的跨 seed 最佳 loss 中位数。"""
    grouped: dict[float, list[float]] = {}
    for row in rows:
        if row["family"] == family:
            grouped.setdefault(float(row["time"]), []).append(float(row["best_loss"]))
    times = sorted(grouped)
    return times, [median(grouped[time]) for time in times]


def trajectory_diagnostics(rows: list[dict[str, str]]) -> tuple[list[float], list[dict[str, str]]]:
    """提取 A/B/C 结果 CSV 内不带模型族名称的 B/C 行。"""
    points = sorted((row for row in rows if not row["family"]), key=lambda row: float(row["time"]))
    return [float(row["time"]) for row in points], points


def plot_trajectory() -> None:
    """比较两种投影优化器的局部与累计 trajectory 误差。"""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, directory, color in (("Adam", ADAM_TRAJECTORY, "#1f77b4"), ("overlap-SR", SR_TRAJECTORY, "#d62728")):
        rows = read_rows(directory / "trajectory.csv")
        times = [float(row["time"]) for row in rows]
        axes[0].semilogy(times, [1.0 - float(row["ed_fidelity"]) for row in rows], "o-", color=color, label=label)
        projected = [row for row in rows if row["projection_loss"]]
        axes[1].semilogy(
            [float(row["time"]) for row in projected], [float(row["projection_loss"]) for row in projected],
            "o-", color=color, label=label,
        )
    axes[0].set_title("累计 ED infidelity")
    axes[0].set_ylabel(r"$1-F(\psi_{ED},\psi_{NQS})$")
    axes[1].set_title("单步局部投影误差")
    axes[1].set_ylabel("projection loss")
    for axis in axes:
        axis.set_xlabel("时间 t")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.savefig(OUTPUT / "trajectory_adam_vs_sr.png", dpi=180)


def plot_abc() -> None:
    """比较 A 的随机重启表现以及 B/C 的核心标量。"""
    adam = read_rows(ADAM_DIAGNOSTICS / "results.csv")
    sr = read_rows(SR_DIAGNOSTICS / "results.csv")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for family, color in (("same_size_restart", "#1f77b4"), ("larger_restart", "#d62728")):
        for label, rows, style in (("Adam", adam, "-"), ("overlap-SR", sr, "--")):
            times, values = family_median(rows, family)
            axes[0, 0].semilogy(times, values, f"o{style}", color=color, label=f"{label}，{family}")
    axes[0, 0].set_title("A：随机重启最佳 loss 的 seed 中位数")
    axes[0, 0].set_ylabel("best infidelity")

    for label, rows, color in (("Adam", adam, "#1f77b4"), ("overlap-SR", sr, "#d62728")):
        times, points = trajectory_diagnostics(rows)
        axes[0, 1].plot(times, [float(row["effective_rank_fraction"]) for row in points], "o-", color=color, label=label)
        axes[1, 0].semilogy(times, [float(row["retained_condition_number"]) for row in points], "o-", color=color, label=label)
        axes[1, 1].semilogy(times, [float(row["tangent_residual"]) for row in points], "o-", color=color, label=label)
    axes[0, 1].set_title("B：QGT 有效秩比例")
    axes[0, 1].set_ylabel(r"$r_{eff}/N_{param}$")
    axes[1, 0].set_title("B：QGT 保留条件数")
    axes[1, 0].set_ylabel("条件数")
    axes[1, 1].set_title("C：切空间残差")
    axes[1, 1].set_ylabel(r"$R_{tan}$")
    for axis in axes.flat:
        axis.set_xlabel("时间 t")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(OUTPUT / "abc_adam_vs_sr.png", dpi=180)


def main() -> None:
    """确认输入完整后写入两张 Adam/SR 对比图。"""
    configure_font()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_trajectory()
    plot_abc()
    print(f"Adam/SR 对比图已写入：{OUTPUT}")


if __name__ == "__main__":
    main()
