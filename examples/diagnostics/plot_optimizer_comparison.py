"""绘制同初值优化对照的最终误差和按实际耗时的收敛曲线。"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    folder = args.directory
    output = folder/'figures'
    output.mkdir(exist_ok=True)
    with (folder/'summary.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    groups = sorted({(r['family'], r['seed']) for r in rows})
    for family, seed in groups:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        selected = [r for r in rows if (r['family'], r['seed']) == (family, seed)]
        for method in sorted({r['method'] for r in selected}):
            points = sorted([r for r in selected if r['method']==method], key=lambda r: float(r['time']))
            # 显示下限只用于 log 坐标，原 CSV 数据保持原样。
            ax.semilogy([float(r['time']) for r in points],
                        [max(float(r['best_loss']), 1e-16) for r in points], 'o-', label=method)
        ax.set(xlabel='Time', ylabel='Best infidelity (display floor: 1e-16)',
               title=f'{family}, seed={seed}')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/f'{family}_seed{seed}_best.png', dpi=180)
        plt.close(fig)
    # 按相同目标与初值分组比较实际耗时，不能把一次迭代当成同等成本。
    histories = {}
    for path in (folder/'histories').glob('*.csv'):
        for method in ['gauss_newton', 'curvature', 'adam', 'sr']:
            suffix = '_'+method
            if path.stem.endswith(suffix):
                histories.setdefault(path.stem[:-len(suffix)], []).append((method, path))
                break
    for label, methods in histories.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, path in methods:
            with path.open(encoding='utf-8', newline='') as f:
                history = list(csv.DictReader(f))
            ax.semilogy([float(r['elapsed_seconds']) for r in history],
                        [max(float(r['best_loss']), 1e-16) for r in history], label=method)
        ax.set(xlabel='Elapsed seconds', ylabel='Best infidelity (display floor: 1e-16)', title=label)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/f'{label}_convergence.png', dpi=180)
        plt.close(fig)
    trajectories = list(folder.glob('*/trajectory.csv'))
    if trajectories:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for path in trajectories:
            with path.open(encoding='utf-8', newline='') as f:
                data = list(csv.DictReader(f))
            ax.semilogy([float(r['time']) for r in data],
                        [max(1-float(r['ed_fidelity']), 1e-16) for r in data], label=path.parent.name)
        ax.set(xlabel='Time', ylabel='Global ED infidelity')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/'rollout_ed_infidelity.png', dpi=180)
        plt.close(fig)
    print(f'Figures: {output}')


if __name__ == '__main__':
    main()
