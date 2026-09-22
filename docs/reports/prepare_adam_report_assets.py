"""从 Adam 原始 CSV 生成报告数值表、图像副本和数据指纹；不修改实验结果。"""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'benchmark_results/diagnostics/late_time_snapshot_adam_1000_seed5'
TRAJ = ROOT / 'benchmark_results/projected_ptvmc_trajectory'


def read(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def sci(value):
    mantissa, exponent = f'{float(value):.3e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def table(filename, caption, label, headers, rows):
    text = '\\begin{table}[htbp]\n\\centering\\small\n'
    text += f'\\caption{{{caption}}}\\label{{{label}}}\n'
    text += '\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{' + 'c' * len(headers) + '}\n\\toprule\n'
    text += ' & '.join(headers) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(row) + r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n'
    (HERE / 'tables' / filename).write_text(text, encoding='utf-8')


def main():
    rows = read(DATA / 'results.csv')
    a = [r for r in rows if r['family']]
    b = [r for r in rows if not r['family']]
    trajectory = read(TRAJ / 'trajectory.csv')
    assert len(a) == 231 and len(b) == 21 and len(trajectory) == 41
    (HERE / 'tables').mkdir(exist_ok=True)
    (HERE / 'figures').mkdir(exist_ok=True)
    selected = (0., .5, 1., 1.5, 2.)
    values = []
    for t in selected:
        groups = {family: [r for r in a if r['family'] == family and float(r['time']) == t]
                  for family in ('trajectory', 'same_size_restart', 'larger_restart')}
        entry = [f'{t:.1f}', sci(groups['trajectory'][0]['initial_loss']), sci(groups['trajectory'][0]['best_loss'])]
        for family in ('same_size_restart', 'larger_restart'):
            losses = [float(r['best_loss']) for r in groups[family]]
            entry += [sci(min(losses)), sci(median(losses))]
        values.append(entry)
    table('snapshot_table.tex', 'A 的代表性时间点。随机重启列分别报告5个种子的最小值和中位数；所有误差均对 ED 态计算。', 'tab:a',
          ['$t$', '轨迹原始', '轨迹再拟合', '4L 最小', '4L 中位', '8L 最小', '8L 中位'], values)
    values = [[f"{float(r['time']):.1f}", r['numerical_rank'], f"{float(r['effective_rank']):.2f}",
               f"{100*float(r['effective_rank_fraction']):.2f}\\%", sci(r['retained_condition_number']), sci(r['tangent_residual'])]
              for r in b if float(r['time']) in selected]
    table('geometry_table.tex', 'B/C 的代表性数值，均在原始4L轨迹上测量。初态残差为代码截断后的值。', 'tab:bc',
          ['$t$', '$r_{\\rm num}$', '$r_{\\rm eff}$', '$r_{\\rm eff}/P$', '$\\kappa_{\\rm ret}$', '$R_{\\rm tan}$'], values)
    inputs = [DATA / name for name in ('results.csv', 'config.json', 'summary.json')]
    inputs += [TRAJ / name for name in ('trajectory.csv', 'run_config.json')]
    for name in ('a_snapshot_infidelity.png', 'b_qgt_scalars.png', 'b_qgt_spectra.png', 'c_tangent_residual.png', 'rollout_quality.png'):
        source = DATA / 'figures' / name
        shutil.copy2(source, HERE / 'figures' / name)
        inputs.append(source)
    manifest = {'counts': dict(Counter(r['family'] or 'B/C' for r in rows)),
                'trajectory_rows': len(trajectory), 'converged': sum(r['converged'] == 'True' for r in a),
                'max_positive_time_residual_difference': max(abs(float(r['tangent_residual']) - float(r['tangent_ls_residual'])) for r in b if float(r['time']) > 0),
                'inputs': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (HERE / 'data_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'inputs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
