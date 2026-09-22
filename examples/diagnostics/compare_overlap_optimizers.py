"""同初值比较 Adam、SR 和二阶量子信赖域，支持冻结快照及独立 rollout。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch

from neural_quantum_solver import ComplexRBM, ExactDiagonalizer, tilted_field_ising
from neural_quantum_solver.solvers import (
    CurvatureSettings, OverlapCurvature, OverlapSRSettings, overlap_sr_step,
)
from neural_quantum_solver.estimators import exact_state


ROOT = Path(__file__).resolve().parents[2]


def arguments() -> argparse.Namespace:
    """所有重要实验参数均可通过命令行保存与复现。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['snapshot', 'rollout'], default='snapshot')
    p.add_argument('--methods', nargs='+', choices=['adam', 'sr', 'curvature', 'gauss_newton'],
                   default=['adam', 'sr', 'curvature'])
    p.add_argument('--times', nargs='+', type=float, default=[1.5, 2.0])
    p.add_argument('--families', nargs='+', choices=['trajectory', 'same_size_restart', 'larger_restart'],
                   default=['trajectory'])
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--num-sites', type=int, default=12)
    p.add_argument('--hidden-multiplier', type=int, default=4)
    p.add_argument('--larger-hidden-multiplier', type=int, default=8)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--tolerance', type=float, default=1e-10)
    p.add_argument('--dt', type=float, default=.05)
    p.add_argument('--final-time', type=float, default=2.)
    p.add_argument('--trajectory-dir', type=Path,
                   default=ROOT/'benchmark_results'/'projected_ptvmc_trajectory_sr')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--adam-lr', type=float, default=.01)
    p.add_argument('--larger-adam-lr', type=float, default=.003)
    p.add_argument('--sr-lr', type=float, default=.05)
    p.add_argument('--regularization', type=float, default=.01)
    p.add_argument('--jacobian-chunk-size', type=int, default=512)
    p.add_argument('--radius', type=float, default=.1)
    p.add_argument('--parameter-radius', type=float, default=.5)
    p.add_argument('--cg-steps', type=int, default=12)
    p.add_argument('--diagnose', action='store_true', help='rollout 后追加原有 B/C 诊断，可能较慢')
    return p.parse_args()


def sync(device: str) -> None:
    """CUDA 计时边界显式同步。"""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def state_dict(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model(args, t):
    """只读取源 checkpoint，不改写之前实验的任何文件。"""
    path = args.trajectory_dir/f'ptvmc_t{t:.2f}.pt'
    data = torch.load(path, map_location=args.device, weights_only=True)
    cfg = data['model_config']
    if cfg['num_visible'] != args.num_sites or abs(float(data['time'])-t) > 1e-9:
        raise ValueError(f'{path} 的系统尺寸或时间与本次配置不符')
    model = ComplexRBM(cfg['num_visible'], cfg['num_hidden'], dtype=torch.complex128,
                       device=args.device, seed=args.seeds[0])
    model.load_state_dict(data['model_state_dict'])
    return model


def fit(model, system, target, method, args, sr_settings, curvature_settings, path, label,
        adam_lr):
    """每次拟合新建优化器；保存完整步历史、最佳模型和冻结目标。"""
    model = deepcopy(model)
    initial_state = state_dict(model)
    target = target.detach()/target.detach().norm()
    def loss():
        psi = exact_state(model, system).amplitudes
        return 1-abs(torch.vdot(target, psi))**2
    optimizer = None
    if method == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    elif method in ('curvature', 'gauss_newton'):
        optimizer = OverlapCurvature(model, system, target,
            replace(curvature_settings, curvature='full' if method == 'curvature' else 'gauss_newton'))
    # 不更新参数的预热，避免初始化开销污染之后的 GPU 时间。
    with torch.no_grad():
        initial = float(loss())
    best, final, best_state = initial, initial, state_dict(model)
    best_iteration, iterations = 0, 0
    sync(args.device)
    start = time.perf_counter()
    with path.with_name(path.name+'.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['iteration', 'elapsed_seconds', 'loss_before',
            'loss_after', 'best_loss', 'accepted', 'details_json'])
        writer.writeheader()
        writer.writerow(dict(iteration=0, elapsed_seconds=0., loss_before=initial,
            loss_after=initial, best_loss=initial, accepted=False, details_json='{}'))
        handle.flush()
        print(f'开始：{label}，method={method}，初始 loss={initial:.3e}', flush=True)
        stop_reason = 'max_steps'
        for iteration in range(1, args.steps+1):
            if best <= args.tolerance:
                stop_reason = 'loss_tolerance'
                break
            before = final
            details = {}
            if method == 'adam':
                optimizer.zero_grad(set_to_none=True)
                objective = loss()
                objective.backward()
                optimizer.step()
                with torch.no_grad():
                    final = float(loss())
                accepted = True
            elif method == 'sr':
                record = overlap_sr_step(model, system, target, settings=sr_settings)
                final, accepted = record.loss_after, record.accepted
                details = record.row(iteration)
            else:
                record = optimizer.step()
                final, accepted = record.loss_after, record.accepted
                details = record.row(iteration)
            if not torch.isfinite(torch.tensor(final)):
                raise FloatingPointError(f'{label} {method} loss 非有限')
            if final < best:
                best, best_iteration, best_state = final, iteration, state_dict(model)
            iterations = iteration
            sync(args.device)
            elapsed = time.perf_counter()-start
            writer.writerow(dict(iteration=iteration, elapsed_seconds=elapsed, loss_before=before,
                loss_after=final, best_loss=best, accepted=accepted,
                details_json=json.dumps(details, ensure_ascii=False, allow_nan=False)))
            handle.flush()
            if iteration % 10 == 0 or iteration == args.steps or best <= args.tolerance:
                print(f'进度：{label}，{method}，step={iteration}/{args.steps}，'
                      f'loss={final:.3e}，best={best:.3e}，耗时={elapsed:.1f}s', flush=True)
            if method in ('curvature', 'gauss_newton') and (
                record.status == 'stationary_direction' or
                (not record.accepted and optimizer.radius <= curvature_settings.min_radius)):
                stop_reason = record.status if record.status == 'stationary_direction' else 'min_radius'
                break
        sync(args.device)
        elapsed = time.perf_counter()-start
    torch.save(dict(model_state_dict=best_state, initial_state_dict=initial_state,
        final_state_dict=state_dict(model), target=target.cpu(), best_iteration=best_iteration,
        model_config=dict(class_name='ComplexRBM', num_visible=args.num_sites,
                          num_hidden=model.num_hidden, dtype='torch.complex128'),
        optimizer_state=optimizer.state_dict() if optimizer is not None else None,
        optimizer_state_matches='final_state_dict', method=method), path.with_name(path.name+'.pt'))
    model.load_state_dict(best_state)
    if best <= args.tolerance:
        stop_reason = 'loss_tolerance'
    row = dict(method=method, initial_loss=initial, best_loss=best, final_loss=final,
               iterations=iterations, best_iteration=best_iteration, elapsed_seconds=elapsed,
               converged=best <= args.tolerance, stop_reason=stop_reason)
    print(f'完成：{label}，{method}，best={best:.3e}，{stop_reason}', flush=True)
    return model, row


def main() -> None:
    args = arguments()
    if (args.steps < 1 or args.num_sites < 1 or args.tolerance < 0 or args.dt <= 0
            or args.final_time <= 0 or any(t < 0 for t in args.times)):
        raise ValueError('尺寸、步数、时间或容差无效')
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('methods 不能重复')
    if args.mode == 'rollout' and abs(args.final_time/args.dt-round(args.final_time/args.dt)) > 1e-9:
        raise ValueError('final_time 必须是 dt 的整数倍')
    torch.manual_seed(args.seeds[0])
    torch.use_deterministic_algorithms(True, warn_only=True)
    sr_settings = OverlapSRSettings(learning_rate=args.sr_lr, regularization=args.regularization,
                                    jacobian_chunk_size=args.jacobian_chunk_size)
    settings = CurvatureSettings(radius=args.radius, parameter_radius=args.parameter_radius,
                                  regularization=args.regularization, cg_steps=args.cg_steps)
    # 在开始计算前检查源文件；随机重启不依赖轨迹 checkpoint。
    needed = [0.] if args.mode == 'rollout' else (args.times if 'trajectory' in args.families else [])
    source_hashes = {}
    for t in needed:
        p = args.trajectory_dir/f'ptvmc_t{t:.2f}.pt'
        source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        load_model(args, t)
    output = args.output or ROOT/'benchmark_results'/'optimizer_comparison'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output/'histories').mkdir()
    def git(*command):
        return subprocess.run(['git', *command], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    config = dict(output_directory=str(output.resolve()),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        sr=sr_settings.config(), curvature=settings.config(), dtype='torch.complex128',
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(args.device) if torch.device(args.device).type=='cuda' else 'cpu',
        git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
        source_checkpoint_sha256=source_hashes,
        ground_system=dict(coupling=0., field_x=.5, field_z=0., periodic=False),
        evolution_system=dict(coupling=1., field_x=.5, field_z=.5, periodic=False),
        exact_summation=True, rollout_selection='best iterate for every method')
    (output/'run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    # 保存本次 Python 源码，避免未提交的新模块无法由 git commit 单独复现。
    shutil.copytree(ROOT/'src'/'neural_quantum_solver', output/'source'/'neural_quantum_solver',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, output/'source'/Path(__file__).name)
    print(f'L={args.num_sites}，mode={args.mode}，steps={args.steps}，methods={args.methods}', flush=True)
    print(f'新方法配置={settings.config()}；输出={output}', flush=True)
    ground = tilted_field_ising(args.num_sites, coupling=0., field_x=.5, field_z=0.)
    system = tilted_field_ising(args.num_sites, coupling=1., field_x=.5, field_z=.5)
    ed = ExactDiagonalizer(dtype=torch.complex128, device=args.device)
    values, vectors = ed.diagonalize(system)
    initial_exact = ed.ground_state(ground).state
    coefficients = vectors.mH @ initial_exact
    def exact_at(t):
        return vectors @ (torch.exp(-1j*t*values)*coefficients)
    summary_fields = ['time', 'family', 'seed', 'method', 'initial_loss', 'best_loss', 'final_loss',
        'iterations', 'best_iteration', 'elapsed_seconds', 'converged', 'stop_reason']
    with (output/'summary.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        if args.mode == 'snapshot':
            for t in args.times:
                target = exact_at(t).detach()
                for family in args.families:
                    for seed in (args.seeds[:1] if family == 'trajectory' else args.seeds):
                        multiplier = args.larger_hidden_multiplier if family == 'larger_restart' else args.hidden_multiplier
                        model = load_model(args, t) if family == 'trajectory' else ComplexRBM(
                            args.num_sites, args.num_sites*multiplier, seed=seed,
                            device=args.device, dtype=torch.complex128)
                        label = f'{family}_t{t:.6f}_seed{seed}'
                        for method in args.methods:
                            _, row = fit(model, system, target, method, args, sr_settings, settings,
                                output/'histories'/f'{label}_{method}', label,
                                args.larger_adam_lr if family == 'larger_restart' else args.adam_lr)
                            writer.writerow(dict(time=t, family=family, seed=seed) | row)
                            handle.flush()
        else:
            base = load_model(args, 0.)
            matrix = ed.matrix(system)
            mz = system.hilbert.all_states(device=args.device).to(torch.float64).mean(dim=1)
            for method in args.methods:
                model = deepcopy(base)
                folder = output/method
                folder.mkdir()
                points = []
                with (folder/'trajectory.csv').open('x', newline='', encoding='utf-8') as trajectory:
                    tw = csv.DictWriter(trajectory, fieldnames=['step', 'time', 'projection_loss',
                        'step_fidelity', 'ed_fidelity', 'energy', 'magnetization_z'])
                    tw.writeheader()
                    for step in range(round(args.final_time/args.dt)+1):
                        t = step*args.dt
                        projection_loss = ''
                        if step:
                            psi = exact_state(model, system).amplitudes.detach()
                            target = vectors @ (torch.exp(-1j*args.dt*values)*(vectors.mH @ psi))
                            label = f'rollout_step{step:04d}'
                            model, row = fit(model, system, target, method, args, sr_settings, settings,
                                output/'histories'/f'{label}_{method}', label, args.adam_lr)
                            projection_loss = row['best_loss']
                            writer.writerow(dict(time=t, family='rollout', seed=args.seeds[0]) | row)
                            handle.flush()
                        psi = exact_state(model, system).amplitudes.detach()
                        tw.writerow(dict(step=step, time=t, projection_loss=projection_loss,
                            step_fidelity='' if step==0 else 1-projection_loss,
                            ed_fidelity=float(abs(torch.vdot(exact_at(t), psi))**2),
                            energy=float(torch.vdot(psi, matrix @ psi).real),
                            magnetization_z=float((psi.abs().square()*mz).sum())))
                        trajectory.flush()
                        torch.save(dict(time=t, model_state_dict=state_dict(model),
                            model_config=dict(num_visible=args.num_sites, num_hidden=model.num_hidden,
                                              dtype='torch.complex128'),
                            run_config=dict(dt=args.dt, projection_optimizer=dict(name=method))),
                            folder/f'ptvmc_step{step:04d}.pt')
                        if args.diagnose:
                            from neural_quantum_solver.experiments import TrajectoryPoint
                            points.append(TrajectoryPoint(t, deepcopy(model)))
                if args.diagnose:
                    from neural_quantum_solver.experiments import diagnose_trajectory, save_diagnostic_artifacts
                    rows, spectra = diagnose_trajectory(system, points)
                    save_diagnostic_artifacts(folder/'diagnostics', config=config, rows=rows,
                        summary=dict(qgt_spectra=spectra, trajectory_rows=rows), checkpoints={})
    print(f'全部完成：{output}', flush=True)


if __name__ == '__main__':
    main()
