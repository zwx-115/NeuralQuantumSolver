"""全求和的子空间二阶量子信赖域；实验算法，不替换 Adam/SR 基线。

复参数按各张量实虚部交错展开为实坐标。用归一化波函数的 JVP/VJP
计算实量子度量，用 Hessian–向量乘积保留完整 infidelity 曲率。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch
from torch.func import functional_call, jvp, vjp

from ..models import NeuralQuantumState
from ..systems import PhysicalSystem


@dataclass(frozen=True)
class CurvatureSettings:
    """信赖域半径是局部量子距离，parameter_radius 是欧氏参数上限。"""

    radius: float = 0.1
    min_radius: float = 1e-7
    max_radius: float = 0.5
    parameter_radius: float = 0.5
    regularization: float = 0.01
    cg_steps: int = 12
    cg_rtol: float = 1e-6
    max_trials: int = 6
    accept_ratio: float = 0.1
    curvature: str = "full"

    def __post_init__(self) -> None:
        values = (self.radius, self.min_radius, self.max_radius, self.parameter_radius,
                  self.regularization, self.cg_rtol, self.accept_ratio)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("所有半径、正则化及阈值必须有限且为正数")
        if not self.min_radius <= self.radius <= self.max_radius or self.accept_ratio >= 1:
            raise ValueError("信赖域半径范围或接受比率不合法")
        if self.cg_steps < 1 or self.max_trials < 1:
            raise ValueError("CG 与试探次数必须为正整数")
        if self.curvature not in ("full", "gauss_newton"):
            raise ValueError("curvature 必须为 full 或 gauss_newton")

    def config(self) -> dict:
        """返回可保存的配置。"""
        return asdict(self)


class FunctionalWavefunction:
    """通过函数式调用获得对实坐标可二阶求导的归一化态，不改变原模型。"""

    def __init__(self, model: NeuralQuantumState, system: PhysicalSystem):
        self.model = model
        self.parameters = dict(model.named_parameters())
        first = next(iter(self.parameters.values()))
        if any(p.dtype != first.dtype or p.device != first.device for p in self.parameters.values()):
            raise ValueError("当前实现要求所有参数具有相同 dtype/device")
        if first.dtype not in (torch.complex128, torch.complex64):
            raise ValueError("当前版本支持复数 NQS 参数")
        if any(not p.requires_grad for p in self.parameters.values()):
            raise ValueError("当前版本要求所有模型参数可训练")
        self.configurations = system.hilbert.all_states(device=first.device)

    def pack(self) -> torch.Tensor:
        """每个复参数按 real, imag 顺序展平。"""
        return torch.cat([torch.view_as_real(p.detach()).reshape(-1)
                          for p in self.parameters.values()]).clone()

    def unpack(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """恢复参数视图，同时保留对实坐标的计算图。"""
        result, offset = {}, 0
        for name, p in self.parameters.items():
            count = 2*p.numel()
            result[name] = torch.view_as_complex(x[offset:offset+count].reshape(-1, 2).contiguous()).reshape(p.shape)
            offset += count
        if offset != x.numel():
            raise ValueError("参数向量长度不匹配")
        return result

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        logs = functional_call(self.model, self.unpack(x), (self.configurations,))
        return torch.exp(logs - 0.5*torch.logsumexp(2*logs.real, dim=0))

    @torch.no_grad()
    def assign(self, x: torch.Tensor) -> None:
        """仅在接受更新后写入原模型。"""
        for name, value in self.unpack(x).items():
            self.parameters[name].copy_(value)


def solve_trust_region(g: torch.Tensor, h: torch.Tensor, metric: torch.Tensor,
                       radius: float) -> torch.Tensor:
    """求小型二次信赖域全局解，含梯度正交于负曲率方向的 hard case。"""
    chol = torch.linalg.cholesky(metric)
    w = torch.linalg.solve_triangular(chol.mT, torch.eye(g.numel(), dtype=g.dtype, device=g.device), upper=True)
    a = w.mT @ h @ w
    eig, q = torch.linalg.eigh((a+a.mT)/2)
    b = q.mT @ (w.mT @ g)
    lower = max(0., -float(eig[0]))
    shifted = eig + lower
    eps = 100*torch.finfo(g.dtype).eps*max(1., float(eig.abs().max()))
    null = shifted <= eps
    y = torch.zeros_like(b)
    y[~null] = -b[~null]/shifted[~null]
    if float(b[null].norm()) <= eps and float(y.norm()) <= radius:
        if lower > 0:
            # 负曲率 hard case：沿最低特征向量走到边界。
            y[0] += torch.sqrt((radius**2 - y.square().sum()).clamp_min(0))
        return w @ (q @ y)
    lo = lower
    hi = lower + max(1., float(b.norm())/radius)
    for _ in range(80):
        mid = (lo+hi)/2
        if float((b/(eig+mid)).norm()) > radius:
            lo = mid
        else:
            hi = mid
    return w @ (q @ (-b/(eig+hi)))


@dataclass(frozen=True)
class CurvatureStep:
    """一次外部优化迭代；trials 内含被拒绝的试探。"""

    loss_before: float
    loss_after: float
    accepted: bool
    trials: int
    radius: float
    next_radius: float
    rho: float | None
    predicted_reduction: float
    gradient_norm: float
    update_norm: float
    quantum_distance_squared: float
    subspace_dimension: int
    min_reduced_hessian_eigenvalue: float
    cg_iterations: int
    cg_relative_residual: float
    nonfinite_trials: int
    status: str

    def row(self, iteration: int) -> dict:
        """转成 CSV 可记录的标量。"""
        return {"iteration": iteration} | asdict(self)


class OverlapCurvature:
    """为一个冻结目标维护信赖域与上一条接受方向；换目标时重新实例化。"""

    def __init__(self, model: NeuralQuantumState, system: PhysicalSystem,
                 target: torch.Tensor, settings: CurvatureSettings):
        self.wave = FunctionalWavefunction(model, system)
        p = next(model.parameters())
        target = target.detach().to(device=p.device, dtype=p.dtype).clone()
        if target.shape != (system.hilbert.size,) or not torch.isfinite(target).all() or target.norm() == 0:
            raise ValueError("目标必须是有限、非零且维度匹配的态矢量")
        self.target = target/target.norm()
        self.settings = settings
        self.radius = settings.radius
        self.previous: torch.Tensor | None = None

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """投影残差范数与归一化 infidelity 恒等，避免 1-F 的消减误差。"""
        psi = self.wave(x)
        residual = psi - self.target*torch.vdot(self.target, psi)
        return torch.vdot(residual, residual).real

    def state_dict(self) -> dict:
        """与模型 checkpoint 一起保存；恢复时必须保持相同目标及配置。"""
        return {"radius": self.radius, "previous": None if self.previous is None else self.previous.cpu().clone()}

    def load_state_dict(self, state: dict) -> None:
        """恢复同一冻结目标的优化器状态。"""
        radius = float(state["radius"])
        if not self.settings.min_radius <= radius <= self.settings.max_radius:
            raise ValueError("checkpoint 信赖域半径超出当前配置")
        self.radius = radius
        old = state["previous"]
        x = self.wave.pack()
        if old is not None and (old.shape != x.shape or not torch.isfinite(old).all()):
            raise ValueError("checkpoint 更新方向不匹配")
        self.previous = None if old is None else old.to(x).clone()

    def step(self) -> CurvatureStep:
        """在至多四维子空间内尝试更新，拒绝时模型保持原样。"""
        settings = self.settings
        x = self.wave.pack().requires_grad_()
        loss = self.loss(x)
        g = torch.autograd.grad(loss, x, create_graph=True)[0]
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise FloatingPointError("当前 loss 或梯度含非有限值")
        x0, grad = x.detach(), g.detach()
        psi, pullback = vjp(self.wave, x0)

        def metric_vector(v):
            _, tangent = jvp(self.wave, (x0,), (v,))
            horizontal = tangent - psi*torch.vdot(psi, tangent)
            return pullback(horizontal)[0].detach()

        # 实坐标 CG 求 (G+lambda I)p=-g；不生成完整 Jacobian/QGT。
        natural = torch.zeros_like(grad)
        residual = -grad.clone()
        direction = residual.clone()
        rr = residual @ residual
        initial_norm = float(residual.norm())
        cg_iterations = 0
        for k in range(settings.cg_steps):
            if float(residual.norm()) <= settings.cg_rtol*initial_norm or initial_norm == 0:
                break
            product = metric_vector(direction) + settings.regularization*direction
            denom = direction @ product
            if not torch.isfinite(denom) or denom <= 0:
                raise FloatingPointError("正则化量子度量 CG 出现非正或非有限曲率")
            alpha = rr/denom
            natural += alpha*direction
            residual -= alpha*product
            new_rr = residual @ residual
            direction = residual + (new_rr/rr)*direction
            rr = new_rr
            cg_iterations = k+1
        cg_residual = float(residual.norm())/initial_norm if initial_norm else 0.
        hv = torch.autograd.grad(g @ natural, x, retain_graph=True)[0].detach()
        candidates = [natural, -grad, hv]
        if self.previous is not None:
            candidates.append(self.previous)
        basis = []
        for candidate in candidates:
            norm = float(candidate.norm())
            if norm <= 1e-14:
                continue
            v = candidate/norm
            # 二次正交化抑制近共线方向带来的病态。
            for _ in range(2):
                for b in basis:
                    v = v - (b @ v)*b
            if float(v.norm()) > 1e-8:
                basis.append(v/v.norm())
        before = float(loss.detach())
        if not basis:
            return CurvatureStep(before, before, False, 0, self.radius, self.radius,
                None, 0., float(grad.norm()), 0., 0., 0, 0., cg_iterations,
                cg_residual, 0, "stationary_direction")
        vmat = torch.stack(basis, dim=1).detach()
        tangents = torch.stack([jvp(self.wave, (x0,), (v,))[1] for v in basis], dim=1)
        horizontal = tangents - psi[:, None]*(psi.conj() @ tangents)[None, :]
        metric = (horizontal.mH @ horizontal).real.detach()
        if settings.curvature == "full":
            hcols = [torch.autograd.grad(g @ v, x, retain_graph=True)[0].detach() for v in basis]
            h = vmat.mT @ torch.stack(hcols, dim=1)
        else:
            jr = tangents - self.target[:, None]*(self.target.conj() @ tangents)[None, :]
            h = 2*(jr.mH @ jr).real.detach()
        h = (h+h.mT)/2
        reduced_g = vmat.mT @ grad
        if not torch.isfinite(h).all() or not torch.isfinite(metric).all():
            raise FloatingPointError("子空间 Hessian 或量子度量含非有限值")
        eigen_min = float(torch.linalg.eigvalsh(h)[0])
        accepted, after, rho, predicted = False, before, None, 0.
        update_norm, distance, nonfinite = 0., 0., 0
        radius = self.radius
        for trial in range(1, settings.max_trials+1):
            radius = self.radius
            # 合并椭球同时保证局部量子距离与欧氏参数长度有界。
            bounded_metric = metric + (radius/settings.parameter_radius)**2*torch.eye(len(basis), device=x.device, dtype=x.dtype)
            z = solve_trust_region(reduced_g, h, bounded_metric, radius)
            update = vmat @ z
            predicted = float(-(reduced_g @ z + .5*z @ h @ z))
            if predicted > 0:
                with torch.no_grad():
                    candidate = self.wave(x0+update)
                    r = candidate - self.target*torch.vdot(self.target, candidate)
                    actual_loss = float(torch.vdot(r, r).real)
                    # 用正交残差计算实际态间 infidelity，避免近一保真度的消减。
                    delta = candidate - psi*torch.vdot(psi, candidate)
                    actual_distance = float(torch.vdot(delta, delta).real)
                if math.isfinite(actual_loss) and math.isfinite(actual_distance):
                    rho = (before-actual_loss)/predicted
                    if rho >= settings.accept_ratio and actual_loss < before and actual_distance <= radius**2:
                        self.wave.assign(x0+update)
                        self.previous = update.detach()
                        accepted, after = True, actual_loss
                        update_norm, distance = float(update.norm()), actual_distance
                        if rho > .75 and float(z @ bounded_metric @ z) > .8*radius**2:
                            self.radius = min(settings.max_radius, 2*radius)
                        break
                else:
                    nonfinite += 1
                    rho = None
            self.radius = max(settings.min_radius, radius*.5)
        return CurvatureStep(before, after, accepted, trial, radius, self.radius, rho,
            predicted, float(grad.norm()), update_norm, distance, len(basis), eigen_min,
            cg_iterations, cg_residual, nonfinite, "accepted" if accepted else "rejected")


def optimize_overlap_curvature(
    model: NeuralQuantumState, system: PhysicalSystem, target: torch.Tensor, *,
    max_steps: int, tolerance: float, settings: CurvatureSettings,
    on_step: Callable[[CurvatureStep, int], None] | None = None,
) -> tuple[float, float, float, int, bool, dict[str, torch.Tensor]]:
    """原地拟合冻结目标；返回与 optimize_overlap_sr 相同顺序的结果。"""
    if max_steps < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max_steps 必须为正且 tolerance 必须有限非负")
    optimizer = OverlapCurvature(model, system, target, settings)
    initial = float(optimizer.loss(optimizer.wave.pack()).detach())
    final, iterations = initial, 0
    for iteration in range(1, max_steps+1):
        if final <= tolerance:
            break
        record = optimizer.step()
        final, iterations = record.loss_after, iteration
        if on_step is not None:
            on_step(record, iteration)
        if record.status == "stationary_direction" or (not record.accepted and optimizer.radius <= settings.min_radius):
            break
    # 只接受下降步，因此最终态就是本次调用的最佳态。
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return initial, final, final, iterations, final <= tolerance, best
