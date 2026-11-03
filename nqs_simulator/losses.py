"""对角门与局部门更新损失。

自已验证的 paper_reproduction/losses.py 抽出。教师振幅和采样不求梯度，学生振幅求梯度。
"""

from __future__ import annotations

import torch

from .circuits import diagonal_phase, gate_matrix


@torch.no_grad()
def mixed_samples(teacher, batch_size, eta):
    bits = teacher.sample(batch_size)
    use_uniform = torch.rand(batch_size, device=bits.device) < eta
    if use_uniform.any():
        bits[use_uniform] = torch.randint(0, 2, (int(use_uniform.sum()), teacher.n_qubits), device=bits.device)
    return bits


def enumerate_active(bits, active):
    """[B,N] -> [B,2**|S|,N]，活动比特顺序与门矩阵一致。"""
    k = 1 << len(active)
    out = bits[:, None, :].expand(-1, k, -1).clone()
    for alpha in range(k):
        for j, qubit in enumerate(active):
            out[:, alpha, qubit] = (alpha >> (len(active) - 1 - j)) & 1
    return out


def diagonal_loss(student, teacher, bits, gates):
    with torch.no_grad():
        old_a, old_phi = teacher.log_amplitude_phase(bits)
        target_phase = diagonal_phase(bits, gates)
    a, phi = student.log_amplitude_phase(bits)
    residual = phi - old_phi - target_phase
    centered = residual - residual.mean()
    magnitude = ((a - old_a) ** 2).mean()
    phase = 4 * (centered ** 2).mean()
    return magnitude + phase, {"magnitude": magnitude, "phase": phase}


def local_teacher(teacher, bits, gate):
    configurations = enumerate_active(bits, gate.qubits)
    b, k, n = configurations.shape
    with torch.no_grad():
        old = teacher.amplitude(configurations.reshape(b * k, n)).reshape(b, k)
        matrix = gate_matrix(gate, bits.device, old.dtype)
        target = old @ matrix.T
    return configurations, target


def local_loss(student, configurations, target):
    b, k, n = configurations.shape
    source = student.amplitude(configurations.reshape(b * k, n)).reshape(b, k)
    star_mass = target.abs().square().sum(-1)
    model_mass = source.abs().square().sum(-1)
    tiny = torch.finfo(star_mass.dtype).tiny
    p = target.abs().square() / star_mass.clamp_min(tiny)[:, None]
    q = source.abs().square() / model_mass.clamp_min(tiny)[:, None]
    positive = p > 0
    kl_terms = torch.where(positive, p * (torch.log(p.clamp_min(tiny)) - torch.log(q.clamp_min(tiny))), 0)
    kl = 4 * kl_terms.sum(-1).mean()
    phase_difference = torch.angle(source) - torch.angle(target)
    phase = 4 * (p * (1 - torch.cos(phase_difference))).sum(-1).mean()
    mass = 4 * ((torch.log(model_mass.clamp_min(tiny)) - torch.log(star_mass.clamp_min(tiny))) ** 2).mean()
    overlap = (target.conj() * source).sum(-1).abs().square() / (star_mass * model_mass).clamp_min(tiny)
    overlap_error = (1 - overlap).mean()
    return kl + phase + mass + overlap_error, {
        "kl": kl, "phase": phase, "mass": mass, "overlap": overlap_error}
