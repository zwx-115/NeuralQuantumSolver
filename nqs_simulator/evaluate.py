"""独立精确态评估。不作为训练标签。"""

from __future__ import annotations

import torch

from .circuits import apply_gate, apply_update, initial_state_vector
from .model import bits_from_indices, indices_from_bits


@torch.no_grad()
def neural_vector(model, batch_size=2048):
    n = model.n_qubits
    pieces = []
    device = model.positions.device
    for start in range(0, 1 << n, batch_size):
        end = min(start + batch_size, 1 << n)
        bits = bits_from_indices(torch.arange(start, end, device=device), n)
        pieces.append(model.amplitude(bits).to("cpu", dtype=torch.complex128))
    return torch.cat(pieces)


def infidelity(a, b):
    overlap = torch.vdot(a, b)
    return max(0.0, 1.0 - float(overlap.abs().square().real))


def fidelity(a, b):
    return 1.0 - infidelity(a, b)


def exact_evolve(circuit, updates=None):
    state, remaining = initial_state_vector(circuit)
    n = circuit.n_qubits
    if updates is None:
        for gate in remaining:
            state = apply_gate(state, gate, n)
        return state
    for update in updates:
        state = apply_update(state, update, n)
    return state


@torch.no_grad()
def sample_check(model, count, expected=None):
    samples = model.sample(count)
    counts = torch.bincount(indices_from_bits(samples).cpu(), minlength=1 << model.n_qubits).double()
    if expected is None:
        expected = neural_vector(model).abs().square()
    frequencies = counts / count
    tv = 0.5 * (frequencies - expected.cpu()).abs().sum()
    return {"sample_tv": float(tv), "sample_count": count}


def evaluate_step(model, teacher, update, exact_before, exact_after, eval_batch_size=2048):
    current = neural_vector(model, eval_batch_size)
    previous = neural_vector(teacher, eval_batch_size)
    teacher_target = apply_update(previous, update, model.n_qubits)
    return {
        "step_infidelity": infidelity(current, teacher_target),
        "total_infidelity": infidelity(current, exact_after),
        "normalization": float(current.abs().square().sum()),
    }


def numpy_apply_gate(state, gate, n):
    """独立的位索引门作用，仅用于数学检查。"""
    import numpy as np

    idx = np.arange(len(state), dtype=np.int64)
    bit = lambda q: (idx >> (n - 1 - q)) & 1
    name, qs, angle = gate.name, gate.qubits, gate.angle
    if name == "I":
        return state
    if name in ("H", "X", "Y", "RX", "RY", "SX"):
        q = qs[0]
        flipped = state[idx ^ (1 << (n - 1 - q))]
        b = bit(q)
        if name == "H":
            return ((1 - 2 * b) * state + flipped) / np.sqrt(2)
        if name == "X":
            return flipped
        if name == "Y":
            return np.where(b == 0, -1j * flipped, 1j * flipped)
        if name == "RX":
            c, s = np.cos(angle / 2), -1j * np.sin(angle / 2)
            return c * state + s * flipped
        if name == "RY":
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            return np.where(b == 0, c * state - s * flipped, s * flipped + c * state)
        # SX = 0.5 * [[1+i, 1-i], [1-i, 1+i]]，落入 H/X/Y/RX/RY 之后的默认分支
        a, bbit = (1 + 1j) / 2, (1 - 1j) / 2
        return np.where(b == 0, a * state + bbit * flipped, bbit * flipped + a * state)
    if name == "CNOT":
        return state[idx ^ (bit(qs[0]) << (n - 1 - qs[1]))]
    if name == "SWAP":
        src = ((bit(qs[0]) << (n - 1 - qs[1])) | (bit(qs[1]) << (n - 1 - qs[0])))
        dst = ((bit(qs[0]) << (n - 1 - qs[0])) | (bit(qs[1]) << (n - 1 - qs[1])))
        return state[idx ^ src ^ dst]
    if name in ("CZ", "CP", "RZZ", "CRZ"):
        if name == "CZ":
            phi = np.pi * bit(qs[0]) * bit(qs[1])
        elif name == "CP":
            phi = angle * bit(qs[0]) * bit(qs[1])
        elif name == "CRZ":
            phi = angle * bit(qs[0]) * (bit(qs[1]) - 0.5)
        else:
            phi = -0.5 * angle * (1 - 2 * bit(qs[0])) * (1 - 2 * bit(qs[1]))
        return state * np.exp(1j * phi)
    if name == "CH":
        control, target = qs
        active = bit(control) == 1
        flipped = state[idx ^ (1 << (n - 1 - target))]
        b = bit(target)
        hadamard = ((1 - 2 * b) * state + flipped) / np.sqrt(2)
        return np.where(active, hadamard, state)
    q = qs[0]
    b = bit(q)
    if name in ("S", "Z", "T", "TDG", "SDG", "P", "RZ"):
        if name == "S":
            phi = (np.pi / 2) * b
        elif name == "SDG":
            phi = -(np.pi / 2) * b
        elif name == "Z":
            phi = np.pi * b
        elif name == "T":
            phi = (np.pi / 4) * b
        elif name == "TDG":
            phi = -(np.pi / 4) * b
        elif name == "P":
            phi = angle * b
        else:
            phi = angle * (b - 0.5)
        return state * np.exp(1j * phi)
    raise ValueError(name)
