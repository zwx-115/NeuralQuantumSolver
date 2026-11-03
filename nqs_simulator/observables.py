"""由祖先采样估计 Pauli 串与 Pauli 求和。

Z 对角项用 bitstring 符号；非对角 X/Y 用 NQS 振幅比。局域估计量的虚部予以记录，不丢弃。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from .circuits import apply_gate, Gate


PAULI_TOKEN = re.compile(r"([IXYZ])(\d+)")


@dataclass(frozen=True)
class PauliTerm:
    name: str
    pauli: str
    qubits: tuple[int, ...]
    coeff: complex = 1.0 + 0.0j

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pauli": self.pauli,
            "qubits": list(self.qubits),
            "coeff": {"real": float(self.coeff.real), "imag": float(self.coeff.imag)},
        }


def parse_pauli_label(label: str) -> tuple[str, tuple[int, ...]]:
    compact = str(label).replace(" ", "").replace("*", "").upper()
    parts = PAULI_TOKEN.findall(compact)
    rebuilt = "".join(op + str(q) for op, q in parts)
    if not parts or rebuilt != compact:
        raise ValueError(f"Cannot parse Pauli label {label!r}")
    ops = "".join(op for op, _ in parts)
    qubits = tuple(int(q) for _, q in parts)
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"Repeated qubit in Pauli label {label!r}")
    return ops, qubits


def parse_observables(observables, n_qubits: int) -> list[tuple[str, tuple[PauliTerm, ...]]]:
    if observables is None:
        return []
    if isinstance(observables, (str, dict, PauliTerm)):
        observables = [observables]
    parsed = []
    for item in observables:
        if isinstance(item, PauliTerm):
            _check_qubits(item.qubits, n_qubits, item.name)
            parsed.append((item.name, (item,)))
            continue
        if isinstance(item, str):
            ops, qubits = parse_pauli_label(item)
            _check_qubits(qubits, n_qubits, item)
            parsed.append((item, (PauliTerm(item, ops, qubits),)))
            continue
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported observable {item!r}")
        name = str(item.get("name") or item.get("pauli") or "observable")
        if "terms" in item:
            terms = []
            for term in item["terms"]:
                if isinstance(term, PauliTerm):
                    terms.append(term)
                    continue
                if isinstance(term, (list, tuple)) and len(term) >= 2:
                    label, coeff = term[0], term[1]
                    ops, qubits = parse_pauli_label(label)
                    terms.append(PauliTerm(label, ops, qubits, complex(coeff)))
                    continue
                label = term.get("pauli") or term.get("label")
                ops, qubits = parse_pauli_label(label)
                coeff = complex(term.get("coeff", 1.0))
                terms.append(PauliTerm(str(label), ops, qubits, coeff))
            for term in terms:
                _check_qubits(term.qubits, n_qubits, name)
            parsed.append((name, tuple(terms)))
            continue
        if "pauli" in item:
            ops, qubits = parse_pauli_label(item["pauli"])
            _check_qubits(qubits, n_qubits, name)
            parsed.append((name, (PauliTerm(name, ops, qubits, complex(item.get("coeff", 1.0))),)))
            continue
        raise ValueError(f"Observable dict needs 'pauli' or 'terms': {item}")
    return parsed


def _check_qubits(qubits, n_qubits, name):
    if any(q < 0 or q >= n_qubits for q in qubits):
        raise ValueError(f"Observable {name} qubit out of range for n={n_qubits}")


def _local_pauli(model, samples, term: PauliTerm):
    bits = samples.clone()
    phase = torch.ones(samples.shape[0], dtype=torch.complex64, device=samples.device)
    flipped = False
    for op, qubit in zip(term.pauli, term.qubits):
        xq = bits[:, qubit]
        sign = (1 - 2 * xq).to(torch.float32)
        if op == "I":
            continue
        if op == "Z":
            phase = phase * sign.to(phase.dtype)
        elif op == "X":
            bits[:, qubit] = bits[:, qubit] ^ 1
            flipped = True
        elif op == "Y":
            phase = phase * (-1j) * sign.to(phase.dtype)
            bits[:, qubit] = bits[:, qubit] ^ 1
            flipped = True
        else:
            raise ValueError(op)
    if flipped:
        ratio = torch.exp(model.log_psi(bits) - model.log_psi(samples))
    else:
        ratio = torch.ones(samples.shape[0], dtype=torch.cfloat, device=samples.device)
    coeff = torch.tensor(term.coeff, dtype=ratio.dtype, device=ratio.device)
    return coeff * phase.to(ratio.dtype) * ratio


@torch.no_grad()
def estimate_observables(model, observables, samples):
    parsed = parse_observables(observables, model.n_qubits)
    count = samples.shape[0]
    out = {}
    for name, terms in parsed:
        values = sum((_local_pauli(model, samples, term) for term in terms),
                     start=torch.zeros(count, dtype=torch.cfloat, device=samples.device))
        real = values.real
        imag = values.imag
        mean = float(real.mean())
        std = float(real.std(unbiased=True)) if count > 1 else 0.0
        out[name] = {
            "value": mean,
            "std_error": std / math.sqrt(count) if count else float("nan"),
            "imag": float(imag.mean()),
            "shots": count,
            "terms": [term.to_dict() for term in terms],
        }
    return out


def exact_observables(state, n_qubits, observables):
    parsed = parse_observables(observables, n_qubits)
    out = {}
    for name, terms in parsed:
        total = 0j
        for term in terms:
            evolved = state.clone()
            for op, qubit in zip(term.pauli, term.qubits):
                if op == "I":
                    continue
                evolved = apply_gate(evolved, Gate(op, (qubit,)), n_qubits)
            total = total + complex(term.coeff) * complex(torch.vdot(state, evolved).cpu())
        out[name] = {"value": float(total.real), "imag": float(total.imag)}
    return out
