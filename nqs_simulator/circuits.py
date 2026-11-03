"""门描述、合法性检查，以及对角/局域更新计划。

门矩阵、HCZCH 连线和精确局域张量更新沿用已验证的 paper_reproduction/circuits.py。
调度器额外接受用户结构化线路，并合并同一物理层内连续对角门。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


GATE_ALIASES = {"CX": "CNOT", "U1": "P", "PHASE": "P"}
ONE_QUBIT = {
    "I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG", "SX", "P", "RX", "RY", "RZ",
}
TWO_QUBIT = {"CNOT", "CZ", "CH", "CY", "SWAP", "CP", "CRX", "CRY", "CRZ", "RZZ"}
DIAGONAL_GATES = {"I", "Z", "S", "SDG", "T", "TDG", "P", "RZ", "CZ", "CP", "CRZ", "RZZ"}
NEED_ANGLE = {"P", "RX", "RY", "RZ", "CP", "CRX", "CRY", "CRZ", "RZZ"}
SUPPORTED_GATES = ONE_QUBIT | TWO_QUBIT


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    physical_layer: int | None = None
    angle: float | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "qubits": list(self.qubits), "angle": self.angle,
                "physical_layer": self.physical_layer}


@dataclass(frozen=True)
class Update:
    name: str
    gates: tuple[Gate, ...]
    physical_layer: int
    diagonal: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "physical_layer": self.physical_layer, "diagonal": self.diagonal,
                "gates": [g.to_dict() for g in self.gates]}


@dataclass(frozen=True)
class Circuit:
    n_qubits: int
    gates: tuple[Gate, ...]
    initial_state: str = "plus"

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "initial_state": self.initial_state,
            "gates": [g.to_dict() for g in self.gates],
        }

    @classmethod
    def from_dict(cls, data) -> "Circuit":
        if isinstance(data, Circuit):
            return data
        if not isinstance(data, dict):
            raise TypeError("Circuit must be a dict or Circuit")
        n_qubits = int(data["n_qubits"])
        initial = str(data.get("initial_state", "plus")).lower()
        gates = tuple(_parse_gate(item) for item in data.get("gates", []))
        return validate_circuit(cls(n_qubits=n_qubits, gates=gates, initial_state=initial))


def _parse_gate(item) -> Gate:
    if isinstance(item, Gate):
        return item
    name = GATE_ALIASES.get(str(item["name"]).upper(), str(item["name"]).upper())
    qubits = tuple(int(q) for q in item["qubits"])
    angle = item.get("angle", None)
    if angle is not None:
        angle = float(angle)
    physical = item.get("physical_layer", None)
    if physical is not None:
        physical = int(physical)
    return Gate(name=name, qubits=qubits, physical_layer=physical, angle=angle)


def validate_circuit(circuit: Circuit) -> Circuit:
    if circuit.n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if circuit.initial_state not in ("plus", "zeros"):
        raise ValueError("initial_state must be 'plus' or 'zeros'")
    for gate in circuit.gates:
        if gate.name not in SUPPORTED_GATES:
            raise ValueError(f"Unsupported gate {gate.name}")
        expected = 1 if gate.name in ONE_QUBIT else 2
        if len(gate.qubits) != expected:
            raise ValueError(f"{gate.name} expects {expected} qubit(s), got {gate.qubits}")
        if len(set(gate.qubits)) != len(gate.qubits):
            raise ValueError(f"Repeated qubits in {gate.name} {gate.qubits}")
        if any(q < 0 or q >= circuit.n_qubits for q in gate.qubits):
            raise ValueError(f"Qubit out of range in {gate.name} {gate.qubits}")
        if gate.name in NEED_ANGLE and gate.angle is None:
            raise ValueError(f"{gate.name} requires an angle")
        if gate.name not in NEED_ANGLE and gate.angle is not None:
            raise ValueError(f"{gate.name} does not take an angle")
    return circuit


def _h_matrix(dtype, device):
    return torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device) / math.sqrt(2)


def _rx_matrix(angle, dtype, device):
    c, s = math.cos(angle / 2), -1j * math.sin(angle / 2)
    return torch.tensor([[c, s], [s, c]], dtype=dtype, device=device)


def _ry_matrix(angle, dtype, device):
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor([[c, -s], [s, c]], dtype=dtype, device=device)


def _y_matrix(dtype, device):
    return torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)


def _controlled(one_qubit, dtype, device):
    out = torch.eye(4, dtype=dtype, device=device)
    out[2:, 2:] = one_qubit
    return out


def gate_matrix(gate, device="cpu", dtype=torch.complex128):
    name = gate.name
    if name == "I":
        return torch.eye(2, dtype=dtype, device=device)
    if name == "H":
        return _h_matrix(dtype, device)
    if name == "X":
        return torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    if name == "Y":
        return _y_matrix(dtype, device)
    if name == "Z":
        return torch.diag(torch.tensor([1, -1], dtype=dtype, device=device))
    if name == "S":
        return torch.diag(torch.tensor([1, 1j], dtype=dtype, device=device))
    if name == "SDG":
        return torch.diag(torch.tensor([1, -1j], dtype=dtype, device=device))
    if name == "T":
        return torch.diag(torch.tensor(
            [1, complex(math.cos(math.pi / 4), math.sin(math.pi / 4))], dtype=dtype, device=device))
    if name == "TDG":
        return torch.diag(torch.tensor(
            [1, complex(math.cos(math.pi / 4), -math.sin(math.pi / 4))], dtype=dtype, device=device))
    if name == "SX":
        return 0.5 * torch.tensor([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=dtype, device=device)
    if name == "P":
        angle = gate.angle
        return torch.diag(torch.tensor([1, complex(math.cos(angle), math.sin(angle))], dtype=dtype, device=device))
    if name == "RZ":
        angle = gate.angle
        return torch.diag(torch.tensor(
            [complex(math.cos(angle / 2), -math.sin(angle / 2)),
             complex(math.cos(angle / 2), math.sin(angle / 2))], dtype=dtype, device=device))
    if name == "RX":
        return _rx_matrix(gate.angle, dtype, device)
    if name == "RY":
        return _ry_matrix(gate.angle, dtype, device)
    if name == "CNOT":
        return torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                             [0, 0, 0, 1], [0, 0, 1, 0]], dtype=dtype, device=device)
    if name == "CZ":
        return torch.diag(torch.tensor([1, 1, 1, -1], dtype=dtype, device=device))
    if name == "CH":
        return _controlled(_h_matrix(dtype, device), dtype, device)
    if name == "CY":
        return _controlled(_y_matrix(dtype, device), dtype, device)
    if name == "SWAP":
        return torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0],
                             [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype, device=device)
    if name == "CP":
        angle = gate.angle
        return torch.diag(torch.tensor(
            [1, 1, 1, complex(math.cos(angle), math.sin(angle))], dtype=dtype, device=device))
    if name == "CRX":
        return _controlled(_rx_matrix(gate.angle, dtype, device), dtype, device)
    if name == "CRY":
        return _controlled(_ry_matrix(gate.angle, dtype, device), dtype, device)
    if name == "CRZ":
        angle = gate.angle
        return torch.diag(torch.tensor(
            [1, 1,
             complex(math.cos(angle / 2), -math.sin(angle / 2)),
             complex(math.cos(angle / 2), math.sin(angle / 2))], dtype=dtype, device=device))
    if name == "RZZ":
        angle = gate.angle
        return torch.diag(torch.tensor(
            [complex(math.cos(angle / 2), -math.sin(angle / 2)),
             complex(math.cos(angle / 2), math.sin(angle / 2)),
             complex(math.cos(angle / 2), math.sin(angle / 2)),
             complex(math.cos(angle / 2), -math.sin(angle / 2))], dtype=dtype, device=device))
    raise ValueError(name)


def is_diagonal_gate(gate: Gate) -> bool:
    return gate.name in DIAGONAL_GATES


def make_hczch(n_qubits: int) -> Circuit:
    if n_qubits < 2 or n_qubits % 2:
        raise ValueError("HCZCH requires even N >= 2")
    h = n_qubits // 2

    def pair(name, offset, layer):
        return tuple(Gate(name, (a, h + (a + offset) % h), layer) for a in range(h))

    layers = [
        pair("CZ", 4, 1),
        tuple(Gate("T", (a,), 2) for a in range(n_qubits)),
        pair("CH", 5, 3),
        tuple(Gate("T", (a,), 4) for a in range(n_qubits)),
        pair("CNOT", 7, 5),
    ]
    gates = tuple(gate for layer in layers for gate in layer)
    return validate_circuit(Circuit(n_qubits=n_qubits, gates=gates, initial_state="plus"))


def make_bell(n_qubits: int = 2) -> Circuit:
    if n_qubits < 2:
        raise ValueError("Bell circuit needs at least 2 qubits")
    gates = (Gate("H", (0,), 1), Gate("CNOT", (0, 1), 2))
    extra = tuple(Gate("I", (q,), 3) for q in range(2, n_qubits))
    return validate_circuit(Circuit(n_qubits=n_qubits, gates=gates + extra, initial_state="zeros"))


def _skip_redundant_hadamards(circuit: Circuit) -> tuple[str, tuple[Gate, ...], dict]:
    """开头全体 H 由 plus 态输出头精确表示，不再训练。"""
    seen = []
    for gate in circuit.gates:
        if gate.name != "H" or len(gate.qubits) != 1:
            break
        qubit = gate.qubits[0]
        if qubit in seen:
            break
        seen.append(qubit)
    skipped = 0
    initial = circuit.initial_state
    gates = circuit.gates
    note = {}
    if len(seen) == circuit.n_qubits:
        skipped = circuit.n_qubits
        gates = circuit.gates[skipped:]
        initial = "plus"
        note = {"skipped_initial_hadamards": skipped, "effective_initial_state": "plus"}
    return initial, gates, note


def plan_updates(circuit: Circuit) -> tuple[tuple[Update, ...], dict]:
    circuit = validate_circuit(circuit)
    initial, gates, note = _skip_redundant_hadamards(circuit)
    note["effective_initial_state"] = initial if "effective_initial_state" not in note else note["effective_initial_state"]
    remaining = [g for g in gates if g.name != "I"]
    updates = []
    buffer: list[Gate] = []
    generated_layer = 0

    def _layer_of(gate: Gate) -> int:
        nonlocal generated_layer
        if gate.physical_layer is not None:
            return gate.physical_layer
        generated_layer += 1
        return generated_layer

    def _same_diagonal_group(current: Gate, incoming: Gate) -> bool:
        if current.physical_layer is None and incoming.physical_layer is None:
            return True
        return current.physical_layer is not None and current.physical_layer == incoming.physical_layer

    def flush_diagonal():
        if not buffer:
            return
        layer = _layer_of(buffer[0])
        named = {}
        for gate in buffer:
            named[gate.name] = named.get(gate.name, 0) + 1
        desc = "+".join(f"{name}x{count}" if count > 1 else name for name, count in named.items())
        updates.append(Update(f"diag_{layer}_{desc}", tuple(buffer), layer, True))
        buffer.clear()

    for gate in remaining:
        if is_diagonal_gate(gate):
            if buffer and not _same_diagonal_group(buffer[-1], gate):
                flush_diagonal()
            buffer.append(gate)
            continue
        flush_diagonal()
        layer = _layer_of(gate)
        qs = "_".join(str(q) for q in gate.qubits)
        updates.append(Update(f"{gate.name}_{qs}", (gate,), layer, False))
    flush_diagonal()
    note.update(n_qubits=circuit.n_qubits, n_input_gates=len(circuit.gates),
                n_trained_gates=len(remaining), n_updates=len(updates),
                original_initial_state=circuit.initial_state)
    return tuple(updates), note


def diagonal_phase(bits, gates):
    phase = torch.zeros(bits.shape[0], device=bits.device, dtype=torch.float32)
    for gate in gates:
        if gate.name == "I":
            continue
        if gate.name == "T":
            phase = phase + (math.pi / 4) * bits[:, gate.qubits[0]]
        elif gate.name == "TDG":
            phase = phase - (math.pi / 4) * bits[:, gate.qubits[0]]
        elif gate.name == "S":
            phase = phase + (math.pi / 2) * bits[:, gate.qubits[0]]
        elif gate.name == "SDG":
            phase = phase - (math.pi / 2) * bits[:, gate.qubits[0]]
        elif gate.name == "Z":
            phase = phase + math.pi * bits[:, gate.qubits[0]]
        elif gate.name == "P":
            phase = phase + gate.angle * bits[:, gate.qubits[0]]
        elif gate.name == "RZ":
            phase = phase + gate.angle * (bits[:, gate.qubits[0]] - 0.5)
        elif gate.name == "CZ":
            i, j = gate.qubits
            phase = phase + math.pi * bits[:, i] * bits[:, j]
        elif gate.name == "CP":
            i, j = gate.qubits
            phase = phase + gate.angle * bits[:, i] * bits[:, j]
        elif gate.name == "CRZ":
            control, target = gate.qubits
            phase = phase + gate.angle * bits[:, control] * (bits[:, target] - 0.5)
        elif gate.name == "RZZ":
            i, j = gate.qubits
            sign = (1 - 2 * bits[:, i]) * (1 - 2 * bits[:, j])
            phase = phase - 0.5 * gate.angle * sign
        else:
            raise ValueError("Non-diagonal gate in diagonal layer")
    return phase


def apply_gate(state, gate, n_qubits):
    """精确局域张量更新；q0 对应最高位轴。"""
    if gate.name == "I":
        return state
    active = list(gate.qubits)
    other = [i for i in range(n_qubits) if i not in active]
    order = other + active
    inverse = [order.index(i) for i in range(n_qubits)]
    k = 2 ** len(active)
    block = state.reshape([2] * n_qubits).permute(order).reshape(-1, k)
    out = block @ gate_matrix(gate, state.device, state.dtype).T
    return out.reshape([2] * n_qubits).permute(inverse).reshape(-1)


def apply_update(state, update, n_qubits):
    for gate in update.gates:
        state = apply_gate(state, gate, n_qubits)
    return state


def plus_state(n_qubits, device="cpu"):
    return torch.full((2 ** n_qubits,), 2 ** (-n_qubits / 2), dtype=torch.complex128, device=device)


def zero_state(n_qubits, device="cpu"):
    state = torch.zeros((2 ** n_qubits,), dtype=torch.complex128, device=device)
    state[0] = 1
    return state


def initial_state_vector(circuit: Circuit, device="cpu"):
    circuit = validate_circuit(circuit)
    initial, gates, _ = _skip_redundant_hadamards(circuit)
    if initial == "plus":
        state = plus_state(circuit.n_qubits, device)
    else:
        state = zero_state(circuit.n_qubits, device)
    return state, gates
