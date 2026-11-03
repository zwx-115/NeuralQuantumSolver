"""对抽出的计算内核做数学检查。不使用训练标签。"""

from __future__ import annotations

import torch

from .circuits import (
    Circuit, Gate, apply_gate, gate_matrix, make_hczch, plan_updates, plus_state,
)
from .evaluate import neural_vector, numpy_apply_gate, sample_check
from .losses import diagonal_loss, enumerate_active, local_loss
from .model import Statebank, bits_from_indices, expected_parameter_count


def run_checks() -> dict:
    torch.manual_seed(42)
    report = {}

    model = Statebank(4)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == expected_parameter_count(4), n_params
    report["parameters_n4"] = n_params

    all_bits = bits_from_indices(torch.arange(16), 4)
    plus = neural_vector(model)
    assert torch.allclose(plus, plus_state(4), atol=1e-6)
    assert abs(float(plus.abs().square().sum()) - 1) < 1e-6
    report["plus_state_ok"] = True

    zero_model = Statebank(4)
    zero_model.initialize_zero()
    zero = neural_vector(zero_model)
    p0 = float(zero[0].abs().square())
    assert p0 > 0.95, p0
    report["zero_state_prob0"] = p0

    torch.nn.init.normal_(model.output.weight, std=0.08)
    torch.nn.init.normal_(model.output.bias, std=0.1)
    for i in range(4):
        left = all_bits[0:1].clone()
        right = left.clone()
        right[0, i:] ^= 1
        p_left, _ = model.conditionals(left)
        p_right, _ = model.conditionals(right)
        assert torch.allclose(p_left[:, :i + 1], p_right[:, :i + 1], atol=1e-6)
    vector = neural_vector(model)
    assert abs(float(vector.abs().square().sum()) - 1) < 1e-6
    check = sample_check(model, 20000, vector.abs().square())
    assert check["sample_tv"] < 0.05, check
    report["sample_tv"] = check["sample_tv"]

    raw = torch.randn(16, dtype=torch.float64) + 1j * torch.randn(16, dtype=torch.float64)
    raw = raw / raw.norm()
    gates = [
        Gate("H", (2,)), Gate("T", (1,)), Gate("CZ", (0, 3)),
        Gate("CNOT", (3, 0)), Gate("CH", (0, 3)), Gate("CH", (3, 0)),
        Gate("X", (1,)), Gate("Y", (0,)), Gate("RX", (2,), angle=0.4),
        Gate("RY", (3,), angle=-0.7), Gate("SWAP", (0, 3)),
        Gate("CP", (1, 2), angle=0.3), Gate("P", (0,), angle=0.5),
        Gate("RZZ", (0, 2), angle=0.25), Gate("CRZ", (3, 1), angle=-0.8),
        Gate("SX", (2,)), Gate("CY", (1, 0)),
    ]
    numpy_raw = raw.numpy()
    worst_local = 0.0
    worst_numpy = 0.0
    for gate in gates:
        full = apply_gate(raw, gate, 4)
        configurations = enumerate_active(all_bits, gate.qubits)
        input_indices = (configurations * torch.tensor([8, 4, 2, 1])).sum(-1)
        local = raw[input_indices] @ gate_matrix(gate).T
        for row in range(16):
            err = float((full[input_indices[row]] - local[row]).abs().max())
            worst_local = max(worst_local, err)
            assert err < 1e-10, gate
        try:
            numpy_full = numpy_apply_gate(numpy_raw, gate, 4)
            numpy_err = float(abs(numpy_full - full.numpy()).max())
            worst_numpy = max(worst_numpy, numpy_err)
            assert numpy_err < 1e-10, (gate, numpy_err)
        except ValueError:
            pass
    report["local_vs_tensor_max_err"] = worst_local
    report["numpy_vs_tensor_max_err"] = worst_numpy

    teacher = Statebank(4)
    student = Statebank(4)
    student.load_state_dict(teacher.state_dict())
    total, _ = diagonal_loss(student, teacher, all_bits, ())
    assert abs(float(total.detach())) < 1e-8
    total.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
    student.zero_grad(set_to_none=True)
    configurations = enumerate_active(all_bits, (0, 3))
    old = teacher.amplitude(configurations.reshape(-1, 4)).reshape(16, 4).detach()
    total, components = local_loss(student, configurations, old)
    assert abs(float(total.detach())) < 1e-5, components
    total.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
    target_with_zeros = old.clone()
    target_with_zeros[:, 0] = 0
    total, _ = local_loss(student, configurations, target_with_zeros)
    assert torch.isfinite(total)
    report["identity_losses_ok"] = True

    updates, note = plan_updates(make_hczch(8))
    assert len(updates) == 11, (len(updates), [u.name for u in updates])
    assert all(u.diagonal for u in updates[:2])
    assert sum(1 for u in updates if u.name.startswith("CH")) == 4
    assert sum(1 for u in updates if u.name.startswith("CNOT")) == 4
    report["hczch_n8_updates"] = len(updates)

    merged, _ = plan_updates(Circuit(2, (
        Gate("T", (0,)), Gate("T", (1,)), Gate("CZ", (0, 1)), Gate("CNOT", (0, 1)), Gate("RZ", (1,), angle=0.2),
    ), "plus"))
    assert merged[0].diagonal and len(merged[0].gates) == 3
    assert not merged[1].diagonal and merged[1].gates[0].name == "CNOT"
    assert merged[2].diagonal and merged[2].gates[0].name == "RZ"
    report["diagonal_merge"] = [u.name for u in merged]

    skipped, skip_note = plan_updates(Circuit(2, (
        Gate("H", (0,)), Gate("H", (1,)), Gate("CNOT", (0, 1)),
    ), "zeros"))
    assert skip_note["effective_initial_state"] == "plus"
    assert skip_note["skipped_initial_hadamards"] == 2
    assert len(skipped) == 1 and skipped[0].gates[0].name == "CNOT"
    report["skip_initial_h"] = skip_note["skipped_initial_hadamards"]

    print("checks passed", report)
    return report


if __name__ == "__main__":
    run_checks()
