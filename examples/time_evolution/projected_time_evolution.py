"""Ground-state preparation followed by projected NQS real-time evolution.

This small-system reference uses full Hilbert-space sums for each projection.
The state is nevertheless represented by an NQS after every time step.  It is
intended as a stable baseline before introducing Monte Carlo overlap estimates.
"""

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import torch


# Allow this example to run directly without `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import (  # noqa: E402
    Adam,
    ComplexRBM,
    ExactDiagonalizer,
    GroundStateDriver,
    MetropolisSampler,
    VariationalState,
    tilted_field_ising,
)
from neural_quantum_solver.estimators import exact_energy, exact_state  # noqa: E402


# ---------------------------------------------------------------------------
# Editable experiment settings
# ---------------------------------------------------------------------------
NUM_SITES = 12
DT = 0.05
TIME_STEPS = 40

GROUND_STATE_STEPS = 100
GROUND_LEARNING_RATE = 0.01
NUM_CHAINS = 1000
THERMAL_SWEEPS = 5
SAMPLES_PER_CHAIN = 1

PROJECTION_STEPS = 100
PROJECTION_LEARNING_RATE = 0.01

# 保存诊断所需的实际 p-tVMC RBM；这些时间必须落在 DT 的整数步上。
SNAPSHOT_TIMES = tuple(round(0.1 * index, 10) for index in range(21))

DTYPE = torch.complex128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7
TRAJECTORY_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "projected_ptvmc_trajectory"


# The initial Hamiltonian defines the ground state to be prepared.
ground_system = tilted_field_ising(
    num_sites=NUM_SITES,
    coupling=0.0,
    field_x=0.5,
    field_z=0.0,
    periodic=False,
)

# At t=0 the field is quenched, producing nontrivial real-time dynamics.
evolution_system = tilted_field_ising(
    num_sites=NUM_SITES,
    coupling=1.0,
    field_x=0.5,
    field_z=0.5,
    periodic=False,
)


def save_trajectory_checkpoint(time: float, model: ComplexRBM) -> None:
    """保存指定时刻的 RBM 参数和可复现实验元数据。"""
    checkpoint = {
        "format_version": 1,
        "time": time,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": {
            "class": "ComplexRBM",
            "num_visible": NUM_SITES,
            "num_hidden": 4 * NUM_SITES,
            "dtype": str(DTYPE),
        },
        "run_config": {
            "dt": DT,
            "ground_system": {"coupling": 0.0, "field_x": 0.5, "field_z": 0.0},
            "evolution_system": {"coupling": 1.0, "field_x": 0.5, "field_z": 0.5},
            "ground_state_steps": GROUND_STATE_STEPS,
            "projection_steps": PROJECTION_STEPS,
            "projection_learning_rate": PROJECTION_LEARNING_RATE,
            "seed": SEED,
        },
    }
    torch.save(checkpoint, TRAJECTORY_DIRECTORY / f"ptvmc_t{time:.2f}.pt")


# ---------------------------------------------------------------------------
# Stage 1: prepare the NQS ground state
# ---------------------------------------------------------------------------
model = ComplexRBM(
    num_visible=NUM_SITES,
    num_hidden=4 * NUM_SITES,
    dtype=DTYPE,
    device=DEVICE,
    seed=SEED,
)

ground_sampler = MetropolisSampler(
    num_chains=NUM_CHAINS,
    thermal_sweeps=THERMAL_SWEEPS,
    sweeps=SAMPLES_PER_CHAIN,
    sweep_size=None,  # One num_sites-long interval between retained samples.
)
ground_state = VariationalState(
    system=ground_system,
    model=model,
    sampler=ground_sampler,
    seed=SEED,
)
ground_optimizer = Adam(learning_rate=GROUND_LEARNING_RATE)

print("Preparing the NQS ground state")
print(f"num_sites (L)          = {NUM_SITES}")
print(f"device                 = {DEVICE}")
print(f"samples per GS step    = {NUM_CHAINS * SAMPLES_PER_CHAIN}")
ground_result = GroundStateDriver(ground_state, ground_optimizer).run(
    steps=GROUND_STATE_STEPS,
    report_every=10,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "time_evolution_ground_state.pt",
)

# Materialize the trained and normalized NQS. This is the actual t=0 state.
initial_nqs_state = exact_state(model, ground_system).amplitudes.detach()

# ED ground state is used only as a small-system accuracy reference.
ground_ed = ExactDiagonalizer(dtype=DTYPE, device=DEVICE)
exact_initial = ground_ed.ground_state(ground_system)
initial_fidelity = torch.abs(torch.vdot(exact_initial.state, initial_nqs_state)) ** 2
initial_nqs_energy = exact_energy(model, ground_system).energy
initial_evolution_energy = exact_energy(model, evolution_system).energy

snapshot_steps = {round(time / DT): time for time in SNAPSHOT_TIMES}
if any(step < 0 or step > TIME_STEPS for step in snapshot_steps):
    raise ValueError("all SNAPSHOT_TIMES must lie between 0 and TIME_STEPS * DT")
TRAJECTORY_DIRECTORY.mkdir(parents=True, exist_ok=True)
trajectory_rows = []
save_trajectory_checkpoint(0.0, model)
trajectory_rows.append({
    "step": 0,
    "time": 0.0,
    "projection_loss": "",
    "step_fidelity": "",
    "ed_fidelity": float(initial_fidelity.item()),
    "energy": float(initial_evolution_energy.item()),
    "magnetization_z": "",
})

print(f"ED ground energy       = {exact_initial.energy.item():.12f}")
print(f"NQS ground energy      = {initial_nqs_energy.item():.12f}")
print(f"initial GS fidelity    = {initial_fidelity.item():.12f}")


# ---------------------------------------------------------------------------
# Stage 2: projected real-time evolution
# ---------------------------------------------------------------------------
# Diagonalize the post-quench Hamiltonian once and reuse its eigenbasis.
evolution_ed = ExactDiagonalizer(dtype=DTYPE, device=DEVICE)
eigenvalues, eigenvectors = evolution_ed.diagonalize(evolution_system)
hamiltonian = evolution_ed.matrix(evolution_system)
step_phase = torch.exp(-1j * DT * eigenvalues)
unitary_step = (eigenvectors * step_phase.unsqueeze(0)) @ eigenvectors.mH

# Exact reference trajectory starts from the exact initial ground state.
exact_coefficients = eigenvectors.mH @ exact_initial.state
all_configurations = evolution_system.hilbert.all_states(device=DEVICE)
magnetization_z = all_configurations.to(DTYPE).mean(dim=1)

current_state = initial_nqs_state
print("\nProjected real-time evolution")
print(f"num_sites (L)          = {NUM_SITES}")
print(f"dt / total time        = {DT} / {DT * TIME_STEPS}")
print(" step      time       loss        step fidelity   ED fidelity     energy          <Mz>")
print(f"{0:5d}  {0.0:8.4f}  {'-':>10}  {'-':>13}  {initial_fidelity.item():.10f}", flush=True)

for time_step in range(1, TIME_STEPS + 1):
    # Exact short-time target generated from the current projected NQS state.
    target_state = (unitary_step @ current_state).detach()

    # Warm-start the next NQS from the current NQS parameters and project the
    # short-time target back onto the variational manifold.
    next_model = deepcopy(model)
    projection_optimizer = torch.optim.Adam(
        next_model.parameters(), lr=PROJECTION_LEARNING_RATE
    )

    projection_loss = torch.tensor(float("nan"), device=DEVICE)
    for _ in range(PROJECTION_STEPS):
        projection_optimizer.zero_grad(set_to_none=True)
        candidate_state = exact_state(next_model, evolution_system).amplitudes
        overlap = torch.vdot(target_state, candidate_state)
        projection_loss = 1.0 - torch.abs(overlap) ** 2
        projection_loss.backward()
        projection_optimizer.step()

    model = next_model
    current_state = exact_state(model, evolution_system).amplitudes.detach()
    step_fidelity = torch.abs(torch.vdot(target_state, current_state)) ** 2

    time = time_step * DT
    exact_state_at_time = eigenvectors @ (
        torch.exp(-1j * time * eigenvalues) * exact_coefficients
    )
    exact_fidelity = torch.abs(torch.vdot(exact_state_at_time, current_state)) ** 2
    energy = torch.vdot(current_state, hamiltonian @ current_state).real
    mz = torch.sum(torch.abs(current_state) ** 2 * magnetization_z).real

    print(
        f"{time_step:5d}  {time:8.4f}  {projection_loss.item():.3e}  "
        f"{step_fidelity.item():.10f}  {exact_fidelity.item():.10f}  "
        f"{energy.item(): .10f}  {mz.item(): .8f}",
        flush=True,
    )
    trajectory_rows.append({
        "step": time_step,
        "time": time,
        "projection_loss": float(projection_loss.item()),
        "step_fidelity": float(step_fidelity.item()),
        "ed_fidelity": float(exact_fidelity.item()),
        "energy": float(energy.item()),
        "magnetization_z": float(mz.item()),
    })
    if time_step in snapshot_steps:
        save_trajectory_checkpoint(snapshot_steps[time_step], model)

with (TRAJECTORY_DIRECTORY / "trajectory.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=trajectory_rows[0].keys())
    writer.writeheader()
    writer.writerows(trajectory_rows)

with (TRAJECTORY_DIRECTORY / "run_config.json").open("w") as handle:
    json.dump(
        {
            "dt": DT,
            "time_steps": TIME_STEPS,
            "snapshot_times": SNAPSHOT_TIMES,
            "ground_system": {"coupling": 0.0, "field_x": 0.5, "field_z": 0.0},
            "evolution_system": {"coupling": 1.0, "field_x": 0.5, "field_z": 0.5},
            "seed": SEED,
        },
        handle,
        indent=2,
    )

