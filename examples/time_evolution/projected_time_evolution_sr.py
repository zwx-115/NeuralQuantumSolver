"""以全 Hilbert 空间 overlap-SR 投影运行小系统 p-tVMC 基线。"""

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import torch


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
from neural_quantum_solver.solvers import OverlapSRSettings, optimize_overlap_sr  # noqa: E402
from neural_quantum_solver.estimators import exact_energy, exact_state  # noqa: E402


NUM_SITES = 12
DT = 0.05
TIME_STEPS = 40
GROUND_STATE_STEPS = 100
GROUND_LEARNING_RATE = 0.01
NUM_CHAINS = 1000
THERMAL_SWEEPS = 5
SAMPLES_PER_CHAIN = 1
PROJECTION_STEPS = 40
SNAPSHOT_TIMES = tuple(round(0.1 * index, 10) for index in range(21))
DTYPE = torch.complex128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7
TRAJECTORY_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "projected_ptvmc_trajectory_sr"
SR_SETTINGS = OverlapSRSettings(
    learning_rate=0.05,
    regularization=1e-2,
    rcond=1e-12,
    jacobian_chunk_size=512,
    max_backtracks=8,
    backtrack_factor=0.5,
)


ground_system = tilted_field_ising(
    num_sites=NUM_SITES, coupling=0.0, field_x=0.5, field_z=0.0, periodic=False,
)
evolution_system = tilted_field_ising(
    num_sites=NUM_SITES, coupling=1.0, field_x=0.5, field_z=0.5, periodic=False,
)


def save_trajectory_checkpoint(time: float, model: ComplexRBM) -> None:
    """保存 SR p-tVMC 轨迹在指定时刻的 RBM 参数。"""
    torch.save(
        {
            "format_version": 1,
            "time": time,
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "model_config": {
                "class": "ComplexRBM", "num_visible": NUM_SITES,
                "num_hidden": 4 * NUM_SITES, "dtype": str(DTYPE),
            },
            "run_config": {
                "dt": DT,
                "ground_optimizer": {"name": "Adam", "learning_rate": GROUND_LEARNING_RATE},
                "projection_optimizer": {"name": "overlap_sr"} | SR_SETTINGS.config(),
                "seed": SEED,
            },
        },
        TRAJECTORY_DIRECTORY / f"ptvmc_t{time:.2f}.pt",
    )


model = ComplexRBM(
    num_visible=NUM_SITES, num_hidden=4 * NUM_SITES,
    dtype=DTYPE, device=DEVICE, seed=SEED,
)
ground_state = VariationalState(
    system=ground_system,
    model=model,
    sampler=MetropolisSampler(
        num_chains=NUM_CHAINS, thermal_sweeps=THERMAL_SWEEPS,
        sweeps=SAMPLES_PER_CHAIN, sweep_size=None,
    ),
    seed=SEED,
)
print("准备与 Adam 基线相同的初态；基态优化仍使用 Adam", flush=True)
GroundStateDriver(ground_state, Adam(learning_rate=GROUND_LEARNING_RATE)).run(
    steps=GROUND_STATE_STEPS,
    report_every=10,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "time_evolution_ground_state_sr.pt",
)

initial_nqs_state = exact_state(model, ground_system).amplitudes.detach()
ground_ed = ExactDiagonalizer(dtype=DTYPE, device=DEVICE)
exact_initial = ground_ed.ground_state(ground_system)
initial_fidelity = torch.abs(torch.vdot(exact_initial.state, initial_nqs_state)) ** 2
initial_evolution_energy = exact_energy(model, evolution_system).energy
snapshot_steps = {round(time / DT): time for time in SNAPSHOT_TIMES}
TRAJECTORY_DIRECTORY.mkdir(parents=True, exist_ok=True)
trajectory_rows: list[dict[str, float | int | str]] = [{
    "step": 0, "time": 0.0, "projection_loss": "", "step_fidelity": "",
    "ed_fidelity": float(initial_fidelity.item()), "energy": float(initial_evolution_energy.item()),
    "magnetization_z": "",
}]
sr_history: list[dict[str, float | int | bool]] = []
save_trajectory_checkpoint(0.0, model)

evolution_ed = ExactDiagonalizer(dtype=DTYPE, device=DEVICE)
eigenvalues, eigenvectors = evolution_ed.diagonalize(evolution_system)
hamiltonian = evolution_ed.matrix(evolution_system)
unitary_step = (eigenvectors * torch.exp(-1j * DT * eigenvalues).unsqueeze(0)) @ eigenvectors.mH
exact_coefficients = eigenvectors.mH @ exact_initial.state
all_configurations = evolution_system.hilbert.all_states(device=DEVICE)
magnetization_z = all_configurations.to(DTYPE).mean(dim=1)
current_state = initial_nqs_state

print("开始 overlap-SR projected p-tVMC", flush=True)
print(
    f"L={NUM_SITES}, dt={DT}, final_time={DT * TIME_STEPS}, "
    f"projection_steps={PROJECTION_STEPS}, SR={SR_SETTINGS.config()}",
    flush=True,
)
for time_step in range(1, TIME_STEPS + 1):
    target_state = (unitary_step @ current_state).detach()
    next_model = deepcopy(model)

    def record_step(record, iteration: int) -> None:
        sr_history.append({"time_step": time_step, "time": time_step * DT} | record.row(iteration))

    _, _, projection_loss, _, _, _ = optimize_overlap_sr(
        next_model,
        evolution_system,
        target_state,
        max_steps=PROJECTION_STEPS,
        tolerance=0.0,
        settings=SR_SETTINGS,
        on_step=record_step,
    )
    model = next_model
    current_state = exact_state(model, evolution_system).amplitudes.detach()
    step_fidelity = torch.abs(torch.vdot(target_state, current_state)) ** 2
    time = time_step * DT
    exact_state_at_time = eigenvectors @ (torch.exp(-1j * time * eigenvalues) * exact_coefficients)
    exact_fidelity = torch.abs(torch.vdot(exact_state_at_time, current_state)) ** 2
    energy = torch.vdot(current_state, hamiltonian @ current_state).real
    mz = torch.sum(torch.abs(current_state) ** 2 * magnetization_z).real
    print(
        f"{time_step:5d}  {time:8.4f}  {projection_loss:.3e}  {step_fidelity.item():.10f}  "
        f"{exact_fidelity.item():.10f}  {energy.item(): .10f}  {mz.item(): .8f}",
        flush=True,
    )
    trajectory_rows.append({
        "step": time_step, "time": time, "projection_loss": projection_loss,
        "step_fidelity": float(step_fidelity.item()), "ed_fidelity": float(exact_fidelity.item()),
        "energy": float(energy.item()), "magnetization_z": float(mz.item()),
    })
    if time_step in snapshot_steps:
        save_trajectory_checkpoint(snapshot_steps[time_step], model)

with (TRAJECTORY_DIRECTORY / "trajectory.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=trajectory_rows[0].keys())
    writer.writeheader()
    writer.writerows(trajectory_rows)
with (TRAJECTORY_DIRECTORY / "projection_sr_history.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=sr_history[0].keys())
    writer.writeheader()
    writer.writerows(sr_history)
with (TRAJECTORY_DIRECTORY / "run_config.json").open("w") as handle:
    json.dump({
        "num_sites": NUM_SITES, "dt": DT, "time_steps": TIME_STEPS,
        "snapshot_times": SNAPSHOT_TIMES, "dtype": str(DTYPE), "device": DEVICE, "seed": SEED,
        "ground_system": {"coupling": 0.0, "field_x": 0.5, "field_z": 0.0, "periodic": False},
        "evolution_system": {"coupling": 1.0, "field_x": 0.5, "field_z": 0.5, "periodic": False},
        "ground_optimizer": {"name": "Adam", "learning_rate": GROUND_LEARNING_RATE},
        "projection_optimizer": {"name": "overlap_sr"} | SR_SETTINGS.config(),
        "projection_steps": PROJECTION_STEPS,
    }, handle, indent=2)
print(f"SR trajectory 已写入 {TRAJECTORY_DIRECTORY}", flush=True)
