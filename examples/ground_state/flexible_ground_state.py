"""Configurable CUDA/MUSA ground-state calculation."""

from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import (  # noqa: E402
    Adam,
    ComplexFNN,
    ComplexRBM,
    ExactDiagonalizer,
    ExactSampler,
    GroundStateDriver,
    LogAmplitudeTable,
    LogJacobian,
    MetropolisSampler,
    SR,
    VariationalState,
    tilted_field_ising,
    update_run_metadata,
)
from neural_quantum_solver.estimators import exact_energy  # noqa: E402


# ---------------------------------------------------------------------------
# Device settings: these are the only two lines needed when changing hardware.
# ---------------------------------------------------------------------------
DEVICE = "cuda:0"  # NVIDIA: "cuda:0"; Moore Threads: "musa"
NUM_GPUS = 1       # 1, 2, 4, ...; uses consecutive cards from DEVICE


# Physical-system settings.
NUM_SITES = 10
COUPLING = 1.0
FIELD_X = 0.5
FIELD_Z = 0.5
PERIODIC = False

# Shared run settings. complex64 is the portable GPU default; complex128 can be
# selected when the installed CUDA/MUSA PyTorch build supports it efficiently.
DTYPE = torch.complex64
SEED = 7
OPTIMIZATION_STEPS = 100
REPORT_EVERY = 10
RUN_NAME = f"sr_{DEVICE.replace(':', '')}_{NUM_GPUS}gpu"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_results"
HISTORY_PATH = BENCHMARK_DIR / f"{RUN_NAME}_steps.csv"
METADATA_PATH = BENCHMARK_DIR / f"{RUN_NAME}_metadata.json"

SCRIPT_STARTED = perf_counter()


system = tilted_field_ising(
    num_sites=NUM_SITES,
    coupling=COUPLING,
    field_x=FIELD_X,
    field_z=FIELD_Z,
    periodic=PERIODIC,
)

# Select one model. The two commented alternatives use the same state/driver.
model = ComplexRBM(
    num_visible=NUM_SITES,
    num_hidden=4 * NUM_SITES,
    dtype=DTYPE,
    device=DEVICE,
    seed=SEED,
)
# model = ComplexFNN([NUM_SITES, 4 * NUM_SITES, 1], dtype=DTYPE,
#                    device=DEVICE, seed=SEED)
# model = LogAmplitudeTable(NUM_SITES, dtype=DTYPE, device=DEVICE)

# Select one sampler. The total retained MC samples remain
# num_chains * sweeps, independent of NUM_GPUS.
sampler = MetropolisSampler(
    num_chains=10_000,
    thermal_sweeps=20,
    sweeps=1,
    sweep_size=None,
)
# sampler = ExactSampler()

state = VariationalState(
    system=system,
    model=model,
    sampler=sampler,
    seed=SEED,
    num_gpus=NUM_GPUS,
)

# Select one optimizer. LogJacobian belongs only to SR and is never evaluated
# by Adam. solver_device="auto" solves on the primary CUDA or MUSA GPU.
optimizer = SR(
    learning_rate=0.05,
    regularization=1e-3,
    rcond=1e-12,
    jacobian=LogJacobian(method="vmap", chunk_size=1_000),
    solver_device="auto",
)
# optimizer = Adam(learning_rate=0.001)

print(f"primary device          = {DEVICE}")
print(f"number of GPUs          = {NUM_GPUS}")
print(f"model                    = {type(model).__name__}")
print(f"sampler                  = {type(sampler).__name__}")
print(f"optimizer                = {type(optimizer).__name__}")

result = GroundStateDriver(state, optimizer).run(
    steps=OPTIMIZATION_STEPS,
    report_every=REPORT_EVERY,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "ground_state.pt",
    history_path=HISTORY_PATH,
    metadata_path=METADATA_PATH,
    experiment_label=RUN_NAME,
    run_metadata={
        "coupling": COUPLING,
        "field_x": FIELD_X,
        "field_z": FIELD_Z,
        "periodic": PERIODIC,
        "optimization_steps": OPTIMIZATION_STEPS,
    },
)

# The NQS full-sum evaluation remains on DEVICE, so backend failures are visible
# instead of being hidden by a CPU fallback. Dense ED is an independent CPU
# complex128 reference and does not participate in NQS optimization.
exact = ExactDiagonalizer(dtype=torch.complex128, device="cpu").ground_state(system)
final_full_sum_energy = exact_energy(model, system).energy.item()
steady_steps = result.history[1:] or result.history
solve_times = [
    step.optimizer_metrics["solve_seconds"]
    for step in steady_steps
    if "solve_seconds" in step.optimizer_metrics
]
update_run_metadata(
    METADATA_PATH,
    exact_ground_energy=exact.energy.item(),
    final_nqs_full_sum_energy=final_full_sum_energy,
    final_energy_error=final_full_sum_energy - exact.energy.item(),
    mean_steady_step_seconds=mean(step.step_seconds for step in steady_steps),
    median_steady_step_seconds=median(step.step_seconds for step in steady_steps),
    mean_steady_solve_seconds=mean(solve_times) if solve_times else None,
    end_to_end_seconds=perf_counter() - SCRIPT_STARTED,
)

print(f"exact ground energy      = {exact.energy.item():.12f}")
print(f"best sampled energy      = {result.best_energy:.12f}")
print(f"final NQS full sum       = {final_full_sum_energy:.12f}")
print(f"full-sum error           = {final_full_sum_energy - exact.energy.item():.3e}")
print(f"step history             = {HISTORY_PATH}")
print(f"run metadata             = {METADATA_PATH}")
