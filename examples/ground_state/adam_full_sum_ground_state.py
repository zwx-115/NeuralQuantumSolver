"""NQS ground-state optimization with Adam and exact Hilbert-space sums."""

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
    FullSumState,
    GroundStateDriver,
    tilted_field_ising,
)
from neural_quantum_solver.estimators import exact_energy  # noqa: E402


# ---------------------------------------------------------------------------
# Editable experiment settings
# ---------------------------------------------------------------------------
NUM_SITES = 14
COUPLING = 1.0
FIELD_X = 0.5
FIELD_Z = 0.5
PERIODIC = False

HIDDEN_DENSITY = 4
LEARNING_RATE = 0.01
OPTIMIZATION_STEPS = 300
REPORT_EVERY = 10

DTYPE = torch.complex64
DEVICE = "cuda:0"  # NVIDIA: "cuda:0"; Moore Threads: "musa"
NUM_GPUS = 1
SEED = 7


# ED and NQS consume exactly the same physical-system object.
system = tilted_field_ising(
    num_sites=NUM_SITES,
    coupling=COUPLING,
    field_x=FIELD_X,
    field_z=FIELD_Z,
    periodic=PERIODIC,
)

model = ComplexRBM(
    num_visible=NUM_SITES,
    num_hidden=HIDDEN_DENSITY * NUM_SITES,
    dtype=DTYPE,
    device=DEVICE,
    seed=SEED,
)

# FullSumState enumerates all 2**NUM_SITES configurations with their exact Born
# probabilities. There are no Markov chains, thermal sweeps, or MC acceptance.
variational_state = FullSumState(
    system=system,
    model=model,
    seed=SEED,
    num_gpus=NUM_GPUS,
)

optimizer = Adam(
    learning_rate=LEARNING_RATE,
)

print(f"device                 = {DEVICE}")
print(f"number of GPUs         = {NUM_GPUS}")
print(f"Hilbert-space size     = {system.hilbert.size}")
print("sampling               = exact full summation")

result = GroundStateDriver(variational_state, optimizer).run(
    steps=OPTIMIZATION_STEPS,
    report_every=REPORT_EVERY,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "ground_state_adam_full_sum.pt",
)

# Both quantities below are noise-free. The first is the exact eigenvalue; the
# second is the exact energy expectation of the final NQS wavefunction.
exact = ExactDiagonalizer(dtype=torch.complex128, device="cpu").ground_state(system)
final_nqs_energy = exact_energy(model, system).energy.item()

print(f"exact ground energy    = {exact.energy.item():.12f}")
print(f"best NQS energy        = {result.best_energy:.12f}")
print(f"final NQS energy       = {final_nqs_energy:.12f}")
print(f"final energy error     = {final_nqs_energy - exact.energy.item():.3e}")
