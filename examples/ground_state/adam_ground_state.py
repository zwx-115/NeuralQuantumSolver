"""Monte Carlo NQS ground-state optimization with Adam."""

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
from neural_quantum_solver.estimators import exact_energy  # noqa: E402


# Change these two values when moving between NVIDIA and Moore Threads.
DEVICE = "cuda:0"  # NVIDIA: "cuda:0"; Moore Threads: "musa"
NUM_GPUS = 1

# Physical system shared by ED and NQS.
system = tilted_field_ising(
    num_sites=10,
    coupling=1.0,
    field_x=0.5,
    field_z=0.5,
    periodic=False,
)

model = ComplexRBM(
    num_visible=system.hilbert.num_sites,
    num_hidden=4 * system.hilbert.num_sites,
    dtype=torch.complex64,
    device=DEVICE,
    seed=7,
)

# Total retained samples per optimization step are num_chains * sweeps.
# sweep_size=None means num_sites local MC updates between retained samples.
sampler = MetropolisSampler(
    num_chains=100,
    thermal_sweeps=20,
    sweeps=100,
    sweep_size=None,
)

variational_state = VariationalState(
    system=system,
    model=model,
    sampler=sampler,
    seed=10,
    num_gpus=NUM_GPUS,
)

# Adam uses one scalar VMC surrogate loss and one backward call. It does not
# construct the per-sample LogJacobian required by SR.
optimizer = Adam(
    learning_rate=0.001,
)

driver = GroundStateDriver(variational_state, optimizer)

print(f"primary device         = {DEVICE}")
print(f"number of GPUs         = {NUM_GPUS}")
print(f"samples per step       = {sampler.num_chains * sampler.sweeps}")
result = driver.run(
    steps=1000,
    report_every=50,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "ground_state_adam.pt",
)

# Evaluate the final NQS without Monte Carlo noise and compare with ED.
exact = ExactDiagonalizer(dtype=torch.complex128, device="cpu").ground_state(system)
final_full_sum_energy = exact_energy(model, system).energy.item()
print(f"exact ground energy    = {exact.energy.item():.12f}")
print(f"best sampled energy    = {result.best_energy:.12f}")
print(f"final NQS full sum     = {final_full_sum_energy:.12f}")
print(f"full-sum error         = {final_full_sum_energy - exact.energy.item():.3e}")
