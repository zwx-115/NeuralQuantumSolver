"""Composable ground-state example, following a NetKet-like object workflow."""

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
    ExactSampler,
    ExactDiagonalizer,
    GroundStateDriver,
    LogJacobian,
    MetropolisSampler,
    SR,
    VariationalState,
    tilted_field_ising,
)
from neural_quantum_solver.estimators import exact_energy  # noqa: E402


# 1. Define one physical system shared by ED and NQS.
system = tilted_field_ising(
    num_sites=10,
    coupling=1.0,
    field_x=0.5,
    field_z=0.5,
    periodic=False,
)

# 2. Select any model implementing log_psi(configurations).
model = ComplexRBM(
    num_visible=system.hilbert.num_sites,
    num_hidden=4 * system.hilbert.num_sites,
    dtype=torch.complex128,
    device="cuda",
    seed=7,
)

# 3. Sampling is independent of the model, optimizer, and driver.
sampler = MetropolisSampler(
    num_chains=10000,
    thermal_sweeps=20,
    sweeps=1,       # Retained samples per chain; total = num_chains * sweeps.
    sweep_size=None,  # None means num_sites local updates between saved samples.
)

# For deterministic full-Hilbert-space summation, replace the line above with:
# sampler = ExactSampler()

variational_state = VariationalState(
    system=system,
    model=model,
    sampler=sampler,
    seed=7,
)

# 4. Optimization is independent of the sampler and model architecture.
# The per-sample log-Jacobian is an SR-only dependency; Adam does not use it.
jacobian = LogJacobian(
    method="vmap",
    chunk_size=1000,
)

optimizer = SR(
    learning_rate=0.05,
    regularization=1e-3,
    rcond=1e-12,
    jacobian=jacobian,
)

# To use Adam instead, replace the optimizer above with:
# optimizer = Adam(learning_rate=0.01)

# 5. The driver only orchestrates the variational state and optimizer.
driver = GroundStateDriver(variational_state, optimizer)
result = driver.run(
    steps=100,
    report_every=10,
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "ground_state.pt",
)

# ED consumes exactly the same PhysicalSystem and is an independent reference.
exact = ExactDiagonalizer(dtype=torch.complex128).ground_state(system)
final_full_sum_energy = exact_energy(model, system).energy.item()
print(f"exact ground energy = {exact.energy.item():.12f}")
print(f"best sampled energy = {result.best_energy:.12f}")
print(f"final NQS full sum  = {final_full_sum_energy:.12f}")
print(f"full-sum error      = {final_full_sum_energy - exact.energy.item():.3e}")
