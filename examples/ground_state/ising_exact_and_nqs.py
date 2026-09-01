from pathlib import Path
import sys

import torch

# Permit running this example directly from any working directory without first
# installing the package. Normal applications should still use `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_quantum_solver import ComplexRBM, ExactDiagonalizer, tilted_field_ising
from neural_quantum_solver.runners import ExactGroundStateRunner

#周期边界条件
periodic = True
n_sites= 8
alpha = 5

system = tilted_field_ising(n_sites, coupling=1.0, field_x=0.5, field_z=0.0, periodic=periodic)
exact = ExactDiagonalizer(dtype=torch.complex128).ground_state(system)
model = ComplexRBM(n_sites, alpha * n_sites, dtype=torch.complex128, seed=7)
fit = ExactGroundStateRunner(learning_rate=0.02, steps=500).run(model, system)

print(f"exact ground energy: {exact.energy.item():.12f}")
print(f"best NQS energy:     {fit.best_energy:.12f}")
