# NeuralQuantumSolver

PyTorch framework for neural quantum-state ground-state optimization and
small-system exact diagonalization. This branch intentionally contains only
ground-state functionality; time evolution remains on branch `code-v1`.

## Install and test

Use the PyTorch build supplied by the target GPU environment, then install this
package without replacing that build:

```bash
python -m pip install -e . --no-deps
python -m pytest
```

- NVIDIA: use a CUDA-enabled PyTorch build.
- Moore Threads: use a `torch_musa` image in which `torch.device("musa")` works.
- CPU: use `DEVICE = "cpu"` and `NUM_GPUS = 1`.

The package does not import `torch_musa` directly. CUDA and MUSA therefore use
the same source code; the installed PyTorch runtime supplies the backend.

## Flexible ground-state run

Edit the settings block at the top of
`examples/ground_state/flexible_ground_state.py`:

```python
DEVICE = "cuda:0"  # NVIDIA; change only this to "musa" for Moore Threads
NUM_GPUS = 1       # 1, 2, 4, ... consecutive cards beginning at DEVICE
```

For example, `DEVICE = "cuda:1", NUM_GPUS = 2` uses `cuda:1` and `cuda:2`;
`DEVICE = "musa", NUM_GPUS = 4` uses `musa:0` through `musa:3`. One run must
use cards from one backend only.

The model, sampler, and optimizer are independent objects:

```python
model = ComplexRBM(num_visible=NUM_SITES, num_hidden=4 * NUM_SITES,
                   dtype=DTYPE, device=DEVICE, seed=SEED)
sampler = MetropolisSampler(num_chains=4096, thermal_sweeps=20,
                            sweeps=2, sweep_size=None)
optimizer = SR(learning_rate=0.05, regularization=1e-3,
               jacobian=LogJacobian(method="vmap", chunk_size=512))
state = VariationalState(system, model, sampler, seed=SEED,
                         num_gpus=NUM_GPUS)
result = GroundStateDriver(state, optimizer).run(steps=200)
```

Available combinations include:

- models: `ComplexRBM`, `ComplexFNN`, `LogAmplitudeTable`, or another model
  implementing `log_psi(configurations)`;
- samplers: `MetropolisSampler` or `ExactSampler`/`FullSumState`;
- optimizers: `Adam` or stochastic reconfiguration (`SR`);
- systems: tilted-field Ising, Heisenberg, XXZ, J1-J2, or a custom Pauli
  Hamiltonian.

`MetropolisSampler` retains `num_chains * sweeps` samples per optimization
step. `sweep_size=None` means `num_sites` single-spin proposals between two
retained samples. In a multi-card run, chains are divided between cards while
the total sample count remains unchanged.

Adam uses one VMC surrogate backward pass per card and never constructs a
per-sample Jacobian. SR supports sequential, `vmap`, and model-provided analytic
log-Jacobians. CUDA and MUSA both use `torch.linalg.solve` on the primary GPU
for the final regularized dense SR system.

## CUDA/MUSA benchmark records

The flexible example writes two files under `benchmark_results/`:

- `*_steps.csv`: one row per optimization step, including energy, variance,
  acceptance, sample count, gradient/update norms, sampling time, local-energy
  time, optimizer time, total step time, and SR Jacobian/QGT/solve timings.
- `*_metadata.json`: GPU/backend, device count, PyTorch/runtime versions, dtype,
  model parameter count, Hilbert size, sampler/optimizer settings, total
  optimization time, exact reference energy, final NQS energy, and steady-step
  timing summaries.

GPU operations are synchronized at every timing boundary, so asynchronous
kernel launch does not make CUDA or MUSA timings look artificially short. The
first step should be treated as warm-up; the JSON steady-state summaries exclude
it. Use the same model, dtype, seed, samples, lattice, optimizer settings, and
number of cards for a fair comparison, and repeat each run at least three times.

Also collect these vendor-tool measurements externally when available: peak
device memory, utilization, power, temperature, and clock frequency. Record any
unsupported operator/OOM traceback and whether the run completed; the solver
does not retry failed accelerator computation on CPU.

Observed MUSA backend limitations and same-device workarounds are tracked in
`docs/MUSA_COMPATIBILITY.md`.

## Exact diagonalization

Exact diagonalization consumes the same `PhysicalSystem` used by the NQS:

```python
import torch
from neural_quantum_solver import ExactDiagonalizer, tilted_field_ising

system = tilted_field_ising(6, coupling=1.0, field_x=0.5,
                            field_z=0.5, periodic=False)
result = ExactDiagonalizer(dtype=torch.complex128, device="cpu").ground_state(system)
print(result.energy)
```

Dense ED is a small-system reference and normally runs on CPU. NQS optimization
and its final full-sum evaluation remain on the selected CUDA or MUSA device.
There is no automatic accelerator-to-CPU computation fallback: an unsupported
GPU operation raises its backend error so it can be recorded and tested.

## Numerical conventions

- Configurations contain integer spins `+1` and `-1`.
- Exact ordering is binary ascending with bit `0 -> +1`.
- `log_psi(configurations)` returns one complex log amplitude per sample.
- `tilted_field_ising` implements `J sum ZZ - hx sum X - hz sum Z`.
- Dense exact methods reject systems larger than their configured safety limit.
