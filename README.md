# NeuralQuantumSolver

PyTorch framework for small-system exact diagonalization, neural quantum-state
ground states, and projected real-time evolution diagnostics.

The first implementation milestone intentionally uses exact Hilbert-space sums.
This gives a `complex128` reference before Monte Carlo noise, SR conditioning,
or accumulated projection errors are introduced.

## Install and test

```bash
python -m pip install -e .
python -m pytest
```

## Minimal example

```python
import torch
from neural_quantum_solver.exact import ExactDiagonalizer
from neural_quantum_solver.systems import tilted_field_ising

system = tilted_field_ising(6, coupling=1.0, field_x=0.5, field_z=0.5)
result = ExactDiagonalizer(dtype=torch.complex128).ground_state(system)
print(result.energy)
```

See `examples/ground_state/ising_exact_and_nqs.py` for an NQS ground-state run.
It can also be run directly, without installation:

```bash
python examples/ground_state/ising_exact_and_nqs.py
```

## Composable ground-state API

Ground-state experiments use independent model, sampler, optimizer, and driver
objects. Switching Exact/Metropolis or Adam/SR does not change the driver:

```python
from neural_quantum_solver import (
    ComplexRBM, GroundStateDriver, LogJacobian, MetropolisSampler, SR,
    VariationalState, tilted_field_ising,
)

system = tilted_field_ising(8, periodic=False)
model = ComplexRBM(8, 40)
sampler = MetropolisSampler(
    num_chains=1024,
    thermal_sweeps=100,
    sweeps=4,
    sweep_size=None,
)
jacobian = LogJacobian(method="vmap", chunk_size=128)
optimizer = SR(
    learning_rate=0.05,
    regularization=1e-3,
    jacobian=jacobian,
)
state = VariationalState(system, model, sampler, seed=7)

result = GroundStateDriver(state, optimizer).run(steps=500)
```

Use `ExactSampler()` instead of `MetropolisSampler(...)` for full summation, or
`Adam(learning_rate=...)` instead of `SR(...)` for ordinary gradient updates.
`LogJacobian` is an SR-only dependency and is not evaluated by Adam. Use
`method="sequential"` for the reference implementation, `method="vmap"` for
batched VJPs, or `method="auto"` to prefer a model-provided analytic Jacobian
and otherwise use `vmap`.
`MetropolisSampler` returns `num_chains * sweeps` configurations. The
`sweep_size` setting controls the number of single-spin proposals per chain
between retained samples; `None` means `num_sites` proposals.
The complete editable example is
`examples/ground_state/flexible_ground_state.py`.
For a noise-free Adam run that enumerates the complete Hilbert space, use
`examples/ground_state/adam_full_sum_ground_state.py`.

## Projected real-time evolution

`examples/time_evolution/projected_time_evolution.py` first prepares an NQS
ground state with MC+Adam, uses that trained wavefunction as the `t=0` state,
and then performs full-sum short-time projection back onto an NQS after every
time step. It also reports fidelity against dense exact evolution for small
systems.

```bash
python examples/time_evolution/projected_time_evolution.py
```

## Numerical conventions

- Spin configurations are integer tensors with values `+1` and `-1`.
- Exact ordering is binary ascending with bit `0 -> +1`, preserving the legacy
  Python implementation.
- `log_psi(configurations) -> complex tensor` is the model contract.
- `tilted_field_ising` implements `J sum ZZ - hx sum X - hz sum Z`.
- General Pauli Hamiltonians store their coefficients explicitly; no hidden
  sign or spin-1/2 factors are applied.
- Dense exact methods reject systems larger than their configured safety limit.
Neural-network quantum states for ground-state optimization and real-time quantum dynamics
