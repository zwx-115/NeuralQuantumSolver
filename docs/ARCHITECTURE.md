# Implemented architecture and migration boundary

This branch is a clean implementation, while `nqs-gs-time-python-origin` remains
the immutable behavioral reference.

## Current dependency flow

`SpinHalfHilbert + Graph -> PauliHamiltonian -> PhysicalSystem`

`NeuralQuantumState -> exact/Metropolis sampling -> estimator -> objective -> runner`

Exact diagonalization consumes only `PhysicalSystem`. It therefore works with
every system assembled from explicit Pauli terms, independently of the NQS.

## Implemented first milestone

- Legacy-compatible spin enumeration and explicit encoding conversion.
- Generic packed connected-state generation for Pauli Hamiltonians.
- Dense `complex128` exact diagonalization and exact real-time propagation.
- Ising, Heisenberg, XXZ, and J1-J2 system constructors.
- Unified complex `log_psi` contract with RBM, FNN, and lookup-table ansatzes.
- Differentiable full-Hilbert-space ground-state energy and Adam runner.
- Exact and Metropolis samplers with distinct result metadata.
- `VariationalState` composition of a model, physical system, and sampler.
- Sampler-independent `Adam` and dense `SR` optimization strategies.
- `GroundStateDriver` orchestration with callbacks, diagnostics, and checkpointing.
- Legacy/direct/normalized/gauge-fixed/log-domain overlap entry points and
  overlap diagnostics.
- QGT construction, regularized SR solve diagnostics, frozen evolution targets,
  and structured checkpoints.

## Deliberately not hidden behind the baseline

The stabilized overlap paths are separate functions. They do not replace
`legacy_ratio_loss`. The present `log_domain_ratio_loss` uses normalized dense
matrix products as the small-system stability reference; a sparse local-gate
complex-log-sum implementation belongs to the next evolution milestone.

The full-sum projected rollout example is a small-system numerical baseline.
Monte Carlo overlap projection, a reusable time-evolution driver, minSR,
persistent-chain/autoregressive samplers, Trotter block scheduling, CUDA
performance benchmarks, and legacy checkpoint adapters remain future migration
phases. Their absence does not affect exact diagonalization or the composable
exact/Monte-Carlo ground-state paths.
