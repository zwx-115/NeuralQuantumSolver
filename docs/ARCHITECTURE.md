# Ground-state-only architecture

This branch contains only exact-diagonalization and neural quantum-state
ground-state functionality. Real-time evolution and overlap-projection modules
remain on the `code-v1` branch.

## Dependency flow

`SpinHalfHilbert + Graph -> PauliHamiltonian -> PhysicalSystem`

`NeuralQuantumState -> Exact/Metropolis sampler -> energy estimator -> Adam/SR -> GroundStateDriver`

Exact diagonalization consumes only `PhysicalSystem` and remains a CPU
`complex128` reference by default.

## Accelerator model

`DeviceMesh` expands one explicit device and `num_gpus` into a homogeneous
single-host device group, for example `cuda:0..3` or `musa:0..3`. The primary
model lives on the first device. Replica models independently perform sampling,
local-energy evaluation, and differentiation. Gradients (Adam) or sufficient
statistics (SR) are reduced onto the primary model.

No NCCL or MCCL dependency is required. CUDA and MUSA execute the same Python
and PyTorch code path. The regularized dense SR system is aggregated and solved
with `torch.linalg.solve` on the primary GPU. `solver_device="cpu"` remains
available as an explicit diagnostic or compatibility choice.

The training and NQS evaluation path never catches an accelerator failure or
automatically retries an operation on CPU. Unsupported CUDA/MUSA operators fail
at their original call site. CPU copies are limited to best-state/checkpoint
storage, while dense exact diagonalization is an explicitly separate reference.
Multi-GPU exact sampling also enumerates integer spin labels on the host before
distributing shards; this is deterministic data preparation, not an operator
fallback, and is identical for CUDA and MUSA.

## Supported composition

- Models: `ComplexRBM`, `ComplexFNN`, `LogAmplitudeTable`, or any
  `NeuralQuantumState` implementing batched `log_psi`.
- Sampling: exact Hilbert-space summation or Metropolis chains.
- Optimization: Adam or dense stochastic reconfiguration.
- Systems: tilted-field Ising, Heisenberg, XXZ, J1-J2, and custom Pauli sums.
- Devices: CPU, NVIDIA CUDA, and Moore Threads MUSA.

Checkpoints store model tensors on CPU so a run can move between accelerator
backends.

## Benchmark timing

`GroundStateDriver` synchronizes every selected accelerator around sampling,
local-energy evaluation, optimization, and complete-step timing boundaries.
SR additionally synchronizes around log-Jacobian construction, QGT/force
construction, and the primary-device dense solve. Per-step records are appended
to CSV immediately, while static environment/configuration and final summaries
are stored in JSON.
