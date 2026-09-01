# MUSA compatibility log

This log records backend limitations observed during ground-state benchmark
runs. Workarounds must preserve the CUDA/MUSA computation path and must not
silently retry accelerator computation on CPU.

## 2026-09-01: `torch.linalg.pinv` fails

- Environment evidence: muDNN v3300.
- Input: regularized complex64 SR matrix on `musa`.
- Failure: `NOT_SUPPORTED`, internal unary `MAX`/`DOUBLE` path.
- Control tests: `torch.linalg.inv`, `torch.linalg.solve`,
  `torch.linalg.eigvalsh`, and `torch.linalg.svdvals` succeeded.
- Resolution: SR now solves the regularized linear system directly with
  `torch.linalg.solve` on the primary CUDA or MUSA GPU. No CPU fallback.

## 2026-09-01: complex matrix-vector (`mv`) fails

- Environment evidence: muDNN v3300.
- Failing expression: `x @ visible_bias` in `ComplexRBM`, with complex64
  operands.
- Failure: `SetMUTensorDType Unsupported tensor dtype: ComplexFloat`.
- Observation: the preceding complex matrix-matrix expression
  `x @ weight.mT` succeeded.
- Resolution: represent the vector as a one-column matrix and use the
  mathematically identical `mm` path on both CUDA and MUSA. The same change is
  applied to the QGT diagnostic product. No CPU fallback.
- Status: CUDA regression and MC+SR smoke tests pass; MUSA rerun required.
