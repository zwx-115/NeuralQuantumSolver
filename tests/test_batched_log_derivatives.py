import pytest
import torch

from neural_quantum_solver import (
    ComplexFNN,
    ComplexRBM,
    FullSumState,
    GroundStateDriver,
    LogJacobian,
    SR,
    log_derivative_matrix_vmap,
    tilted_field_ising,
)
from neural_quantum_solver.optimizers import log_derivative_matrix


CONFIGURATIONS = torch.tensor(
    [
        [1, 1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, 1],
        [1, -1, 1],
    ],
    dtype=torch.int8,
)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "model_factory",
    [
        lambda dtype: ComplexRBM(3, 4, dtype=dtype, seed=17),
        lambda dtype: ComplexFNN([3, 5, 1], dtype=dtype, seed=23),
    ],
)
def test_vmap_log_derivatives_match_sequential(model_factory, dtype):
    model = model_factory(dtype)
    expected = log_derivative_matrix(model, CONFIGURATIONS)
    actual = log_derivative_matrix_vmap(model, CONFIGURATIONS)

    tolerance = 2e-5 if dtype == torch.complex64 else 1e-12
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_vmap_log_derivatives_support_sample_chunks():
    model = ComplexRBM(3, 4, dtype=torch.complex128, seed=29)
    unchunked = log_derivative_matrix_vmap(model, CONFIGURATIONS)
    chunked = log_derivative_matrix_vmap(model, CONFIGURATIONS, chunk_size=2)

    assert torch.allclose(chunked, unchunked, atol=1e-12, rtol=1e-12)


def test_log_jacobian_component_selects_sequential_or_vmap():
    model = ComplexRBM(3, 4, dtype=torch.complex128, seed=37)
    sequential = LogJacobian(method="sequential")(model, CONFIGURATIONS)
    batched = LogJacobian(method="vmap", chunk_size=2)(model, CONFIGURATIONS)

    assert torch.allclose(batched, sequential, atol=1e-12, rtol=1e-12)


def test_sr_uses_injected_log_jacobian_strategy():
    class RecordingJacobian:
        def __init__(self):
            self.calls = 0
            self.implementation = LogJacobian(method="vmap", chunk_size=2)

        def __call__(self, model, configurations):
            self.calls += 1
            return self.implementation(model, configurations)

    strategy = RecordingJacobian()
    system = tilted_field_ising(3)
    model = ComplexRBM(3, 4, dtype=torch.complex128, seed=41)
    optimizer = SR(learning_rate=0.02, jacobian=strategy)
    driver = GroundStateDriver(FullSumState(system, model), optimizer)

    driver.advance()

    assert strategy.calls == 1


def test_sr_default_preserves_sequential_log_jacobian():
    optimizer = SR()
    assert isinstance(optimizer.jacobian, LogJacobian)
    assert optimizer.jacobian.method == "sequential"


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_vmap_log_derivatives_reject_invalid_chunk_size(chunk_size):
    model = ComplexRBM(3, 4, seed=31)
    with pytest.raises(ValueError, match="chunk_size"):
        log_derivative_matrix_vmap(
            model, CONFIGURATIONS, chunk_size=chunk_size
        )


def test_log_jacobian_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown log-Jacobian method"):
        LogJacobian(method="not-a-method")
