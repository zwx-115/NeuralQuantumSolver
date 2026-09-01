from copy import deepcopy

import torch

from neural_quantum_solver import (
    Adam,
    AmplitudePhaseFNN,
    AmplitudePhaseRBM,
    ExactSampler,
    GroundStateDriver,
    LogJacobian,
    SR,
    VariationalState,
    tilted_field_ising,
)
from neural_quantum_solver.derivatives import RealLogDerivativeParts
from neural_quantum_solver.qgt import (
    quantum_geometric_tensor,
    real_quantum_geometric_tensor,
)
from neural_quantum_solver.real_estimators import (
    RealPairSampledEnergy,
    real_pair_exact_energy,
    real_pair_sampled_energy,
    real_pair_sharded_sampled_energy,
)
from neural_quantum_solver.samplers import SampleBatch
from neural_quantum_solver.systems import PhysicalSystem
from neural_quantum_solver.hilbert import SpinHalfHilbert
from neural_quantum_solver.operators import PauliHamiltonian, PauliTerm


def _model(seed=101):
    return AmplitudePhaseRBM(3, 4, dtype=torch.float64, seed=seed)


def _dense_energy(model, system):
    configurations = system.hilbert.all_states()
    parts = model.log_psi_parts(configurations)
    amplitudes = torch.exp(parts.log_amplitude) * torch.complex(
        torch.cos(parts.phase), torch.sin(parts.phase)
    )
    amplitudes = amplitudes / torch.linalg.vector_norm(amplitudes)
    matrix = system.hamiltonian.dense_matrix(
        system.hilbert, dtype=torch.complex128
    )
    return torch.vdot(amplitudes, matrix @ amplitudes).real


def test_real_pair_models_use_only_real_parameters_and_outputs():
    configurations = torch.tensor([[1, -1, 1], [-1, 1, -1]], dtype=torch.int8)
    for model in (
        _model(),
        AmplitudePhaseFNN([3, 5, 1], dtype=torch.float64, seed=103),
    ):
        parts = model.log_psi_parts(configurations)
        assert all(not parameter.is_complex() for parameter in model.parameters())
        assert not parts.log_amplitude.is_complex()
        assert not parts.phase.is_complex()
        assert parts.log_amplitude.shape == (2,)
        assert parts.phase.shape == (2,)


def test_real_connections_match_complex_connections_with_pauli_y():
    hamiltonian = PauliHamiltonian(
        2,
        (
            PauliTerm(0.7 + 0.2j, ((0, "Y"), (1, "Z"))),
            PauliTerm(-0.3j, ((1, "X"),)),
        ),
    )
    configurations = SpinHalfHilbert(2).all_states()
    complex_connections = hamiltonian.connections(
        configurations, dtype=torch.complex128
    )
    real_connections = hamiltonian.real_connections(
        configurations, dtype=torch.float64
    )
    assert torch.equal(real_connections.states, complex_connections.states)
    assert torch.equal(
        real_connections.sample_indices, complex_connections.sample_indices
    )
    assert torch.allclose(
        real_connections.matrix_elements_real,
        complex_connections.matrix_elements.real,
    )
    assert torch.allclose(
        real_connections.matrix_elements_imag,
        complex_connections.matrix_elements.imag,
    )


def test_real_pair_exact_energy_matches_dense_complex_reference():
    system = tilted_field_ising(3, field_x=0.7, field_z=0.2)
    model = _model(seed=107)
    estimate = real_pair_exact_energy(model, system)
    reference = _dense_energy(model, system)
    assert torch.allclose(estimate.energy, reference, atol=1e-12, rtol=1e-12)
    assert not estimate.local_energy_real.is_complex()
    assert not estimate.local_energy_imag.is_complex()


def test_real_pair_adam_score_gradient_matches_dense_energy_gradient():
    system = tilted_field_ising(3)
    direct_model = _model(seed=109)
    score_model = deepcopy(direct_model)
    _dense_energy(direct_model, system).backward()
    direct = torch.cat(
        [parameter.grad.reshape(-1) for parameter in direct_model.parameters()]
    )
    sample = ExactSampler().sample(score_model, system)
    statistics = real_pair_sampled_energy(score_model, system, sample)
    Adam._backward_real_shard(score_model, statistics)
    score = torch.cat(
        [parameter.grad.reshape(-1) for parameter in score_model.parameters()]
    )
    assert torch.allclose(score, direct, atol=1e-11, rtol=1e-11)


def test_real_pair_jacobian_methods_and_qgt_match_complex_definition():
    model = _model(seed=113)
    configurations = SpinHalfHilbert(3).all_states()
    sequential = LogJacobian("sequential")(model, configurations)
    batched = LogJacobian("vmap", chunk_size=3)(model, configurations)
    analytic = LogJacobian("analytic")(model, configurations)
    assert isinstance(sequential, RealLogDerivativeParts)
    for candidate in (batched, analytic):
        assert torch.allclose(
            candidate.log_amplitude, sequential.log_amplitude,
            atol=1e-12, rtol=1e-12,
        )
        assert torch.allclose(
            candidate.phase, sequential.phase, atol=1e-12, rtol=1e-12
        )
    probabilities = torch.softmax(torch.randn(8, dtype=torch.float64), dim=0)
    real_qgt = real_quantum_geometric_tensor(
        analytic.log_amplitude, analytic.phase, probabilities
    )
    complex_derivatives = torch.complex(
        analytic.log_amplitude, analytic.phase
    )
    reference = quantum_geometric_tensor(complex_derivatives, probabilities).real
    assert torch.allclose(real_qgt, reference, atol=1e-12, rtol=1e-12)


def test_real_pair_adam_sr_and_sharded_statistics():
    system = tilted_field_ising(3)
    for optimizer in (
        Adam(0.005),
        SR(0.01, jacobian=LogJacobian("analytic")),
    ):
        model = _model(seed=127)
        step = GroundStateDriver(
            VariationalState(system, model, ExactSampler()), optimizer
        ).advance()
        assert torch.isfinite(torch.tensor(step.energy))
        assert step.update_norm > 0

    primary = _model(seed=131)
    replica = deepcopy(primary)
    full_sample = ExactSampler().sample(primary, system)
    configuration_shards = torch.tensor_split(full_sample.configurations, 2)
    weight_shards = torch.tensor_split(full_sample.weights, 2)
    samples = tuple(
        SampleBatch(configurations, weights, None, True)
        for configurations, weights in zip(configuration_shards, weight_shards)
    )
    single = real_pair_sampled_energy(primary, system, full_sample)
    sharded = real_pair_sharded_sampled_energy(
        (primary, replica), system, samples
    )
    assert isinstance(single, RealPairSampledEnergy)
    assert torch.allclose(sharded.energy, single.energy, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        sharded.energy_imag, single.energy_imag, atol=1e-12, rtol=1e-12
    )
    assert torch.allclose(
        sharded.variance, single.variance, atol=1e-12, rtol=1e-12
    )


def test_real_pair_training_path_never_constructs_complex_tensor(monkeypatch):
    def reject_complex(*args, **kwargs):
        raise AssertionError("real-pair training attempted to construct a complex tensor")

    monkeypatch.setattr(torch, "complex", reject_complex)
    system = tilted_field_ising(3)
    model = _model(seed=137)
    step = GroundStateDriver(
        VariationalState(system, model, ExactSampler()),
        SR(0.01, jacobian=LogJacobian("analytic")),
    ).advance()
    assert step.update_norm > 0


def test_real_pair_sharded_adam_and_sr_match_single_batch_updates():
    system = tilted_field_ising(3)
    for make_optimizer in (
        lambda: Adam(0.003),
        lambda: SR(0.01, jacobian=LogJacobian("analytic")),
    ):
        single_model = _model(seed=139)
        primary = deepcopy(single_model)
        replica = deepcopy(single_model)
        full_sample = ExactSampler().sample(single_model, system)
        single_statistics = real_pair_sampled_energy(
            single_model, system, full_sample
        )
        configuration_shards = torch.tensor_split(
            full_sample.configurations, 2
        )
        weight_shards = torch.tensor_split(full_sample.weights, 2)
        samples = tuple(
            SampleBatch(configurations, weights, None, True)
            for configurations, weights in zip(
                configuration_shards, weight_shards
            )
        )
        sharded_statistics = real_pair_sharded_sampled_energy(
            (primary, replica), system, samples
        )

        make_optimizer().step(single_model, single_statistics)
        step = make_optimizer().step(primary, sharded_statistics)

        assert step.metrics["num_devices"] == 2
        for actual, expected in zip(
            primary.parameters(), single_model.parameters()
        ):
            assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
