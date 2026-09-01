from copy import deepcopy

import torch

from neural_quantum_solver import Adam, ComplexRBM, ExactSampler, LogJacobian, SR
from neural_quantum_solver.estimators import sampled_energy, sharded_sampled_energy
from neural_quantum_solver.samplers import SampleBatch
from neural_quantum_solver.systems import tilted_field_ising


def _model(seed=71):
    return ComplexRBM(3, 5, dtype=torch.complex128, seed=seed)


def _exact_shards(model, system):
    sample = ExactSampler().sample(model, system)
    configurations = torch.tensor_split(sample.configurations, 2)
    weights = torch.tensor_split(sample.weights, 2)
    return tuple(
        SampleBatch(configs, shard_weights, None, True)
        for configs, shard_weights in zip(configurations, weights)
    )


def test_sharded_exact_statistics_match_single_batch():
    system = tilted_field_ising(3, field_x=0.7, field_z=0.2)
    primary = _model()
    replica = deepcopy(primary)
    full_sample = ExactSampler().sample(primary, system)

    single = sampled_energy(primary, system, full_sample)
    sharded = sharded_sampled_energy(
        (primary, replica), system, _exact_shards(primary, system)
    )

    assert sharded.num_samples == system.hilbert.size
    assert torch.allclose(sharded.energy, single.energy, atol=1e-12, rtol=1e-12)
    assert torch.allclose(sharded.variance, single.variance, atol=1e-12, rtol=1e-12)


def test_sharded_adam_update_matches_single_batch():
    system = tilted_field_ising(3)
    single_model = _model(seed=73)
    primary = deepcopy(single_model)
    replica = deepcopy(single_model)

    single_stats = sampled_energy(
        single_model, system, ExactSampler().sample(single_model, system)
    )
    sharded_stats = sharded_sampled_energy(
        (primary, replica), system, _exact_shards(primary, system)
    )

    Adam(learning_rate=0.003).step(single_model, single_stats)
    step = Adam(learning_rate=0.003).step(primary, sharded_stats)

    assert step.metrics["num_devices"] == 2
    for actual, expected in zip(primary.parameters(), single_model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_sharded_sr_update_matches_single_batch():
    system = tilted_field_ising(3)
    single_model = _model(seed=79)
    primary = deepcopy(single_model)
    replica = deepcopy(single_model)

    single_stats = sampled_energy(
        single_model, system, ExactSampler().sample(single_model, system)
    )
    sharded_stats = sharded_sampled_energy(
        (primary, replica), system, _exact_shards(primary, system)
    )
    settings = dict(
        learning_rate=0.02,
        regularization=1e-3,
        jacobian=LogJacobian(method="sequential"),
        solver_device="cpu",
    )

    SR(**settings).step(single_model, single_stats)
    step = SR(**settings).step(primary, sharded_stats)

    assert step.metrics["num_devices"] == 2
    for actual, expected in zip(primary.parameters(), single_model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
