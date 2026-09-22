import torch

from neural_quantum_solver import ComplexRBM, LogAmplitudeTable, tilted_field_ising
from neural_quantum_solver.diagnostics import (
    OverlapSRSettings, evolve_exact_snapshots, fit_snapshot, fit_snapshot_sr,
    qgt_spectrum, tangent_space_residual,
)
from neural_quantum_solver.experiments import (
    SnapshotFittingExperiment, SnapshotFittingSettings, TrajectoryPoint,
    diagnose_trajectory, save_diagnostic_artifacts,
)


def test_qgt_spectrum_separates_numerical_and_participation_ranks():
    system = tilted_field_ising(3)
    model = ComplexRBM(3, 3, seed=4)
    result = qgt_spectrum(model, system, rtol=1e-10)
    assert result.eigenvalues.shape == (15,)
    assert 0 <= result.numerical_rank <= result.num_parameters
    assert 0 <= result.effective_rank <= result.num_parameters
    assert torch.allclose(result.effective_rank_fraction, result.effective_rank / 15)


def test_tangent_residual_matches_explicit_least_squares():
    system = tilted_field_ising(3)
    model = ComplexRBM(3, 3, seed=3)
    result = tangent_space_residual(model, system, rtol=1e-10)
    assert result.defined
    assert 0 <= result.normalized_residual <= 1 + 1e-10
    assert torch.allclose(
        result.normalized_residual, result.least_squares_residual, atol=1e-10, rtol=1e-9
    )


def test_trajectory_diagnostics_returns_scalar_rows_and_spectra():
    system = tilted_field_ising(2)
    source = ComplexRBM(2, 2, seed=6)
    rows, spectra = diagnose_trajectory(system, [TrajectoryPoint(0.5, source)])
    assert rows[0]["time"] == 0.5
    assert rows[0]["num_parameters"] == 8
    assert spectra["t0.5"]["eigenvalues"].shape == (8,)


def test_snapshot_factory_receives_its_target_time():
    system = tilted_field_ising(2)
    initial = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
    snapshots = evolve_exact_snapshots(system, initial, [0.0, 0.5])
    seen = []

    def factory(seed, snapshot):
        seen.append((seed, snapshot.time))
        return LogAmplitudeTable(2)

    experiment = SnapshotFittingExperiment(
        system, snapshots, families={"trajectory": factory},
        settings=SnapshotFittingSettings(max_steps=1, seeds=(7,)),
        family_seeds={"trajectory": (7,)},
    )
    rows, _ = experiment.run()
    assert [row["time"] for row in rows] == [0.0, 0.5]
    assert seen == [(7, 0.0), (7, 0.5)]


def test_overlap_sr_lowers_a_frozen_snapshot_loss_without_mutating_source():
    system = tilted_field_ising(2, field_x=0.5, field_z=0.2)
    source = ComplexRBM(2, 2, seed=9)
    before = {key: value.detach().clone() for key, value in source.state_dict().items()}
    initial = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
    snapshot = evolve_exact_snapshots(system, initial, [0.2])[0]
    result = fit_snapshot_sr(
        source,
        system,
        snapshot,
        settings=OverlapSRSettings(learning_rate=0.05, regularization=1e-3, jacobian_chunk_size=8),
        max_steps=3,
    )
    assert result.best_loss < result.initial_loss
    assert all(torch.equal(before[key], value) for key, value in source.state_dict().items())


def test_snapshots_fitting_and_artifact_output_do_not_mutate_source(tmp_path):
    system = tilted_field_ising(2)
    model = LogAmplitudeTable(2)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
    snapshot = evolve_exact_snapshots(system, initial, [0.0])[0]
    result = fit_snapshot(
        model, system, snapshot,
        optimizer_factory=lambda parameters: torch.optim.Adam(parameters, lr=0.05),
        max_steps=3,
    )
    assert result.iterations >= 1
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    destination = save_diagnostic_artifacts(
        tmp_path / "run", config={"seed": 1}, rows=[{"best_loss": result.best_loss}],
        summary={"best_loss": result.best_loss}, checkpoints={"best": result.best_state_dict},
    )
    assert (destination / "config.json").is_file()
    assert (destination / "results.csv").is_file()
    assert (destination / "summary.json").is_file()
    assert (destination / "checkpoints.pt").is_file()
