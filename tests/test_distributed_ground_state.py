import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neural_quantum_solver import (
    AmplitudePhaseRBM,
    GroundStateDriver,
    LogJacobian,
    MetropolisSampler,
    ParallelContext,
    SR,
    VariationalState,
    tilted_field_ising,
)


def _distributed_worker(
    rank: int, world_size: int, init_file: str, output_dir: str
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        context = ParallelContext.from_initialized("cpu", local_rank=rank)
        system = tilted_field_ising(
            num_sites=4,
            coupling=1.0,
            field_x=0.5,
            field_z=0.5,
        )
        model = AmplitudePhaseRBM(
            4, 8, dtype=torch.float64, device="cpu", seed=7
        )
        state = VariationalState(
            system,
            model,
            MetropolisSampler(
                num_chains=8,
                thermal_sweeps=1,
                sweeps=1,
            ),
            seed=11,
            num_gpus=world_size,
            parallel_context=context,
        )
        optimizer = SR(
            learning_rate=0.02,
            jacobian=LogJacobian(method="auto"),
            solver_device="cpu",
        )
        output = Path(output_dir)
        result = GroundStateDriver(state, optimizer).run(
            steps=2,
            report_every=None,
            history_path=output / f"rank{rank}_steps.csv",
            metadata_path=output / f"rank{rank}_metadata.json",
            experiment_label="gloo_test",
        )
        assert result.history[-1].num_samples == 8
        flat = torch.cat(
            [parameter.detach().reshape(-1) for parameter in model.parameters()]
        )
        gathered = [torch.empty_like(flat) for _ in range(world_size)]
        dist.all_gather(gathered, flat)
        for replica in gathered[1:]:
            torch.testing.assert_close(gathered[0], replica)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo distributed backend is unavailable",
)
def test_two_process_driver_uses_rank_zero_io(tmp_path):
    world_size = 2
    mp.spawn(
        _distributed_worker,
        args=(world_size, str(tmp_path / "init"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    assert (tmp_path / "rank0_steps.csv").exists()
    assert (tmp_path / "rank0_metadata.json").exists()
    assert not (tmp_path / "rank1_steps.csv").exists()
    assert not (tmp_path / "rank1_metadata.json").exists()
    metadata = json.loads(
        (tmp_path / "rank0_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["num_gpus"] == 2
    assert metadata["communication_backend"] == "gloo"
    assert metadata["completed_steps"] == 2
