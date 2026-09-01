import torch
from pathlib import Path

from neural_quantum_solver import ComplexRBM
from neural_quantum_solver.runners import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip():
    model = ComplexRBM(3, 5, seed=9)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = Path("checkpoint-test.pt")
    try:
        save_checkpoint(
            path, model,
            model_config={"kind": "rbm", "num_visible": 3, "num_hidden": 5},
            run_config={"seed": 9}, step=12,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        payload = load_checkpoint(path, model)
        assert payload["step"] == 12
        for key, value in model.state_dict().items():
            assert torch.equal(value, expected[key])
    finally:
        path.unlink(missing_ok=True)
