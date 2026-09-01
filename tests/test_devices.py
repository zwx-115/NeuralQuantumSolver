import pytest
import torch

from neural_quantum_solver import DeviceMesh, expand_device_names


def test_device_names_expand_from_an_explicit_first_card():
    assert expand_device_names("cuda:1", 3) == (
        "cuda:1", "cuda:2", "cuda:3"
    )
    assert expand_device_names("musa", 2) == ("musa:0", "musa:1")


def test_cpu_mesh_is_single_device():
    mesh = DeviceMesh.create("cpu", 1)
    assert mesh.devices == (torch.device("cpu"),)
    assert mesh.primary.type == "cpu"
    assert mesh.num_devices == 1


@pytest.mark.parametrize(
    "device,num_gpus,message",
    [
        ("gpu", 1, "cpu, cuda, or musa"),
        ("cpu", 2, "CPU execution"),
        ("cuda:x", 1, "invalid device index"),
        ("cuda:0", 0, "positive integer"),
    ],
)
def test_invalid_device_mesh_settings_are_rejected(device, num_gpus, message):
    with pytest.raises(ValueError, match=message):
        expand_device_names(device, num_gpus)
