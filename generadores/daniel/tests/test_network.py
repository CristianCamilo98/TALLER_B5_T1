import pytest
import torch

from generadores.daniel.src.network import SinusoidalTimeEmbedding, TemporalDenoiser


def test_timestep_embedding_shape_finiteness_and_determinism() -> None:
    embedding = SinusoidalTimeEmbedding(32)
    timesteps = torch.tensor([0, 1, 50, 99], dtype=torch.long)
    first = embedding(timesteps)
    second = embedding(timesteps)
    assert first.shape == (4, 32)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[-1])


@pytest.mark.parametrize("batch_size", [1, 4])
def test_temporal_denoiser_shape_finiteness_and_backward(batch_size: int) -> None:
    model = TemporalDenoiser(base_channels=8, time_embedding_dim=16)
    x = torch.randn((batch_size, 65, 3), requires_grad=True)
    timesteps = torch.arange(batch_size, dtype=torch.long) * 7
    output = model(x, timesteps)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_different_timesteps_condition_the_output() -> None:
    model = TemporalDenoiser(base_channels=8, time_embedding_dim=16).eval()
    x = torch.ones((2, 65, 3))
    output = model(x, torch.tensor([0, 99]))
    assert not torch.equal(output[0], output[1])
