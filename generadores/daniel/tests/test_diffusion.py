import pytest
import torch

from generadores.daniel.src.diffusion import GaussianDiffusion
from generadores.daniel.src.network import TemporalDenoiser


@pytest.fixture
def diffusion() -> GaussianDiffusion:
    model = TemporalDenoiser(base_channels=8, time_embedding_dim=16)
    return GaussianDiffusion(model, steps=10, beta_start=0.0001, beta_end=0.02)


def test_linear_schedule_buffers_are_coherent(diffusion: GaussianDiffusion) -> None:
    assert diffusion.betas.shape == (10,)
    assert torch.all(diffusion.betas > 0)
    assert torch.all(diffusion.betas[1:] > diffusion.betas[:-1])
    assert torch.allclose(diffusion.alphas, 1.0 - diffusion.betas)
    assert torch.allclose(diffusion.alpha_bars, torch.cumprod(diffusion.alphas, dim=0))
    assert torch.all(diffusion.alpha_bars[1:] < diffusion.alpha_bars[:-1])
    assert torch.isfinite(diffusion.posterior_variance).all()


def test_q_sample_explicit_noise_is_exactly_reproducible(diffusion: GaussianDiffusion) -> None:
    x0 = torch.randn((4, 65, 3))
    timesteps = torch.tensor([0, 1, 5, 9], dtype=torch.long)
    noise = torch.randn_like(x0)
    first, returned_noise = diffusion.q_sample(x0, timesteps, noise)
    second, _ = diffusion.q_sample(x0, timesteps, noise)
    assert first.shape == noise.shape == x0.shape
    assert torch.equal(returned_noise, noise)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_q_sample_rejects_invalid_timestep(diffusion: GaussianDiffusion) -> None:
    with pytest.raises(ValueError, match="between"):
        diffusion.q_sample(torch.randn((1, 65, 3)), torch.tensor([10]))


def test_epsilon_prediction_loss_is_finite_and_differentiable(
    diffusion: GaussianDiffusion,
) -> None:
    x0 = torch.randn((3, 65, 3))
    loss = diffusion.training_loss(x0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in diffusion.model.parameters())
