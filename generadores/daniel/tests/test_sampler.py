import pytest
import torch

from generadores.daniel.src.diffusion import GaussianDiffusion
from generadores.daniel.src.network import TemporalDenoiser
from generadores.daniel.src.sampler import DDPMSampler


@pytest.fixture
def sampler() -> DDPMSampler:
    torch.manual_seed(11)
    model = TemporalDenoiser(base_channels=8, time_embedding_dim=16)
    diffusion = GaussianDiffusion(model, steps=6)
    return DDPMSampler(diffusion)


def test_sampler_shape_finiteness_and_exact_count(sampler: DDPMSampler) -> None:
    samples = sampler.sample(3, seed=42)
    assert samples.shape == (3, 65, 3)
    assert samples.dtype == torch.float32
    assert torch.isfinite(samples).all()


def test_sampler_seed_contract(sampler: DDPMSampler) -> None:
    first = sampler.sample(2, seed=42)
    repeated = sampler.sample(2, seed=42)
    different = sampler.sample(2, seed=43)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_sampler_rejects_nonpositive_count(sampler: DDPMSampler) -> None:
    with pytest.raises(ValueError, match="positive"):
        sampler.sample(0, seed=42)
