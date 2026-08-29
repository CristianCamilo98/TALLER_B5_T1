"""Mechanical DDPM sampler in normalized feature space."""

from __future__ import annotations

import torch

from .diffusion import GaussianDiffusion


class DDPMSampler:
    """Expose the stable `sample(n_samples, seed)` generator contract."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        *,
        input_length: int = 65,
        input_channels: int = 3,
    ) -> None:
        self.diffusion = diffusion
        self.input_length = int(input_length)
        self.input_channels = int(input_channels)

    def sample(self, n_samples: int, seed: int) -> torch.Tensor:
        """Return `(n_samples, 65, 3)` samples in normalized space."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        parameter = next(self.diffusion.model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        generator = torch.Generator(device=device).manual_seed(int(seed))
        sample = torch.randn(
            (n_samples, self.input_length, self.input_channels),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        was_training = self.diffusion.model.training
        self.diffusion.model.eval()
        with torch.inference_mode():
            for step in range(self.diffusion.steps - 1, -1, -1):
                timesteps = torch.full(
                    (n_samples,), step, device=device, dtype=torch.long
                )
                sample = self.diffusion.p_sample(
                    sample, timesteps, generator=generator
                )
        self.diffusion.model.train(was_training)
        if not torch.isfinite(sample).all().item():
            raise RuntimeError("Sampler produced non-finite values")
        return sample.detach().cpu()
