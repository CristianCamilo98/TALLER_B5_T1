"""DDPM forward process, epsilon objective, and reverse transition."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GaussianDiffusion(nn.Module):
    """Linear-schedule DDPM with an epsilon-prediction objective."""

    def __init__(
        self,
        model: nn.Module,
        *,
        steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if steps < 2:
            raise ValueError("steps must be at least 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1")
        self.model = model
        self.steps = int(steps)
        self.objective = "epsilon_prediction"

        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_reciprocal_alphas", torch.rsqrt(alphas))
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=0.0))

    def _validate_timesteps(self, timesteps: torch.Tensor, batch_size: int) -> None:
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape (B,)")
        if torch.is_floating_point(timesteps):
            raise TypeError("timesteps must use an integer dtype")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.steps):
            raise ValueError(f"timesteps must be between 0 and {self.steps - 1}")

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        selected = values.gather(0, timesteps.to(values.device))
        return selected.reshape((target.shape[0],) + (1,) * (target.ndim - 1)).to(target.dtype)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample `q(x_t | x_0)` and return both `x_t` and epsilon."""

        self._validate_timesteps(timesteps, x0.shape[0])
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError("noise must have the same shape as x0")
        if not torch.isfinite(x0).all().item() or not torch.isfinite(noise).all().item():
            raise ValueError("x0 and noise must be finite")
        coefficient_x0 = self._extract(self.sqrt_alpha_bars, timesteps, x0)
        coefficient_noise = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, x0)
        xt = coefficient_x0 * x0 + coefficient_noise * noise
        return xt, noise

    def training_loss(
        self,
        x0: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Compute the unweighted epsilon-prediction mean squared error."""

        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.steps,
                (x0.shape[0],),
                device=x0.device,
                generator=generator,
            )
        self._validate_timesteps(timesteps, x0.shape[0])
        if noise is None:
            noise = torch.randn(
                x0.shape,
                dtype=x0.dtype,
                device=x0.device,
                generator=generator,
            )
        xt, target_noise = self.q_sample(x0, timesteps, noise)
        predicted_noise = self.model(xt, timesteps)
        if predicted_noise.shape != target_noise.shape:
            raise ValueError("Model output shape does not match epsilon target")
        return F.mse_loss(predicted_noise, target_noise)

    def p_mean_variance(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the DDPM reverse mean and posterior variance."""

        self._validate_timesteps(timesteps, xt.shape[0])
        predicted_noise = self.model(xt, timesteps)
        beta_t = self._extract(self.betas, timesteps, xt)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars, timesteps, xt
        )
        mean = self._extract(self.sqrt_reciprocal_alphas, timesteps, xt) * (
            xt - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t
        )
        variance = self._extract(self.posterior_variance, timesteps, xt)
        return mean, variance

    def p_sample(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one reverse DDPM transition."""

        mean, variance = self.p_mean_variance(xt, timesteps)
        noise = torch.randn(
            xt.shape,
            dtype=xt.dtype,
            device=xt.device,
            generator=generator,
        )
        nonzero = (timesteps > 0).to(xt.dtype).reshape(
            (xt.shape[0],) + (1,) * (xt.ndim - 1)
        )
        return mean + nonzero * torch.sqrt(variance) * noise
