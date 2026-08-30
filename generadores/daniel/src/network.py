"""Compact timestep-conditioned temporal denoiser."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep features followed by a small learnable MLP."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4:
            raise ValueError("embedding_dim must be at least 4")
        self.embedding_dim = int(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape (batch,), got {tuple(timesteps.shape)}")
        if torch.is_floating_point(timesteps):
            raise TypeError("timesteps must use an integer dtype")
        if torch.any(timesteps < 0):
            raise ValueError("timesteps cannot be negative")

        half = self.embedding_dim // 2
        denominator = max(half - 1, 1)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / denominator
        )
        arguments = timesteps.to(torch.float32)[:, None] * frequencies[None, :]
        embedding = torch.cat((arguments.sin(), arguments.cos()), dim=1)
        if embedding.shape[1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, self.embedding_dim - embedding.shape[1]))
        return self.mlp(embedding)


class TemporalResidualBlock(nn.Module):
    """Length-preserving Conv1D residual block with timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + residual


class TemporalDenoiser(nn.Module):
    """Predict epsilon while preserving the external `(B, 65, 3)` layout."""

    def __init__(
        self,
        *,
        input_length: int = 65,
        input_channels: int = 3,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if input_length <= 0 or input_channels <= 0 or base_channels <= 0:
            raise ValueError("Model dimensions must be positive")
        self.input_length = int(input_length)
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.time_embedding_dim = int(time_embedding_dim)

        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.input_conv = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(base_channels, base_channels, time_embedding_dim),
                TemporalResidualBlock(base_channels, base_channels * 2, time_embedding_dim),
                TemporalResidualBlock(base_channels * 2, base_channels * 2, time_embedding_dim),
                TemporalResidualBlock(base_channels * 2, base_channels, time_embedding_dim),
            ]
        )
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output_conv = nn.Conv1d(base_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        expected = (self.input_length, self.input_channels)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"x must have shape (B, {expected[0]}, {expected[1]})")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("timesteps must have shape (B,) aligned with x")
        if not torch.isfinite(x).all().item():
            raise ValueError("x contains NaN or infinite values")

        time_embedding = self.time_embedding(timesteps)
        hidden = self.input_conv(x.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        output = self.output_conv(F.silu(self.output_norm(hidden)))
        return output.transpose(1, 2)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
