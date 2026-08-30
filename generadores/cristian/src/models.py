from __future__ import annotations

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Flatten,
    LeakyReLU,
    Reshape,
)

from .data import N_CHANNELS, WINDOW_LENGTH


def build_generator(
    latent_dim: int,
    *,
    seq_len: int = WINDOW_LENGTH,
    n_channels: int = N_CHANNELS,
) -> Model:
    """Map noise z -> window [seq_len, n_channels]."""
    noise = Input(shape=(latent_dim,), name="generator_input")
    x = Dense(256)(noise)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(seq_len * n_channels)(x)
    x = Reshape((seq_len, n_channels))(x)
    return Model(noise, x, name="generator")


def build_critic(
    *,
    seq_len: int = WINDOW_LENGTH,
    n_channels: int = N_CHANNELS,
) -> Model:
    """Critic for WGAN: real-valued score, no sigmoid."""
    window = Input(shape=(seq_len, n_channels), name="critic_input")
    x = Conv1D(64, kernel_size=5, padding="same")(window)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Conv1D(128, kernel_size=5, strides=2, padding="same")(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Conv1D(256, kernel_size=5, strides=2, padding="same")(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    score = Dense(1)(x)
    return Model(window, score, name="critic")
