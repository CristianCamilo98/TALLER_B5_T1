from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

from .data import N_CHANNELS, WINDOW_LENGTH


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_loss_history(history: list[dict], path: Path) -> None:
    frame = pd.DataFrame(history)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_checkpoint(
    generator: Model,
    critic: Model,
    run_dir: Path,
    *,
    epoch: int,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    generator_path = run_dir / f"generator_epoch_{epoch:05d}.keras"
    critic_path = run_dir / f"critic_epoch_{epoch:05d}.keras"
    generator.save(generator_path)
    critic.save(critic_path)
    return generator_path


def load_generator(path: Path) -> Model:
    return tf.keras.models.load_model(path)


@dataclass
class RunMetadata:
    model: str
    seed: int
    started_at: str
    epochs: int
    batch_size: int
    latent_dim: int
    n_critic: int
    lambda_gp: float
    learning_rate: float
    n_train_windows: int
    checkpoint: str | None = None

    @classmethod
    def now(
        cls,
        *,
        seed: int,
        epochs: int,
        batch_size: int,
        latent_dim: int,
        n_critic: int,
        lambda_gp: float,
        learning_rate: float,
        n_train_windows: int,
    ) -> RunMetadata:
        return cls(
            model="wgan_gp",
            seed=seed,
            started_at=datetime.now(timezone.utc).isoformat(),
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            n_critic=n_critic,
            lambda_gp=lambda_gp,
            learning_rate=learning_rate,
            n_train_windows=n_train_windows,
        )


def save_run_metadata(metadata: RunMetadata, path: Path) -> None:
    save_json(asdict(metadata), path)


def save_synthetic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def save_synthetic_outputs(frame: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Guarda ventanas sintéticas en parquet y CSV (mismo schema)."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    csv_path = parquet_path.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def flatten_generated(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_LENGTH, N_CHANNELS):
        raise ValueError(f"Expected (N, {WINDOW_LENGTH}, {N_CHANNELS}), got {windows.shape}")
    return windows
