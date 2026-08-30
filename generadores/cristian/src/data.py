from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .paths import default_experiment_config, default_windows_dir

CHANNELS = ["log_return", "log_high_low_range", "log1p_volume"]
WINDOW_LENGTH = 65
N_CHANNELS = len(CHANNELS)
FEATURE_DIM = WINDOW_LENGTH * N_CHANNELS


@dataclass(frozen=True)
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self.mean) / self.std

    def denormalize(self, windows: np.ndarray) -> np.ndarray:
        return windows * self.std + self.mean

    def to_dict(self) -> dict:
        return {
            "channels": CHANNELS,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ChannelNormalizer:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )


def load_experiment_config(config_path: Path | None = None) -> dict:
    path = config_path or default_experiment_config()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def windows_to_array(frame: pd.DataFrame) -> np.ndarray:
    stacked = np.stack(
        [np.asarray(values, dtype=np.float64).reshape(WINDOW_LENGTH, N_CHANNELS) for values in frame["features_flat"]],
        axis=0,
    )
    if stacked.shape[1:] != (WINDOW_LENGTH, N_CHANNELS):
        raise ValueError(f"Expected shape (N, {WINDOW_LENGTH}, {N_CHANNELS}), got {stacked.shape}")
    return stacked


def load_donor_windows(
    split: str,
    *,
    windows_dir: Path | None = None,
) -> pd.DataFrame:
    root = windows_dir or default_windows_dir()
    path = root / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    frame = pd.read_parquet(path)
    required = {"split", "ticker", "window_start_date", "window_end_date", "features_flat"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path.name}: {sorted(missing)}")
    return frame.sort_values(["ticker", "window_start_date"]).reset_index(drop=True)


def fit_normalizer(train_windows: np.ndarray) -> ChannelNormalizer:
    if train_windows.ndim != 3 or train_windows.shape[1:] != (WINDOW_LENGTH, N_CHANNELS):
        raise ValueError(f"train_windows debe ser (N, {WINDOW_LENGTH}, {N_CHANNELS})")
    mean = train_windows.mean(axis=(0, 1))
    std = train_windows.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    return ChannelNormalizer(mean=mean, std=std)


def load_normalizer(path: Path) -> ChannelNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ChannelNormalizer.from_dict(payload)


def save_normalizer(normalizer: ChannelNormalizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalizer.to_dict(), indent=2), encoding="utf-8")


def make_tf_dataset(
    windows: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(windows.astype(np.float32))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(windows), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def synthetic_windows_to_frame(
    windows: np.ndarray,
    *,
    seed: int,
    ratio: float | None = None,
    checkpoint: str,
) -> pd.DataFrame:
    records: list[dict] = []
    for idx, window in enumerate(windows):
        records.append(
            {
                "synthetic_id": idx,
                "seed": seed,
                "ratio": ratio,
                "split": "synthetic",
                "source_model": "wgan_gp",
                "checkpoint": checkpoint,
                "features_flat": window.reshape(-1).tolist(),
            }
        )
    return pd.DataFrame.from_records(records)
