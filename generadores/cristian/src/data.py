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
GLOBAL_NORMALIZED_SPACE = "global_channel_normalized"
SOURCE_MODEL = "wgan_gp"


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


def load_nvda_hidden_windows(
    *,
    windows_dir: Path | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Ventanas NVDA en periodo hidden (2012 — 2022-H1), filtradas desde nvda_full_history."""
    config = load_experiment_config(config_path)
    hidden_end = pd.Timestamp(config["dates"]["target_hidden_end"])
    frame = load_donor_windows("nvda_full_history", windows_dir=windows_dir)
    frame = frame.copy()
    frame["window_end_date"] = pd.to_datetime(frame["window_end_date"])
    hidden = frame.loc[frame["window_end_date"] <= hidden_end].copy()
    hidden["split"] = "nvda_hidden"
    return hidden.sort_values(["ticker", "window_start_date"]).reset_index(drop=True)


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


def synthetic_seed_column(frame: pd.DataFrame) -> str:
    if "training_seed" in frame.columns:
        return "training_seed"
    if "seed" in frame.columns:
        return "seed"
    raise ValueError("El parquet sintético no tiene training_seed ni seed")


def synthetic_windows_to_contract_frame(
    windows: np.ndarray,
    *,
    training_seed: int,
    source_model: str = SOURCE_MODEL,
) -> pd.DataFrame:
    """Schema común (01_contract): mismas 8 columnas que generadores/daniel."""
    records: list[dict] = []
    for synthetic_id, window in enumerate(windows):
        flat = np.asarray(window, dtype=np.float32).reshape(-1)
        records.append(
            {
                "synthetic_id": synthetic_id,
                "source_model": source_model,
                "training_seed": training_seed,
                "space": GLOBAL_NORMALIZED_SPACE,
                "window_length": WINDOW_LENGTH,
                "n_channels": N_CHANNELS,
                "channel_order": list(CHANNELS),
                "features_flat": flat.tolist(),
            }
        )
    return pd.DataFrame.from_records(records)


def synthetic_windows_to_local_frame(
    windows: np.ndarray,
    *,
    seed: int,
    checkpoint: str,
    ratio: float | None = None,
) -> pd.DataFrame:
    """Export local desnormalizado (evaluación vs donors en escala original)."""
    records: list[dict] = []
    for idx, window in enumerate(windows):
        records.append(
            {
                "synthetic_id": idx,
                "seed": seed,
                "ratio": ratio,
                "split": "synthetic",
                "source_model": SOURCE_MODEL,
                "checkpoint": checkpoint,
                "features_flat": window.reshape(-1).tolist(),
            }
        )
    return pd.DataFrame.from_records(records)
