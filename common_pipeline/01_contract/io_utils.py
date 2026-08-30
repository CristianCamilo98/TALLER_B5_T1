"""Small I/O helpers for the contract module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from constants import FEATURE_DIM, N_CHANNELS, WINDOW_LENGTH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flatten_window(window: np.ndarray) -> list[float]:
    if window.shape != (WINDOW_LENGTH, N_CHANNELS):
        raise ValueError(f"Expected window shape {(WINDOW_LENGTH, N_CHANNELS)}, got {window.shape}")
    return window.reshape(-1).astype(np.float64).tolist()


def reconstruct_tensor(features_flat: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(features_flat, dtype=np.float64)
    if array.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected flat length {FEATURE_DIM}, got {array.shape}")
    return array.reshape(WINDOW_LENGTH, N_CHANNELS)


def stack_features(frame: pd.DataFrame) -> np.ndarray:
    rows = [reconstruct_tensor(values) for values in frame["features_flat"]]
    return np.stack(rows, axis=0)


def stats_close(actual: tuple[float, ...] | list[float], expected: tuple[float, ...]) -> bool:
    return bool(np.allclose(np.asarray(actual, dtype=np.float64), expected, rtol=0.0, atol=1e-6))
