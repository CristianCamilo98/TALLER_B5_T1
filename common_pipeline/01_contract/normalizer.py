"""Canonical global channel normalizer and provenance checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    CANONICAL_MEAN,
    CANONICAL_NORMALIZER_SHA256,
    CANONICAL_STD,
    CHANNEL_ORDER,
    DONOR_TRAIN_PATH,
    DONOR_TRAIN_ROWS,
    DONOR_TRAIN_SHAPE,
    N_CHANNELS,
    REPO_ROOT,
    STATS_ATOL,
    WINDOW_LENGTH,
)
from io_utils import reconstruct_tensor, sha256_file, stats_close


@dataclass(frozen=True)
class NormalizerState:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, windows: np.ndarray) -> np.ndarray:
        normalized = (windows.astype(np.float64) - self.mean) / self.std
        return normalized.astype(np.float32)

    def to_contract_dict(self) -> dict:
        return {
            "type": "global_channel_zscore",
            "scope": "all donor_train windows and sessions",
            "fit_split": "donor_train",
            "channels": list(CHANNEL_ORDER),
            "fit_axes": [0, 1],
            "ddof": 0,
            "fit_dtype": "float64",
            "output_dtype": "float32",
            "std_threshold": 1e-8,
            "zero_variance_policy": "replace sigma with 1.0 when sigma < threshold",
            "mean": self.mean.tolist(),
            "raw_std": self.std.tolist(),
            "std": self.std.tolist(),
        }


def fit_donor_train_normalizer(path: Path = DONOR_TRAIN_PATH) -> NormalizerState:
    frame = pd.read_parquet(path)
    if len(frame) != DONOR_TRAIN_ROWS:
        raise ValueError(f"donor_train rows={len(frame)}, expected {DONOR_TRAIN_ROWS}")
    windows = np.stack([reconstruct_tensor(values) for values in frame["features_flat"]], axis=0)
    if windows.shape != DONOR_TRAIN_SHAPE:
        raise ValueError(f"donor_train shape={windows.shape}, expected {DONOR_TRAIN_SHAPE}")
    mean = windows.mean(axis=(0, 1))
    raw_std = windows.std(axis=(0, 1), ddof=0)
    std = np.where(raw_std < 1e-8, 1.0, raw_std)
    return NormalizerState(mean=mean, std=std)


def canonical_normalizer() -> NormalizerState:
    return NormalizerState(
        mean=np.asarray(CANONICAL_MEAN, dtype=np.float64),
        std=np.asarray(CANONICAL_STD, dtype=np.float64),
    )


def load_donor_train_normalized(path: Path = DONOR_TRAIN_PATH) -> np.ndarray:
    normalizer = fit_donor_train_normalizer(path)
    if not stats_close(normalizer.mean.tolist(), CANONICAL_MEAN):
        raise ValueError("donor_train mean does not match canonical contract")
    if not stats_close(normalizer.std.tolist(), CANONICAL_STD):
        raise ValueError("donor_train std does not match canonical contract")
    frame = pd.read_parquet(path)
    windows = np.stack([reconstruct_tensor(values) for values in frame["features_flat"]], axis=0)
    return normalizer.normalize(windows)


def normalizer_file_hash(path: Path) -> str:
    return sha256_file(path)


def _json_stats(payload: dict) -> tuple[list[float] | None, list[float] | None]:
    normalization = payload.get("normalization", {})
    mean = payload.get("mean") or payload.get("scaler_mean") or normalization.get("mean")
    std = payload.get("std") or payload.get("scaler_std") or normalization.get("std")
    if mean is None or std is None:
        return None, None
    return list(mean), list(std)


def _output_provenance_files(output_path: Path | None) -> list[Path]:
    if output_path is None:
        return []
    candidates = (
        output_path.with_suffix(".provenance.json"),
        output_path.with_name(f"{output_path.stem}_manifest.json"),
        output_path.with_suffix(".json"),
    )
    return [path for path in candidates if path.is_file()]


def assess_normalization_provenance(
    generator_id: str,
    output_path: Path | None = None,
) -> str:
    """Return VERIFIED, NORMALIZATION_MISMATCH, or PROVENANCE_NOT_VERIFIABLE."""

    verified_hash = False
    stats_match = False
    stats_mismatch = False

    for path in _output_provenance_files(output_path):
        if normalizer_file_hash(path) == CANONICAL_NORMALIZER_SHA256:
            verified_hash = True
            break

        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            mean, std = _json_stats(payload)
            if mean is None or std is None:
                continue
            if stats_close(mean, CANONICAL_MEAN) and stats_close(std, CANONICAL_STD):
                stats_match = True
            else:
                stats_mismatch = True

        if path.suffix == ".npz":
            try:
                payload = np.load(path)
                mean = payload.get("mean")
                std = payload.get("std")
                if mean is not None and std is not None:
                    if stats_close(mean.tolist(), CANONICAL_MEAN) and stats_close(std.tolist(), CANONICAL_STD):
                        stats_match = True
                    else:
                        stats_mismatch = True
            except Exception:
                continue

    if verified_hash:
        return "NORMALIZATION_PROVENANCE_VERIFIED"
    if stats_mismatch:
        return "NORMALIZATION_MISMATCH"
    if stats_match:
        return "NORMALIZATION_NUMERICALLY_MATCHES"
    return "NORMALIZATION_PROVENANCE_NOT_VERIFIABLE"
