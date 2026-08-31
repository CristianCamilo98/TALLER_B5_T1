"""Bootstrap + Gaussian jitter baseline in normalized space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    BASELINE_NOISE_SCALE,
    BASELINE_OUTPUT_PATH,
    BASELINE_SEED,
    BASELINE_SOURCE_MODEL,
    CANONICAL_MEAN,
    CANONICAL_NORMALIZER_SHA256,
    CANONICAL_STD,
    CHANNEL_ORDER,
    DONOR_TRAIN_PATH,
    DONOR_TRAIN_SHA256,
    EXPECTED_ROWS,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from io_utils import flatten_window, sha256_file, write_json
from normalizer import load_donor_train_normalized


@dataclass(frozen=True)
class BaselineResult:
    path: Path
    sha256: str
    shape: tuple[int, int, int]
    seed: int
    noise_scale: float
    donor_train_path: Path
    donor_train_sha256: str


def generate_baseline_windows(
    donor_normalized: np.ndarray,
    *,
    seed: int,
    noise_scale: float,
    n_windows: int,
) -> np.ndarray:
    """Bootstrap complete windows and add deterministic Gaussian jitter."""

    donor = np.asarray(donor_normalized, dtype=np.float32)
    if donor.ndim != 3 or donor.shape[1:] != (WINDOW_LENGTH, N_CHANNELS):
        raise ValueError("donor_normalized must have shape (N, 65, 3)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, donor.shape[0], size=n_windows)
    bootstrap = donor[indices].astype(np.float32, copy=True)
    noise = rng.normal(0.0, noise_scale, size=bootstrap.shape).astype(np.float32)
    return bootstrap + noise


def build_baseline(
    *,
    output_path: Path = BASELINE_OUTPUT_PATH,
    seed: int = BASELINE_SEED,
    noise_scale: float = BASELINE_NOISE_SCALE,
    n_windows: int = EXPECTED_ROWS,
) -> BaselineResult:
    if not DONOR_TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Missing canonical donor_train: {DONOR_TRAIN_PATH}")

    donor_sha = sha256_file(DONOR_TRAIN_PATH)
    if donor_sha != DONOR_TRAIN_SHA256:
        raise RuntimeError(
            "CANONICAL DONOR HASH MISMATCH: "
            f"calculated={donor_sha}, expected={DONOR_TRAIN_SHA256}"
        )
    donor_normalized = load_donor_train_normalized(DONOR_TRAIN_PATH)
    synthetic = generate_baseline_windows(
        donor_normalized,
        seed=seed,
        noise_scale=noise_scale,
        n_windows=n_windows,
    )
    reproduced = generate_baseline_windows(
        donor_normalized,
        seed=seed,
        noise_scale=noise_scale,
        n_windows=n_windows,
    )
    if not np.array_equal(synthetic, reproduced):
        raise RuntimeError("Baseline reproducibility check failed")

    records = []
    for synthetic_id, window in enumerate(synthetic):
        records.append(
            {
                "synthetic_id": synthetic_id,
                "source_model": BASELINE_SOURCE_MODEL,
                "training_seed": seed,
                "space": GLOBAL_NORMALIZED_SPACE,
                "window_length": WINDOW_LENGTH,
                "n_channels": N_CHANNELS,
                "channel_order": list(CHANNEL_ORDER),
                "features_flat": flatten_window(window),
            }
        )

    frame = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    provenance_path = output_path.with_suffix(".provenance.json")
    output_sha = sha256_file(output_path)
    write_json(
        provenance_path,
        {
            "source_model": BASELINE_SOURCE_MODEL,
            "algorithm": "bootstrap_resample + gaussian_jitter",
            "seed": seed,
            "noise_scale": noise_scale,
            "donor_train_path": "data/features/windows/donor_train.parquet",
            "donor_train_sha256": donor_sha,
            "n_windows": n_windows,
            "logical_shape": [n_windows, WINDOW_LENGTH, N_CHANNELS],
            "channel_order": list(CHANNEL_ORDER),
            "space": GLOBAL_NORMALIZED_SPACE,
            "normalization": {
                "type": "global_channel_zscore",
                "fit_split": "donor_train",
                "fit_axes": [0, 1],
                "fit_dtype": "float64",
                "output_dtype": "float32",
                "ddof": 0,
                "std_threshold": 1e-8,
                "mean": list(CANONICAL_MEAN),
                "std": list(CANONICAL_STD),
                "normalizer_sha256": CANONICAL_NORMALIZER_SHA256,
            },
            "parquet_sha256": output_sha,
            "reproducibility_pass": True,
        },
    )

    return BaselineResult(
        path=output_path,
        sha256=output_sha,
        shape=(n_windows, WINDOW_LENGTH, N_CHANNELS),
        seed=seed,
        noise_scale=noise_scale,
        donor_train_path=DONOR_TRAIN_PATH,
        donor_train_sha256=donor_sha,
    )
