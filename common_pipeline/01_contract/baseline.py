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
    donor_normalized = load_donor_train_normalized(DONOR_TRAIN_PATH)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, donor_normalized.shape[0], size=n_windows)
    bootstrap = donor_normalized[indices].astype(np.float32, copy=True)
    noise = rng.normal(0.0, noise_scale, size=bootstrap.shape).astype(np.float32)
    synthetic = bootstrap + noise

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
        },
    )

    return BaselineResult(
        path=output_path,
        sha256=sha256_file(output_path),
        shape=(n_windows, WINDOW_LENGTH, N_CHANNELS),
        seed=seed,
        noise_scale=noise_scale,
        donor_train_path=DONOR_TRAIN_PATH,
        donor_train_sha256=donor_sha,
    )
