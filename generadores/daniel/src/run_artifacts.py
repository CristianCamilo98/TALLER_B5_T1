"""Local schemas and writers for reproducible training-run artifacts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "learning_rate",
    "epoch_seconds",
)

REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "model",
        "seed",
        "validation_seed",
        "git_commit",
        "base_master_commit",
        "canonical_raw_sha256",
        "donor_train_sha256",
        "donor_validation_sha256",
        "train_count",
        "validation_count",
        "window_shape",
        "channels",
        "train_dates",
        "validation_dates",
        "normalization_type",
        "effective_config",
        "device",
        "python_version",
        "torch_version",
        "cuda_version",
        "start_time",
        "end_time",
        "epochs_completed",
        "best_epoch",
        "best_validation_loss",
        "stopping_reason",
        "checkpoint_best",
        "checkpoint_last",
    }
)

FROZEN_BASELINE = {
    "model": {
        "type": "ddpm_temporal_1d",
        "input_length": 65,
        "input_channels": 3,
        "base_channels": 64,
        "time_embedding_dim": 128,
    },
    "diffusion": {
        "steps": 100,
        "objective": "epsilon_prediction",
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.0002,
        "optimizer": "adamw",
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "max_epochs": 200,
        "early_stopping_patience": 20,
    },
    "normalization": {
        "type": "global_channel_zscore",
        "fit_split": "donor_train",
        "fit_axes": [0, 1],
        "fit_dtype": "float64",
        "output_dtype": "float32",
        "ddof": 0,
        "std_threshold": 0.00000001,
        "zero_variance_replacement": 1.0,
    },
    "reproducibility": {"seed": 42, "validation_seed": 424242},
}

FROZEN_TRAINING_SEEDS = (42, 123, 2026)
FROZEN_RUN_IDS = {
    42: "diffusion_seed42_frozen",
    123: "diffusion_seed123_frozen",
    2026: "diffusion_seed2026_frozen",
}

GLOBAL_CHANNEL_RUN_IDS = {
    42: "diffusion_seed42_global_channel",
    123: "diffusion_seed123_global_channel",
    2026: "diffusion_seed2026_global_channel",
}

LONG_TRAINING_RUN_ID = "diffusion_seed42_global_channel_long_training_diagnostic"
LONG_TRAINING_MAX_EPOCHS = 300
LONG_TRAINING_PATIENCE = 30


def validate_frozen_baseline(config: dict[str, Any]) -> None:
    """Reject any drift from the source baseline configuration."""

    if config != FROZEN_BASELINE:
        raise ValueError(
            "diffusion.yaml does not exactly match the frozen seed-42 baseline"
        )


def frozen_config_for_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a frozen config whose only runtime change is the training seed."""

    validate_frozen_baseline(config)
    seed = int(seed)
    if seed not in FROZEN_TRAINING_SEEDS:
        raise ValueError(f"Training seed must be one of {FROZEN_TRAINING_SEEDS}")
    effective = deepcopy(config)
    effective["reproducibility"]["seed"] = seed
    return effective


def validate_frozen_effective_config(config: dict[str, Any], seed: int) -> None:
    """Verify that an effective run config differs only by its approved seed."""

    expected = deepcopy(FROZEN_BASELINE)
    expected["reproducibility"]["seed"] = int(seed)
    if int(seed) not in FROZEN_TRAINING_SEEDS or config != expected:
        raise ValueError("Effective config drifted from the frozen multi-seed baseline")


def frozen_run_id(seed: int) -> str:
    try:
        return FROZEN_RUN_IDS[int(seed)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Training seed must be one of {FROZEN_TRAINING_SEEDS}") from error


def global_channel_run_id(seed: int) -> str:
    """Return the non-overwriting run ID for current global-normalized runs."""

    try:
        return GLOBAL_CHANNEL_RUN_IDS[int(seed)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Training seed must be one of {FROZEN_TRAINING_SEEDS}") from error


def long_training_diagnostic_config(config: dict[str, Any]) -> dict[str, Any]:
    """Derive the isolated seed-42 diagnostic by changing exactly two values."""

    validate_frozen_baseline(config)
    effective = deepcopy(config)
    effective["training"]["max_epochs"] = LONG_TRAINING_MAX_EPOCHS
    effective["training"]["early_stopping_patience"] = LONG_TRAINING_PATIENCE
    return effective


def validate_long_training_config(config: dict[str, Any]) -> None:
    """Reject any long-training drift beyond max epochs and patience."""

    expected = deepcopy(FROZEN_BASELINE)
    expected["training"]["max_epochs"] = LONG_TRAINING_MAX_EPOCHS
    expected["training"]["early_stopping_patience"] = LONG_TRAINING_PATIENCE
    if config != expected:
        raise ValueError("Long-training config differs from its two approved overrides")


def write_history(history: list[dict], path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    if tuple(frame.columns) != HISTORY_COLUMNS:
        raise ValueError(
            f"History columns must be {HISTORY_COLUMNS}, got {tuple(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("History cannot be empty")
    numeric = frame.loc[:, HISTORY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("History contains non-numeric or non-finite metrics")
    destination.write_text(frame.to_csv(index=False), encoding="utf-8")


def read_history(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if tuple(frame.columns) != HISTORY_COLUMNS:
        raise ValueError("History CSV has an incompatible schema")
    if frame.empty or not np.isfinite(
        frame.loc[:, HISTORY_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("History CSV is empty or contains non-finite metrics")
    return frame


def write_manifest(manifest: dict[str, Any], path: Path | str) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"Run manifest is missing fields: {sorted(missing)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path | str) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"Run manifest is missing fields: {sorted(missing)}")
    return manifest
