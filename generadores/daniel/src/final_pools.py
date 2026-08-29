"""Runtime artifact contracts for calibrated final Diffusion pools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .frozen_runs import sha256_file
from .validation import CHANNEL_ORDER, validate_window_tensor


FINAL_POOL_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "model",
        "training_seed",
        "sampling_seed",
        "frozen_training_sha",
        "generation_code_sha",
        "base_master_commit",
        "checkpoint_path",
        "checkpoint_sha256",
        "donor_train_sha256",
        "donor_validation_sha256",
        "normalizer_sha256",
        "nvda_calibration_source_path",
        "nvda_calibration_source_sha256",
        "nvda_visible_start",
        "nvda_visible_end",
        "nvda_daily_observation_count",
        "nvda_calibration_mean",
        "nvda_calibration_std",
        "nvda_calibration_ddof",
        "window_shape",
        "channels",
        "n_requested",
        "n_candidates_generated",
        "n_accepted",
        "n_rejected",
        "rejection_rate",
        "rejection_reason_counts",
        "normalized_pool_path",
        "normalized_pool_sha256",
        "nvda_like_pool_path",
        "nvda_like_pool_sha256",
        "generation_runtime_seconds",
        "device",
        "python_version",
        "torch_version",
        "reproducibility_pass",
        "NVDA_hidden_used",
        "NVDA_test_used",
    }
)

SUMMARY_COLUMNS = (
    "seed",
    "checkpoint_sha256",
    "n_candidates",
    "n_accepted",
    "n_rejected",
    "rejection_rate",
    "normalized_pool_sha256",
    "nvda_like_pool_sha256",
    "runtime_seconds",
    "reproducibility_pass",
)


def save_final_pool(
    path: Path | str,
    samples: torch.Tensor,
    *,
    training_seed: int,
    sampling_seed: int,
    space: str,
    checkpoint_sha256: str,
    calibration_stats_sha256: str,
    generation_commit: str,
) -> str:
    validate_window_tensor(samples, expected_count=5000, name="final_pool")
    if space not in {"normalized", "nvda_like"}:
        raise ValueError("Pool space must be normalized or nvda_like")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        samples=samples.detach().cpu().numpy(),
        model=np.asarray("diffusion"),
        training_seed=np.asarray(training_seed, dtype=np.int64),
        sampling_seed=np.asarray(sampling_seed, dtype=np.int64),
        window_shape=np.asarray([65, 3], dtype=np.int64),
        channel_order=np.asarray(CHANNEL_ORDER),
        space=np.asarray(space),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        calibration_stats_sha256=np.asarray(calibration_stats_sha256),
        generation_commit=np.asarray(generation_commit),
    )
    return sha256_file(destination)


def load_final_pool(path: Path | str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    samples = torch.from_numpy(payload["samples"])
    validate_window_tensor(samples, expected_count=5000, name="stored_final_pool")
    if tuple(payload["window_shape"].tolist()) != (65, 3):
        raise ValueError("Stored pool has an incompatible window_shape")
    if tuple(payload["channel_order"].tolist()) != CHANNEL_ORDER:
        raise ValueError("Stored pool has an incompatible channel order")
    return {
        "samples": samples,
        "model": str(payload["model"]),
        "training_seed": int(payload["training_seed"]),
        "sampling_seed": int(payload["sampling_seed"]),
        "window_shape": (65, 3),
        "channel_order": CHANNEL_ORDER,
        "space": str(payload["space"]),
        "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        "calibration_stats_sha256": str(payload["calibration_stats_sha256"]),
        "generation_commit": str(payload["generation_commit"]),
    }


def channel_sanity_summary(samples: torch.Tensor) -> list[dict[str, float | str]]:
    validate_window_tensor(samples, expected_count=5000, name="nvda_like_pool")
    array = samples.detach().cpu().numpy().astype(np.float64, copy=False)
    rows = []
    for index, channel in enumerate(CHANNEL_ORDER):
        values = array[:, :, index].reshape(-1)
        percentiles = np.percentile(values, [1, 5, 50, 95, 99])
        rows.append(
            {
                "channel": channel,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p01": float(percentiles[0]),
                "p05": float(percentiles[1]),
                "median": float(percentiles[2]),
                "p95": float(percentiles[3]),
                "p99": float(percentiles[4]),
                "max": float(values.max()),
            }
        )
    return rows


def pairing_max_abs_error(
    normalized: torch.Tensor,
    calibrated: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    expected = mean.to(calibrated) + std.to(calibrated) * normalized.to(calibrated)
    return float(torch.max(torch.abs(calibrated - expected)).item())


def write_json_artifact(payload: dict, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_final_pool_manifest(manifest: dict, path: Path | str) -> None:
    missing = FINAL_POOL_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"Final pool manifest is missing fields: {sorted(missing)}")
    if manifest["n_accepted"] != 5000 or manifest["n_requested"] != 5000:
        raise ValueError("Final pool manifest must certify exactly 5000 windows")
    if manifest["NVDA_hidden_used"] or manifest["NVDA_test_used"]:
        raise ValueError("Forbidden NVDA data cannot be used")
    write_json_artifact(manifest, path)


def build_final_pool_summary(manifests: list[dict]) -> pd.DataFrame:
    if {int(item["training_seed"]) for item in manifests} != {42, 123, 2026}:
        raise ValueError("Summary requires final pool manifests for all three seeds")
    rows = [
        {
            "seed": int(item["training_seed"]),
            "checkpoint_sha256": item["checkpoint_sha256"],
            "n_candidates": int(item["n_candidates_generated"]),
            "n_accepted": int(item["n_accepted"]),
            "n_rejected": int(item["n_rejected"]),
            "rejection_rate": float(item["rejection_rate"]),
            "normalized_pool_sha256": item["normalized_pool_sha256"],
            "nvda_like_pool_sha256": item["nvda_like_pool_sha256"],
            "runtime_seconds": float(item["generation_runtime_seconds"]),
            "reproducibility_pass": bool(item["reproducibility_pass"]),
        }
        for item in sorted(manifests, key=lambda value: int(value["training_seed"]))
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
