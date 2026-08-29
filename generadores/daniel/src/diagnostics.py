"""Preliminary normalized-space diagnostics local to Daniel's DDPM.

These functions are not the project's final common fidelity implementation.
They make no pass/fail claim about financial quality and never access target
data. Temporal ACF is computed inside each window before averaging, so window
boundaries can never create artificial lag pairs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .validation import CHANNEL_ORDER, INPUT_CHANNELS, WINDOW_LENGTH

NEAR_EXACT_TOLERANCE = 1e-6
COLLAPSED_STD_TOLERANCE = 1e-8
PAIRWISE_SUBSAMPLE_SIZE = 256
DIAGNOSTIC_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "model",
        "training_seed",
        "sampling_seed",
        "n_samples",
        "checkpoint_path",
        "checkpoint_sha256",
        "training_commit",
        "base_master_commit",
        "donor_train_sha256",
        "donor_validation_sha256",
        "normalizer_path",
        "normalizer_sha256",
        "window_shape",
        "channels",
        "space",
        "sampling_runtime_seconds",
        "device",
        "sample_file",
        "sample_sha256",
        "finite",
        "n_unique",
        "duplicate_count",
        "tables",
        "figures",
    }
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_numpy_windows(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 3 or tuple(array.shape[1:]) != (WINDOW_LENGTH, INPUT_CHANNELS):
        raise ValueError(f"Expected (N, {WINDOW_LENGTH}, {INPUT_CHANNELS}), got {array.shape}")
    if array.dtype not in (np.float32, np.float64):
        raise TypeError(f"Expected float32/float64, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("Windows contain NaN or infinite values")
    return array


def save_sample_pool(
    path: Path | str,
    samples: np.ndarray | torch.Tensor,
    *,
    seed: int,
) -> str:
    array = as_numpy_windows(samples)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        samples=array,
        seed=np.asarray(seed, dtype=np.int64),
        window_shape=np.asarray([WINDOW_LENGTH, INPUT_CHANNELS], dtype=np.int64),
        channel_order=np.asarray(CHANNEL_ORDER),
    )
    return sha256_file(destination)


def load_sample_pool(path: Path | str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {
            "samples": archive["samples"],
            "seed": int(archive["seed"]),
            "window_shape": tuple(int(value) for value in archive["window_shape"]),
            "channel_order": tuple(str(value) for value in archive["channel_order"]),
        }
    as_numpy_windows(payload["samples"])
    if payload["window_shape"] != (WINDOW_LENGTH, INPUT_CHANNELS):
        raise ValueError("Stored window_shape is incompatible")
    if payload["channel_order"] != CHANNEL_ORDER:
        raise ValueError("Stored channel_order is incompatible")
    return payload


def _skewness_and_excess_kurtosis(values: np.ndarray) -> tuple[float, float]:
    centered = values - values.mean()
    variance = np.mean(centered**2)
    if variance <= 0.0:
        return float("nan"), float("nan")
    std = np.sqrt(variance)
    skewness = np.mean(centered**3) / std**3
    excess_kurtosis = np.mean(centered**4) / variance**2 - 3.0
    return float(skewness), float(excess_kurtosis)


def channel_statistics(
    real_validation: np.ndarray | torch.Tensor,
    synthetic: np.ndarray | torch.Tensor,
) -> pd.DataFrame:
    real = as_numpy_windows(real_validation).astype(np.float64, copy=False)
    generated = as_numpy_windows(synthetic).astype(np.float64, copy=False)
    rows: list[dict[str, Any]] = []
    for source, windows in (("real_validation", real), ("synthetic", generated)):
        for channel_index, channel in enumerate(CHANNEL_ORDER):
            values = windows[:, :, channel_index].reshape(-1)
            skewness, kurtosis = _skewness_and_excess_kurtosis(values)
            percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
            rows.append(
                {
                    "source": source,
                    "channel": channel,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "skewness": skewness,
                    "excess_kurtosis": kurtosis,
                    "p01": float(percentiles[0]),
                    "p05": float(percentiles[1]),
                    "p25": float(percentiles[2]),
                    "p50": float(percentiles[3]),
                    "p75": float(percentiles[4]),
                    "p95": float(percentiles[5]),
                    "p99": float(percentiles[6]),
                }
            )
    return pd.DataFrame(rows)


def wasserstein_1d(first: np.ndarray, second: np.ndarray) -> float:
    """Exact empirical W1 via the integral between the two empirical CDFs."""

    left = np.sort(np.asarray(first, dtype=np.float64).reshape(-1))
    right = np.sort(np.asarray(second, dtype=np.float64).reshape(-1))
    if left.size == 0 or right.size == 0:
        raise ValueError("Wasserstein inputs cannot be empty")
    combined = np.sort(np.concatenate((left, right)))
    deltas = np.diff(combined)
    left_cdf = np.searchsorted(left, combined[:-1], side="right") / left.size
    right_cdf = np.searchsorted(right, combined[:-1], side="right") / right.size
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def wasserstein_by_channel(
    real_validation: np.ndarray | torch.Tensor,
    synthetic: np.ndarray | torch.Tensor,
) -> pd.DataFrame:
    real = as_numpy_windows(real_validation)
    generated = as_numpy_windows(synthetic)
    return pd.DataFrame(
        [
            {
                "channel": channel,
                "wasserstein_1": wasserstein_1d(
                    real[:, :, index], generated[:, :, index]
                ),
            }
            for index, channel in enumerate(CHANNEL_ORDER)
        ]
    )


def mean_window_acf(
    windows: np.ndarray | torch.Tensor,
    *,
    channel_index: int = 0,
    max_lag: int = 20,
    absolute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-window biased ACF; no lag pair crosses a boundary."""

    array = as_numpy_windows(windows).astype(np.float64, copy=False)
    if not 0 <= channel_index < INPUT_CHANNELS:
        raise ValueError("Invalid channel index")
    if not 1 <= max_lag < WINDOW_LENGTH:
        raise ValueError("max_lag must be between 1 and 64")
    values = array[:, :, channel_index]
    if absolute:
        values = np.abs(values)
    centered = values - values.mean(axis=1, keepdims=True)
    denominators = np.sum(centered**2, axis=1)
    acfs: list[float] = []
    valid_counts: list[int] = []
    for lag in range(1, max_lag + 1):
        numerators = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        valid = denominators > 0.0
        if not np.any(valid):
            raise ValueError("Every window has zero temporal variance")
        acfs.append(float(np.mean(numerators[valid] / denominators[valid])))
        valid_counts.append(int(valid.sum()))
    return np.asarray(acfs), np.asarray(valid_counts)


def acf_comparison(
    real_validation: np.ndarray | torch.Tensor,
    synthetic: np.ndarray | torch.Tensor,
    *,
    max_lag: int = 20,
    absolute: bool = False,
) -> pd.DataFrame:
    real_acf, real_counts = mean_window_acf(
        real_validation, max_lag=max_lag, absolute=absolute
    )
    synthetic_acf, synthetic_counts = mean_window_acf(
        synthetic, max_lag=max_lag, absolute=absolute
    )
    return pd.DataFrame(
        {
            "lag": np.arange(1, max_lag + 1),
            "real_validation_acf": real_acf,
            "synthetic_acf": synthetic_acf,
            "difference": synthetic_acf - real_acf,
            "absolute_difference": np.abs(synthetic_acf - real_acf),
            "real_valid_windows": real_counts,
            "synthetic_valid_windows": synthetic_counts,
        }
    )


def cross_channel_correlations(
    real_validation: np.ndarray | torch.Tensor,
    synthetic: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    real = as_numpy_windows(real_validation).reshape(-1, INPUT_CHANNELS)
    generated = as_numpy_windows(synthetic).reshape(-1, INPUT_CHANNELS)
    real_matrix = np.corrcoef(real, rowvar=False)
    synthetic_matrix = np.corrcoef(generated, rowvar=False)
    absolute_difference = np.abs(synthetic_matrix - real_matrix)
    off_diagonal = ~np.eye(INPUT_CHANNELS, dtype=bool)
    return {
        "real": real_matrix,
        "synthetic": synthetic_matrix,
        "absolute_difference": absolute_difference,
        "mean_absolute_off_diagonal_difference": float(
            absolute_difference[off_diagonal].mean()
        ),
        "max_absolute_off_diagonal_difference": float(
            absolute_difference[off_diagonal].max()
        ),
    }


def correlation_table(correlations: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for matrix_name in ("real", "synthetic", "absolute_difference"):
        matrix = correlations[matrix_name]
        for row_index, row_channel in enumerate(CHANNEL_ORDER):
            for column_index, column_channel in enumerate(CHANNEL_ORDER):
                rows.append(
                    {
                        "matrix": matrix_name,
                        "row_channel": row_channel,
                        "column_channel": column_channel,
                        "value": float(matrix[row_index, column_index]),
                    }
                )
    return pd.DataFrame(rows)


def _flatten_observations(windows: np.ndarray | torch.Tensor) -> np.ndarray:
    array = as_numpy_windows(windows)
    return array.reshape(array.shape[0], -1).astype(np.float64, copy=False)


def nearest_neighbor_distances(
    queries: np.ndarray | torch.Tensor,
    reference: np.ndarray | torch.Tensor,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Euclidean NN distances using bounded query×reference matrix chunks."""

    query = _flatten_observations(queries)
    candidates = _flatten_observations(reference)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    reference_norms = np.sum(candidates**2, axis=1)
    distances = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk_size):
        block = query[start : start + chunk_size]
        squared = (
            np.sum(block**2, axis=1, keepdims=True)
            + reference_norms[None, :]
            - 2.0 * block @ candidates.T
        )
        nearest_indices = np.argmin(squared, axis=1)
        nearest = candidates[nearest_indices]
        distances[start : start + len(block)] = np.linalg.norm(block - nearest, axis=1)
    return distances


def distance_summary(distances: np.ndarray) -> dict[str, float]:
    values = np.asarray(distances, dtype=np.float64)
    percentiles = np.percentile(values, [1, 5, 50, 95])
    return {
        "min": float(values.min()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "median": float(percentiles[2]),
        "mean": float(values.mean()),
        "p95": float(percentiles[3]),
        "max": float(values.max()),
    }


def memorization_diagnostics(
    synthetic: np.ndarray | torch.Tensor,
    train: np.ndarray | torch.Tensor,
    validation: np.ndarray | torch.Tensor,
    *,
    near_exact_tolerance: float = NEAR_EXACT_TOLERANCE,
) -> dict[str, Any]:
    synthetic_distances = nearest_neighbor_distances(synthetic, train)
    validation_distances = nearest_neighbor_distances(validation, train)
    exact_count = int(np.count_nonzero(synthetic_distances == 0.0))
    near_count = int(np.count_nonzero(synthetic_distances <= near_exact_tolerance))
    return {
        "synthetic_to_train": synthetic_distances,
        "validation_to_train": validation_distances,
        "synthetic_to_train_summary": distance_summary(synthetic_distances),
        "validation_to_train_summary": distance_summary(validation_distances),
        "exact_duplicate_count": exact_count,
        "near_exact_count_including_exact": near_count,
        "near_exact_tolerance": near_exact_tolerance,
    }


def diversity_diagnostics(
    synthetic: np.ndarray | torch.Tensor,
    *,
    seed: int = 42,
    subsample_size: int = PAIRWISE_SUBSAMPLE_SIZE,
    collapsed_std_tolerance: float = COLLAPSED_STD_TOLERANCE,
) -> dict[str, Any]:
    array = as_numpy_windows(synthetic)
    flattened = array.reshape(array.shape[0], -1)
    n_unique = int(np.unique(flattened, axis=0).shape[0])
    duplicate_count = int(array.shape[0] - n_unique)
    temporal_stds = array.std(axis=1, ddof=0)
    collapsed_by_channel = {
        channel: int(np.count_nonzero(temporal_stds[:, index] <= collapsed_std_tolerance))
        for index, channel in enumerate(CHANNEL_ORDER)
    }
    collapsed_any = int(
        np.count_nonzero(np.any(temporal_stds <= collapsed_std_tolerance, axis=1))
    )

    size = min(int(subsample_size), array.shape[0])
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(array.shape[0], size=size, replace=False))
    subset = flattened[indices].astype(np.float64, copy=False)
    squared = (
        np.sum(subset**2, axis=1, keepdims=True)
        + np.sum(subset**2, axis=1)[None, :]
        - 2.0 * subset @ subset.T
    )
    pairwise = np.sqrt(np.maximum(squared, 0.0))
    upper = pairwise[np.triu_indices(size, k=1)]
    return {
        "n_unique": n_unique,
        "duplicate_count": duplicate_count,
        "duplicate_percent": 100.0 * duplicate_count / array.shape[0],
        "variance_by_channel": {
            channel: float(array[:, :, index].var(ddof=0))
            for index, channel in enumerate(CHANNEL_ORDER)
        },
        "collapsed_std_tolerance": collapsed_std_tolerance,
        "collapsed_windows_by_channel": collapsed_by_channel,
        "collapsed_windows_any_channel": collapsed_any,
        "pairwise_subsample_seed": seed,
        "pairwise_subsample_size": size,
        "pairwise_distance_summary": distance_summary(upper),
    }


def write_diagnostic_manifest(manifest: dict[str, Any], path: Path | str) -> None:
    missing = DIAGNOSTIC_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"Diagnostic manifest is missing fields: {sorted(missing)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
