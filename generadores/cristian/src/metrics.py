from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CHANNELS, WINDOW_LENGTH, windows_to_array


def channel_summary(windows: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, channel in enumerate(CHANNELS):
        values = windows[:, :, idx].reshape(-1)
        rows.append(
            {
                "channel": channel,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def lag1_autocorr(values: np.ndarray) -> float:
    if values.ndim != 1 or len(values) < 2:
        return float("nan")
    centered = values - values.mean()
    denom = np.dot(centered, centered)
    if denom <= 0:
        return float("nan")
    return float(np.dot(centered[:-1], centered[1:]) / denom)


def temporal_autocorr_report(windows: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, channel in enumerate(CHANNELS):
        per_window = [lag1_autocorr(window[:, idx]) for window in windows]
        rows.append(
            {
                "channel": channel,
                "autocorr_lag1_mean": float(np.nanmean(per_window)),
                "autocorr_lag1_std": float(np.nanstd(per_window)),
            }
        )
    return pd.DataFrame(rows)


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel (flattened windows)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if gamma is None:
        combined = np.concatenate([x, y], axis=0)
        gamma = 1.0 / max(np.median(np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2)), 1e-8)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * sq)

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    n = x.shape[0]
    m = y.shape[0]
    return float(
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2 * k_xy.mean()
    )


@dataclass(frozen=True)
class ValidationReport:
    n_real: int
    n_synthetic: int
    channel_summary_real: pd.DataFrame
    channel_summary_synthetic: pd.DataFrame
    autocorr_real: pd.DataFrame
    autocorr_synthetic: pd.DataFrame
    mmd_flat: float

    def to_dict(self) -> dict:
        return {
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "mmd_flat": self.mmd_flat,
            "channel_summary_real": self.channel_summary_real.to_dict(orient="records"),
            "channel_summary_synthetic": self.channel_summary_synthetic.to_dict(orient="records"),
            "autocorr_real": self.autocorr_real.to_dict(orient="records"),
            "autocorr_synthetic": self.autocorr_synthetic.to_dict(orient="records"),
        }


def compare_windows(real: np.ndarray, synthetic: np.ndarray) -> ValidationReport:
    real_flat = real.reshape(len(real), -1)
    synthetic_flat = synthetic.reshape(len(synthetic), -1)
    sample_size = min(len(real_flat), len(synthetic_flat), 512)
    rng = np.random.default_rng(42)
    real_idx = rng.choice(len(real_flat), size=sample_size, replace=False)
    synth_idx = rng.choice(len(synthetic_flat), size=sample_size, replace=False)
    return ValidationReport(
        n_real=len(real),
        n_synthetic=len(synthetic),
        channel_summary_real=channel_summary(real),
        channel_summary_synthetic=channel_summary(synthetic),
        autocorr_real=temporal_autocorr_report(real),
        autocorr_synthetic=temporal_autocorr_report(synthetic),
        mmd_flat=mmd_rbf(real_flat[real_idx], synthetic_flat[synth_idx]),
    )


def save_validation_report(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def compare_parquet_splits(
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
) -> ValidationReport:
    return compare_windows(windows_to_array(real_frame), windows_to_array(synthetic_frame))


def per_seed_validation_summary(
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in sorted(int(value) for value in synthetic_frame["seed"].unique()):
        subset = synthetic_frame.loc[synthetic_frame["seed"].eq(seed)]
        report = compare_parquet_splits(real_frame, subset)
        real_summary = report.channel_summary_real.set_index("channel")
        synth_summary = report.channel_summary_synthetic.set_index("channel")
        for channel in CHANNELS:
            rows.append(
                {
                    "seed": seed,
                    "channel": channel,
                    "n_synthetic": report.n_synthetic,
                    "mmd_flat": report.mmd_flat,
                    "mean_validation": float(real_summary.loc[channel, "mean"]),
                    "mean_synthetic": float(synth_summary.loc[channel, "mean"]),
                    "std_validation": float(real_summary.loc[channel, "std"]),
                    "std_synthetic": float(synth_summary.loc[channel, "std"]),
                    "autocorr_validation": float(
                        report.autocorr_real.set_index("channel").loc[channel, "autocorr_lag1_mean"]
                    ),
                    "autocorr_synthetic": float(
                        report.autocorr_synthetic.set_index("channel").loc[channel, "autocorr_lag1_mean"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def per_seed_mmd_summary(
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in sorted(int(value) for value in synthetic_frame["seed"].unique()):
        subset = synthetic_frame.loc[synthetic_frame["seed"].eq(seed)]
        report = compare_parquet_splits(real_frame, subset)
        rows.append(
            {
                "seed": seed,
                "n_synthetic": report.n_synthetic,
                "mmd_flat": report.mmd_flat,
            }
        )
    return pd.DataFrame(rows)
