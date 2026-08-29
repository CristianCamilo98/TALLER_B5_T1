"""TEMPORARY / REPLACE WITH COMMON CALIBRATION WHEN AVAILABLE.

This module reads only the canonical daily ``nvda_visible`` feature block and
applies a post-sampling affine transform. It never reads hidden, full-history,
or test windows and never repairs physically invalid synthetic windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .validation import CHANNEL_ORDER, validate_window_tensor


NVDA_VISIBLE_SOURCE = Path("data/features/daily_features_by_split.parquet")
NVDA_VISIBLE_SOURCE_SHA256 = (
    "6cc2d5c7c1490e03e34b3b7dbf5fb49ce0c71d8a00fd81da6f2e38e2d8c3419e"
)
NVDA_VISIBLE_START = pd.Timestamp("2022-07-01")
NVDA_VISIBLE_END = pd.Timestamp("2022-12-30")
NVDA_VISIBLE_FEATURE_COUNT = 126
REJECTION_REASONS = (
    "non_finite_return",
    "negative_range",
    "non_finite_range",
    "invalid_volume",
    "non_finite_volume",
    "other",
)
SUBSEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class CalibrationStats:
    mean: torch.Tensor
    std: torch.Tensor
    n_daily_observations: int
    observation_start: str
    observation_end: str
    ddof: int = 0

    def __post_init__(self) -> None:
        if tuple(self.mean.shape) != (3,) or tuple(self.std.shape) != (3,):
            raise ValueError("Calibration mean/std must each have shape (3,)")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("Calibration statistics must be finite")
        if torch.any(self.std <= 0):
            raise ValueError("Calibration standard deviations must be positive")
        if self.ddof != 0:
            raise ValueError("Frozen NVDA calibration uses population std (ddof=0)")

    def as_dict(self) -> dict:
        return {
            "channels": list(CHANNEL_ORDER),
            "mean": dict(zip(CHANNEL_ORDER, self.mean.tolist(), strict=True)),
            "std": dict(zip(CHANNEL_ORDER, self.std.tolist(), strict=True)),
            "ddof": self.ddof,
            "n_daily_observations": self.n_daily_observations,
            "observation_start": self.observation_start,
            "observation_end": self.observation_end,
        }


def validate_nvda_visible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"feature_block", "date", "ticker", *CHANNEL_ORDER}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"NVDA visible source is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    if result.empty:
        raise ValueError("NVDA visible daily feature block is empty")
    if set(result["feature_block"]) != {"nvda_visible"}:
        raise ValueError("Calibration input includes a non-visible feature block")
    if set(result["ticker"]) != {"NVDA"}:
        raise ValueError("Calibration input includes a non-NVDA ticker")
    if result["date"].min() < NVDA_VISIBLE_START:
        raise ValueError("Calibration input includes pre-visible NVDA observations")
    if result["date"].max() > NVDA_VISIBLE_END:
        raise ValueError("Calibration input includes hidden/test NVDA observations")
    if result.duplicated(["ticker", "date"]).any():
        raise ValueError("Calibration input repeats daily observations")
    values = result.loc[:, CHANNEL_ORDER].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Calibration input contains NaN or infinite values")
    return result.sort_values("date").reset_index(drop=True)


def load_nvda_visible_daily(repository_root: Path | str) -> pd.DataFrame:
    """Read only NVDA visible rows through explicit Parquet predicates."""

    source = Path(repository_root) / NVDA_VISIBLE_SOURCE
    columns = ["feature_block", "date", "ticker", *CHANNEL_ORDER]
    table = pq.read_table(
        source,
        columns=columns,
        filters=[
            ("feature_block", "=", "nvda_visible"),
            ("ticker", "=", "NVDA"),
            ("date", ">=", NVDA_VISIBLE_START),
            ("date", "<=", NVDA_VISIBLE_END),
        ],
    )
    frame = validate_nvda_visible_frame(table.to_pandas())
    if len(frame) != NVDA_VISIBLE_FEATURE_COUNT:
        raise ValueError(
            f"Expected {NVDA_VISIBLE_FEATURE_COUNT} unique visible feature rows, "
            f"got {len(frame)}"
        )
    return frame


def calculate_calibration_stats(frame: pd.DataFrame) -> CalibrationStats:
    visible = validate_nvda_visible_frame(frame)
    values = visible.loc[:, CHANNEL_ORDER].to_numpy(dtype=np.float64)
    return CalibrationStats(
        mean=torch.tensor(values.mean(axis=0), dtype=torch.float64),
        std=torch.tensor(values.std(axis=0, ddof=0), dtype=torch.float64),
        n_daily_observations=len(visible),
        observation_start=visible["date"].min().strftime("%Y-%m-%d"),
        observation_end=visible["date"].max().strftime("%Y-%m-%d"),
    )


class TemporaryNVDACalibrator:
    """Apply the frozen affine calibration without clipping or donor scalers."""

    def __init__(self, stats: CalibrationStats) -> None:
        self.stats = stats

    def transform(self, normalized: torch.Tensor) -> torch.Tensor:
        validate_window_tensor(normalized, name="normalized")
        mean = self.stats.mean.to(dtype=normalized.dtype, device=normalized.device)
        std = self.stats.std.to(dtype=normalized.dtype, device=normalized.device)
        calibrated = mean + std * normalized
        if calibrated.shape != normalized.shape:
            raise RuntimeError("Calibration changed the synthetic tensor shape")
        return calibrated


def physical_window_mask(calibrated: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    """Return whole-window validity and mutually exclusive rejection counts."""

    if calibrated.ndim != 3 or tuple(calibrated.shape[1:]) != (65, 3):
        raise ValueError("Calibrated tensor must have shape (N, 65, 3)")
    count = calibrated.shape[0]
    remaining = torch.ones(count, dtype=torch.bool, device=calibrated.device)
    reasons = {name: 0 for name in REJECTION_REASONS}

    def reject(name: str, condition: torch.Tensor) -> None:
        nonlocal remaining
        selected = remaining & condition
        reasons[name] = int(selected.sum().item())
        remaining = remaining & ~selected

    returns = calibrated[:, :, 0]
    ranges = calibrated[:, :, 1]
    log_volumes = calibrated[:, :, 2]
    reject("non_finite_return", ~torch.isfinite(returns).all(dim=1))
    reject("non_finite_range", ~torch.isfinite(ranges).all(dim=1))
    reject("negative_range", (ranges < 0).any(dim=1))
    finite_logs = torch.isfinite(log_volumes).all(dim=1)
    volume_proxy = torch.expm1(log_volumes)
    finite_proxy = torch.isfinite(volume_proxy).all(dim=1)
    reject("non_finite_volume", ~(finite_logs & finite_proxy))
    reject("invalid_volume", (volume_proxy < 0).any(dim=1))
    reject("other", torch.zeros(count, dtype=torch.bool, device=calibrated.device))
    return remaining, reasons


def candidate_subseed(base_seed: int, batch_index: int) -> int:
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    return (int(base_seed) + batch_index * SUBSEED_STRIDE) % (2**63 - 1)


def generate_accepted_pool(
    sample_fn: Callable[[int, int], torch.Tensor],
    calibrator: TemporaryNVDACalibrator,
    *,
    n_requested: int,
    base_seed: int,
    batch_size: int = 256,
) -> dict:
    """Reject whole invalid windows and resample until exactly N are accepted."""

    if n_requested <= 0 or batch_size <= 0:
        raise ValueError("n_requested and batch_size must be positive")
    normalized_parts: list[torch.Tensor] = []
    calibrated_parts: list[torch.Tensor] = []
    reasons = {name: 0 for name in REJECTION_REASONS}
    subseeds: list[int] = []
    accepted = 0
    candidates = 0
    batch_index = 0
    while accepted < n_requested:
        requested_now = min(batch_size, n_requested - accepted)
        subseed = candidate_subseed(base_seed, batch_index)
        candidate = sample_fn(requested_now, subseed).detach().cpu()
        validate_window_tensor(candidate, name="candidate")
        calibrated = calibrator.transform(candidate)
        valid, batch_reasons = physical_window_mask(calibrated)
        normalized_parts.append(candidate[valid])
        calibrated_parts.append(calibrated[valid])
        valid_count = int(valid.sum().item())
        accepted += valid_count
        candidates += requested_now
        for name in REJECTION_REASONS:
            reasons[name] += batch_reasons[name]
        subseeds.append(subseed)
        batch_index += 1
    normalized = torch.cat(normalized_parts, dim=0)
    calibrated = torch.cat(calibrated_parts, dim=0)
    if len(normalized) != n_requested or candidates != n_requested + sum(reasons.values()):
        raise RuntimeError("Reject/resample accounting is inconsistent")
    return {
        "normalized": normalized,
        "calibrated": calibrated,
        "n_candidates": candidates,
        "n_accepted": n_requested,
        "n_rejected": candidates - n_requested,
        "rejection_reasons": reasons,
        "subseeds": subseeds,
        "subseed_mechanism": f"(base_seed + batch_index * {SUBSEED_STRIDE}) mod (2^63-1)",
    }
