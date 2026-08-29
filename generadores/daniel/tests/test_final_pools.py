from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from generadores.daniel.src.final_pool_plots import generate_final_pool_figures
from generadores.daniel.src.final_pools import (
    FINAL_POOL_MANIFEST_FIELDS,
    build_final_pool_summary,
    pairing_max_abs_error,
    write_final_pool_manifest,
)
from generadores.daniel.src.temporary_nvda_calibration import (
    CalibrationStats,
    TemporaryNVDACalibrator,
    calculate_calibration_stats,
    generate_accepted_pool,
    load_nvda_visible_daily,
    physical_window_mask,
    validate_nvda_visible_frame,
)
from generadores.daniel.src.validation import CHANNEL_ORDER


ROOT = Path(__file__).resolve().parents[3]


def _stats(mean=(1.0, 2.0, 3.0), std=(0.5, 1.5, 2.0)) -> CalibrationStats:
    return CalibrationStats(
        mean=torch.tensor(mean, dtype=torch.float64),
        std=torch.tensor(std, dtype=torch.float64),
        n_daily_observations=3,
        observation_start="2022-07-05",
        observation_end="2022-07-07",
    )


def test_calibration_formula_is_affine_without_clipping_or_repair() -> None:
    normalized = torch.zeros((2, 65, 3), dtype=torch.float64)
    normalized[0, 0] = torch.tensor([-4.0, -3.0, -2.0])
    calibrator = TemporaryNVDACalibrator(_stats())
    calibrated = calibrator.transform(normalized)
    expected = _stats().mean + _stats().std * normalized
    assert torch.equal(calibrated, expected)
    assert calibrated[0, 0, 1] < 0


def test_canonical_calibration_source_is_unique_visible_only_and_deterministic() -> None:
    first = load_nvda_visible_daily(ROOT)
    second = load_nvda_visible_daily(ROOT)
    assert len(first) == first["date"].nunique() == 126
    assert set(first["ticker"]) == {"NVDA"}
    assert set(first["feature_block"]) == {"nvda_visible"}
    left = calculate_calibration_stats(first)
    right = calculate_calibration_stats(second)
    assert torch.equal(left.mean, right.mean)
    assert torch.equal(left.std, right.std)


@pytest.mark.parametrize(
    ("feature_block", "date"),
    [("nvda_hidden", "2022-06-30"), ("nvda_test", "2023-01-03")],
)
def test_hidden_and_test_rows_are_rejected(feature_block: str, date: str) -> None:
    frame = pd.DataFrame(
        {
            "feature_block": [feature_block],
            "date": [pd.Timestamp(date)],
            "ticker": ["NVDA"],
            **{channel: [1.0] for channel in CHANNEL_ORDER},
        }
    )
    with pytest.raises(ValueError):
        validate_nvda_visible_frame(frame)


def test_physical_validator_rejects_whole_windows_by_reason() -> None:
    values = torch.ones((6, 65, 3), dtype=torch.float64)
    values[0, 3, 0] = torch.nan
    values[1, 4, 1] = -0.01
    values[2, 5, 1] = torch.inf
    values[3, 6, 2] = -0.01
    values[4, 7, 2] = torch.inf
    valid, reasons = physical_window_mask(values)
    assert valid.tolist() == [False, False, False, False, False, True]
    assert reasons == {
        "non_finite_return": 1,
        "negative_range": 1,
        "non_finite_range": 1,
        "invalid_volume": 1,
        "non_finite_volume": 1,
        "other": 0,
    }


def test_reject_resample_is_exact_paired_and_reproducible() -> None:
    calibrator = TemporaryNVDACalibrator(_stats(mean=(0, 0, 0), std=(1, 1, 1)))

    def sample_fn(count: int, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        values = torch.rand((count, 65, 3), generator=generator)
        if seed == 42:
            values[:2, 0, 1] = -1
        return values

    first = generate_accepted_pool(
        sample_fn, calibrator, n_requested=10, base_seed=42, batch_size=4
    )
    second = generate_accepted_pool(
        sample_fn, calibrator, n_requested=10, base_seed=42, batch_size=4
    )
    assert first["n_candidates"] == 12
    assert first["n_accepted"] == 10
    assert first["n_rejected"] == 2
    assert first["rejection_reasons"]["negative_range"] == 2
    assert first["normalized"].shape == first["calibrated"].shape == (10, 65, 3)
    assert torch.equal(first["normalized"], first["calibrated"])
    assert torch.equal(first["normalized"], second["normalized"])
    assert pairing_max_abs_error(
        first["normalized"], first["calibrated"], _stats((0, 0, 0), (1, 1, 1)).mean,
        _stats((0, 0, 0), (1, 1, 1)).std,
    ) == 0.0


def _manifest(seed: int) -> dict:
    payload = {field: "value" for field in FINAL_POOL_MANIFEST_FIELDS}
    payload.update(
        {
            "training_seed": seed,
            "n_requested": 5000,
            "n_accepted": 5000,
            "n_candidates_generated": 5001,
            "n_rejected": 1,
            "rejection_rate": 1 / 5001,
            "generation_runtime_seconds": 1.0,
            "reproducibility_pass": True,
            "NVDA_hidden_used": False,
            "NVDA_test_used": False,
        }
    )
    return payload


def test_manifest_summary_and_plots_are_constructible(tmp_path) -> None:
    manifests = [_manifest(seed) for seed in (42, 123, 2026)]
    for seed, manifest in zip((42, 123, 2026), manifests, strict=True):
        manifest["checkpoint_sha256"] = f"checkpoint-{seed}"
        manifest["normalized_pool_sha256"] = f"normalized-{seed}"
        manifest["nvda_like_pool_sha256"] = f"calibrated-{seed}"
    path = tmp_path / "manifest.json"
    write_final_pool_manifest(manifests[0], path)
    summary = build_final_pool_summary(manifests)
    visible = pd.DataFrame({"log_return": np.linspace(-1, 1, 10)})
    pools = {
        seed: np.random.default_rng(seed).normal(size=(10, 65, 3))
        for seed in (42, 123, 2026)
    }
    figures = generate_final_pool_figures(summary, pools, visible, tmp_path / "figures")
    assert len(figures) == 3
    assert all(figure.is_file() and figure.stat().st_size > 0 for figure in figures.values())
