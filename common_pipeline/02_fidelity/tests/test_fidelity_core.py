from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fidelity_core
from fidelity_core import (
    CHANNEL_ORDER,
    FEATURE_COUNT,
    C2STResult,
    apply_common_subset,
    assemble_joint_embedding_input,
    c2st_out_of_fold,
    channel_correlation_matrix,
    common_subset_indices,
    load_canonical_windows,
    load_synthetic_pool,
    GlobalChannelStatistics,
    mean_window_acf,
    reconstruct_windows,
    validate_windows,
    wasserstein_table,
)


def _windows(count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, 65, 3)).astype(np.float32)


def test_shape_reconstruction_is_session_major_65_by_3() -> None:
    flat = np.arange(2 * FEATURE_COUNT, dtype=np.float32).reshape(2, FEATURE_COUNT)
    reconstructed = reconstruct_windows(flat, name="fixture", expected_count=2)
    assert reconstructed.shape == (2, 65, 3)
    assert np.array_equal(reconstructed[0, 0], [0.0, 1.0, 2.0])
    assert np.array_equal(reconstructed[0, 64], [192.0, 193.0, 194.0])


def test_synthetic_loader_accepts_repeated_list_metadata(tmp_path: Path) -> None:
    path = tmp_path / "method_seed42_normalized.parquet"
    windows = _windows(2)
    pd.DataFrame(
        {
            "training_seed": [42, 42],
            "space": ["global_channel_normalized"] * 2,
            "channel_order": [list(CHANNEL_ORDER), list(CHANNEL_ORDER)],
            "features_flat": [window.reshape(-1) for window in windows],
        }
    ).to_parquet(path, index=False)
    pool = load_synthetic_pool(path, method="fixture", expected_count=2)
    assert pool.channel_order == CHANNEL_ORDER
    assert np.array_equal(pool.windows, windows)


def test_synthetic_loader_rejects_incompatible_published_scaler(tmp_path: Path) -> None:
    path = tmp_path / "method_seed42_normalized.parquet"
    windows = _windows(2)
    pd.DataFrame({"features_flat": [window.reshape(-1) for window in windows]}).to_parquet(
        path, index=False
    )
    path.with_name(f"{path.stem}_manifest.json").write_text(
        '{"seed": 42, "space": "z-score donor_train", '
        '"channels": ["log_return", "log_high_low_range", "log1p_volume"], '
        '"scaler_mean": [9, 9, 9], "scaler_std": [1, 1, 1]}',
        encoding="utf-8",
    )
    expected = GlobalChannelStatistics(
        mean=np.zeros(3), std=np.ones(3), raw_std=np.ones(3)
    )
    with pytest.raises(ValueError, match="not the common float64"):
        load_synthetic_pool(
            path, method="fixture", expected_count=2, expected_normalizer=expected
        )


def test_common_subset_is_reproducible_and_shared_across_methods() -> None:
    first = common_subset_indices(pool_size=100, subset_size=20, evaluation_subset_seed=42)
    second = common_subset_indices(pool_size=100, subset_size=20, evaluation_subset_seed=42)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 20

    base = np.arange(100 * 65 * 3, dtype=np.float32).reshape(100, 65, 3)
    selected = apply_common_subset({"a": base, "b": base + 1.0}, first)
    assert np.array_equal(selected["a"], base[first])
    assert np.array_equal(selected["b"] - 1.0, base[first])


def test_wasserstein_is_zero_for_identical_distributions() -> None:
    windows = _windows(12)
    table = wasserstein_table(windows, {"identical": windows.copy()})
    assert table["channel"].tolist() == list(CHANNEL_ORDER)
    assert np.allclose(table["wasserstein_1"], 0.0, atol=0.0)
    assert np.allclose(table["mean_wasserstein_across_channels"], 0.0, atol=0.0)


def test_acf_has_twenty_lags_and_never_crosses_window_boundaries() -> None:
    windows = _windows(8)
    calculated, valid = mean_window_acf(windows, max_lag=20)
    assert calculated.shape == (20,)
    assert valid.shape == (20,)

    values = windows[:, :, 0].astype(np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    expected_lag_one = np.mean(
        np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
        / np.sum(centered**2, axis=1)
    )
    assert calculated[0] == pytest.approx(expected_lag_one)


def test_channel_correlation_is_finite_three_by_three() -> None:
    matrix = channel_correlation_matrix(_windows(10))
    assert matrix.shape == (3, 3)
    assert np.isfinite(matrix).all()
    assert np.allclose(np.diag(matrix), 1.0)


def test_c2st_uses_pipeline_and_out_of_fold_probabilities(monkeypatch) -> None:
    recorded = {}

    def fake_cross_val_predict(estimator, features, labels, *, cv, method):
        recorded["steps"] = list(estimator.named_steps)
        recorded["features"] = features.shape
        recorded["splits"] = cv.n_splits
        recorded["shuffle"] = cv.shuffle
        recorded["random_state"] = cv.random_state
        recorded["method"] = method
        return np.column_stack((1.0 - (0.2 + 0.6 * labels), 0.2 + 0.6 * labels))

    monkeypatch.setattr(fidelity_core, "cross_val_predict", fake_cross_val_predict)
    result = c2st_out_of_fold(_windows(20, 1), _windows(20, 2))
    assert isinstance(result, C2STResult)
    assert recorded == {
        "steps": ["scaler", "logistic"],
        "features": (40, 195),
        "splits": 5,
        "shuffle": True,
        "random_state": 42,
        "method": "predict_proba",
    }
    assert result.probabilities.shape == (40,)
    assert result.labels.shape == (40,)
    assert result.roc_auc == 1.0
    assert result.accuracy == 1.0


def test_joint_tsne_input_has_balanced_labels_and_195_features() -> None:
    real = _windows(6, 1)
    synthetic = {"vae": _windows(6, 2), "diffusion": _windows(6, 3)}
    matrix, labels, row_ids = assemble_joint_embedding_input(real, synthetic)
    assert matrix.shape == (18, 195)
    assert {label: int(np.count_nonzero(labels == label)) for label in np.unique(labels)} == {
        "diffusion": 6,
        "real": 6,
        "vae": 6,
    }
    assert np.array_equal(row_ids[:6], np.arange(6))


def test_real_loader_rejects_nvda_reference_before_reading(tmp_path: Path) -> None:
    forbidden = tmp_path / "nvda_visible.parquet"
    with pytest.raises(ValueError, match="only donor_train and donor_validation"):
        load_canonical_windows(forbidden, expected_count=380)


def test_normalized_validation_does_not_apply_physical_sign_rules() -> None:
    windows = _windows(4)
    windows[:, :, 1] = -5.0
    windows[:, :, 2] = -10.0
    validated = validate_windows(windows, name="normalized_synthetic")
    assert np.array_equal(validated, windows)
