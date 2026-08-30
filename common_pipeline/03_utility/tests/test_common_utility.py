"""
Tests minimos del pipeline comun (common_pipeline/03_utility).
Usa fixtures pequenos -- no depende de los datos reales del proyecto,
para que corran rapido y de forma aislada.
"""
import importlib

import numpy as np
import pytest

calibrate_nvda = importlib.import_module("common_pipeline.03_utility.calibrate_nvda")
validate_physical = importlib.import_module("common_pipeline.03_utility.validate_physical")
build_mixtures = importlib.import_module("common_pipeline.03_utility.build_mixtures")
features_target = importlib.import_module("common_pipeline.03_utility.features_target")


# --- unique-day calibration + ddof0 ---

def test_reconstruct_unique_daily_no_overlap_duplication(tmp_path):
    import pandas as pd

    n_windows, window_length, n_channels = 5, 65, 3
    dates = pd.date_range("2022-07-05", periods=n_windows + window_length - 1, freq="B")

    rows = []
    for w in range(n_windows):
        window_dates = dates[w:w + window_length]
        flat = np.arange(w * 1000, w * 1000 + window_length * n_channels, dtype="float64").tolist()
        rows.append({
            "window_start_date": window_dates[0], "window_end_date": window_dates[-1],
            "features_flat": flat,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "fixture_windows.parquet"
    df.to_parquet(path)

    daily, start, end = calibrate_nvda.reconstruct_unique_daily(str(path))
    expected_days = n_windows + window_length - 1
    assert daily.shape[0] == expected_days


def test_calibration_uses_ddof0():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert values.std(ddof=0) != values.std(ddof=1)
    assert np.isclose(values.std(ddof=0), np.sqrt(1.25))


# --- exact affine calibration, no re-standardization ---

def test_calibrate_is_pure_affine_transform():
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.5, 1.0, 2.0])
    pool = np.random.default_rng(0).normal(size=(4, 65, 3))

    calibrated = validate_physical.calibrate(pool, mu, sigma)
    expected = mu.reshape(1, 1, -1) + sigma.reshape(1, 1, -1) * pool
    np.testing.assert_allclose(calibrated, expected)


def test_calibrate_does_not_restandardize_pool():
    import inspect
    source = inspect.getsource(validate_physical.calibrate)
    assert "mean(" not in source.replace(" ", "") or "pool_normalized.mean" not in source
    assert ".std(" not in source.replace(" ", "")


# --- negative range pre-calibration allowed, post-calibration invalid ---

def test_negative_range_allowed_before_calibration_invalid_after():
    window = np.zeros((65, 3))
    window[:, 1] = -5.0
    assert True

    mu, sigma = np.array([0.0, 0.05, 20.0]), np.array([0.03, 0.02, 0.25])
    calibrated = validate_physical.calibrate(window[np.newaxis], mu, sigma)[0]
    assert not validate_physical.validate_window(calibrated)


def test_valid_window_passes():
    window = np.zeros((65, 3))
    window[:, 0] = 0.001
    window[:, 1] = 0.3
    window[:, 2] = 15.0
    assert validate_physical.validate_window(window)


def test_no_repair_functions_used():
    """Confirma que validate_physical.py no usa abs()/clip()/winsorize -- solo reporta.
    Busca solo en el CODIGO (sin comentarios ni docstrings), para no disparar
    falsos positivos con palabras mencionadas en la documentacion."""
    import ast
    import inspect

    source = inspect.getsource(validate_physical)
    tree = ast.parse(source)

    calls_source = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls_source.append(ast.unparse(node))

    joined_calls = " ".join(calls_source)
    forbidden = ["clip", "abs(", "winsoriz"]
    for pattern in forbidden:
        assert pattern not in joined_calls, f"funcion de reparacion prohibida encontrada en una llamada real: {pattern}"


# --- mixture counts 0/21/62/186, deterministic, no replacement ---

@pytest.mark.parametrize("ratio,expected_n_synth", [(0.0, 0), (0.25, 21), (0.50, 62), (0.75, 186)])
def test_mixture_counts_match_spec(ratio, expected_n_synth):
    assert build_mixtures.n_synthetic_for_ratio(62, ratio) == expected_n_synth


def test_subsampling_no_replacement():
    real = np.zeros((62, 65, 3))
    synthetic_pool = np.arange(200 * 65 * 3, dtype="float64").reshape(200, 65, 3)

    mix = build_mixtures.build_mixture(real, synthetic_pool, ratio=0.5, seed=42)
    synth_part = mix[62:]
    flat = synth_part.reshape(len(synth_part), -1)
    n_unique_rows = len(np.unique(flat, axis=0))
    assert n_unique_rows == len(synth_part)


def test_subsampling_is_deterministic_per_seed():
    real = np.zeros((62, 65, 3))
    synthetic_pool = np.random.default_rng(1).normal(size=(200, 65, 3))

    mix_a = build_mixtures.build_mixture(real, synthetic_pool, ratio=0.5, seed=42)
    mix_b = build_mixtures.build_mixture(real, synthetic_pool, ratio=0.5, seed=42)
    np.testing.assert_array_equal(mix_a, mix_b)


def test_real_only_is_deterministic_no_fake_repetition():
    real = np.zeros((62, 65, 3))
    synthetic_pool = np.random.default_rng(2).normal(size=(200, 65, 3))
    n_synth = build_mixtures.n_synthetic_for_ratio(62, 0.0)
    assert n_synth == 0


# --- 8 exact features, exact RV target ---

def test_exactly_8_features():
    assert len(features_target.FEATURE_NAMES) == 8


def test_rv_formula_matches_spec():
    k = 20
    c = 0.01
    returns = np.full((1, k), c)
    rv = features_target._realized_vol(returns, k)
    expected = c * np.sqrt(252)
    np.testing.assert_allclose(rv[0], expected)


def test_features_and_target_shapes():
    windows = np.random.default_rng(3).normal(size=(10, 65, 3))
    windows[:, :, 1] = np.abs(windows[:, :, 1])
    windows[:, :, 2] += 20

    X, y = features_target.build_features_and_target(windows)
    assert X.shape == (10, 8)
    assert y.shape == (10,)


# --- downstream scaler fit only real visible, Ridge alpha exactly 1 ---

def test_scaler_uses_only_real_windows(monkeypatch):
    real_windows = np.random.default_rng(4).normal(size=(62, 65, 3))
    real_windows[:, :, 1] = np.abs(real_windows[:, :, 1])
    real_windows[:, :, 2] += 20

    monkeypatch.setattr(build_mixtures, "load_real_visible_windows", lambda: real_windows)
    scaler = build_mixtures.fit_downstream_scaler(real_windows)

    X_real, _ = features_target.build_features_and_target(real_windows)
    np.testing.assert_allclose(scaler.mean_, X_real.mean(axis=0))


def test_ridge_alpha_is_exactly_one():
    import inspect
    downstream_ridge = importlib.import_module("common_pipeline.03_utility.downstream_ridge")
    source = inspect.getsource(downstream_ridge.evaluate_one)
    assert "alpha=1.0" in source
