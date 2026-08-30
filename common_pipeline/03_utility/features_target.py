"""
Common feature engineering + target (Parts 3, 4, 5).

8 features EXACTAS de las primeras 60 sesiones, target = RV anualizada de
las ultimas 5. Misma formula para real y para cualquier metodo sintetico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION = 252
CONTEXT_LENGTH = 60
HORIZON_LENGTH = 5

FEATURE_NAMES = (
    "rv5", "rv20", "rv60",
    "mean_abs_return20", "momentum20",
    "mean_range20", "mean_log_volume20", "std_log_volume20",
)

IDX_RETURN, IDX_RANGE, IDX_VOLUME = 0, 1, 2


def _realized_vol(returns: np.ndarray, k: int) -> np.ndarray:
    """rv_k = sqrt((252/k) * sum(ultimos k log_returns^2))"""
    last_k = returns[:, -k:]
    return np.sqrt((ANNUALIZATION / k) * np.sum(last_k ** 2, axis=-1))


def build_features_and_target(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """windows: [N, 65, 3]. Devuelve X [N, 8], y [N]."""
    context_return = windows[:, :CONTEXT_LENGTH, IDX_RETURN]
    context_range = windows[:, :CONTEXT_LENGTH, IDX_RANGE]
    context_volume = windows[:, :CONTEXT_LENGTH, IDX_VOLUME]
    future_return = windows[:, CONTEXT_LENGTH:, IDX_RETURN]

    X = np.stack([
        _realized_vol(context_return, 5),
        _realized_vol(context_return, 20),
        _realized_vol(context_return, 60),
        np.mean(np.abs(context_return[:, -20:]), axis=-1),          # mean_abs_return20
        np.sum(context_return[:, -20:], axis=-1),                    # momentum20
        np.mean(context_range[:, -20:], axis=-1),                    # mean_range20
        np.mean(context_volume[:, -20:], axis=-1),                   # mean_log_volume20
        np.std(context_volume[:, -20:], axis=-1, ddof=0),            # std_log_volume20
    ], axis=-1).astype("float64")

    y = _realized_vol(future_return, HORIZON_LENGTH).astype("float64")
    return X, y


def load_real_visible_windows() -> np.ndarray:
    """Part 3: definicion canonica de las 62 ventanas reales visibles.
    NO se regenera ninguna definicion alternativa."""
    df = pd.read_parquet("data/features/windows/nvda_visible.parquet")
    windows = np.stack([
        np.asarray(row, dtype="float64").reshape(65, 3) for row in df["features_flat"]
    ])
    assert len(windows) == 62, f"esperadas 62 ventanas reales visibles, encontradas {len(windows)}"
    return windows


def load_real_test_windows() -> np.ndarray:
    """Part 10: test canonico, 150 ventanas, stride=5."""
    df = pd.read_parquet("data/features/test_index.parquet")
    windows = np.stack([
        np.asarray(row, dtype="float64").reshape(65, 3) for row in df["features_flat"]
    ])
    assert len(windows) == 150, f"esperadas 150 ventanas de test, encontradas {len(windows)}"
    return windows


if __name__ == "__main__":
    real_windows = load_real_visible_windows()
    X, y = build_features_and_target(real_windows)
    print("X real visible:", X.shape, "| y:", y.shape)
    print(pd.DataFrame(X, columns=FEATURE_NAMES).describe().to_string())

    test_windows = load_real_test_windows()
    X_test, y_test = build_features_and_target(test_windows)
    print("\nX test:", X_test.shape, "| y_test:", y_test.shape)
