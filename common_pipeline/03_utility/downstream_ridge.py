"""
Common downstream Ridge, real test, and metrics (Parts 9, 10, 11).
"""
from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

build_mixtures = importlib.import_module("common_pipeline.03_utility.build_mixtures")
features_target = importlib.import_module("common_pipeline.03_utility.features_target")

n_synthetic_for_ratio = build_mixtures.n_synthetic_for_ratio
load_valid_calibrated_pool = build_mixtures.load_valid_calibrated_pool
fit_downstream_scaler = build_mixtures.fit_downstream_scaler
build_mixture = build_mixtures.build_mixture
RATIOS = build_mixtures.RATIOS
SUBSAMPLING_SEEDS = build_mixtures.SUBSAMPLING_SEEDS
N_REAL = build_mixtures.N_REAL

build_features_and_target = features_target.build_features_and_target
load_real_visible_windows = features_target.load_real_visible_windows
load_real_test_windows = features_target.load_real_test_windows

CALIBRATED_POOLS_DIR = Path("common_pipeline/03_utility/results/calibrated_pools")
RAW_OUTPUT_PATH = Path("common_pipeline/03_utility/results/tables/downstream_results_raw.csv")
SUMMARY_OUTPUT_PATH = Path("common_pipeline/03_utility/results/tables/downstream_results_summary.csv")


def evaluate_one(X_train_raw: np.ndarray, y_train: np.ndarray, scaler, X_test_scaled: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    """Ridge(alpha=1.0) identico para todos -- sin busqueda de hiperparametros."""
    X_train_scaled = scaler.transform(X_train_raw)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    return rmse, mae


def main() -> None:
    real_windows = load_real_visible_windows()
    scaler = fit_downstream_scaler(real_windows)

    test_windows = load_real_test_windows()
    X_test, y_test = build_features_and_target(test_windows)
    X_test_scaled = scaler.transform(X_test)
    print(f"Test set: {X_test.shape[0]} ventanas (esperado 150)")

    methods = sorted(p.stem.replace("_calibrated", "") for p in CALIBRATED_POOLS_DIR.glob("*_calibrated.npz"))
    print(f"Metodos: {methods}\n")

    raw_rows = []
    for method in methods:
        synthetic_pool = load_valid_calibrated_pool(method)

        for ratio in RATIOS:
            n_synth = n_synthetic_for_ratio(N_REAL, ratio)

            if n_synth == 0:
                # real-only: deterministico, no se repite por seed
                X_train, y_train = build_features_and_target(real_windows)
                rmse, mae = evaluate_one(X_train, y_train, scaler, X_test_scaled, y_test)
                raw_rows.append({"method": method, "ratio": ratio, "subsampling_seed": None, "rmse": rmse, "mae": mae})
                continue

            for seed in SUBSAMPLING_SEEDS:
                mix_windows = build_mixture(real_windows, synthetic_pool, ratio, seed)
                X_train, y_train = build_features_and_target(mix_windows)
                rmse, mae = evaluate_one(X_train, y_train, scaler, X_test_scaled, y_test)
                raw_rows.append({"method": method, "ratio": ratio, "subsampling_seed": seed, "rmse": rmse, "mae": mae})

    raw_df = pd.DataFrame(raw_rows)
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_OUTPUT_PATH, index=False)
    print(f"Guardado: {RAW_OUTPUT_PATH} ({len(raw_df)} filas)")

    # Agregacion: media/std entre seeds para ratio>0; real-only pasa tal cual (1 solo valor, std=0)
    summary_rows = []
    for method in methods:
        real_only = raw_df[(raw_df["method"] == method) & (raw_df["ratio"] == 0.0)].iloc[0]
        base_rmse, base_mae = real_only["rmse"], real_only["mae"]

        for ratio in RATIOS:
            subset = raw_df[(raw_df["method"] == method) & (raw_df["ratio"] == ratio)]
            mean_rmse, std_rmse = subset["rmse"].mean(), subset["rmse"].std(ddof=0) if len(subset) > 1 else 0.0
            mean_mae, std_mae = subset["mae"].mean(), subset["mae"].std(ddof=0) if len(subset) > 1 else 0.0
            summary_rows.append({
                "method": method, "ratio": ratio,
                "mean_rmse": mean_rmse, "std_rmse": std_rmse,
                "mean_mae": mean_mae, "std_mae": std_mae,
                "delta_rmse_vs_real_only": mean_rmse - base_rmse,
                "delta_rmse_pct_vs_real_only": (mean_rmse - base_rmse) / base_rmse * 100,
                "delta_mae_vs_real_only": mean_mae - base_mae,
                "delta_mae_pct_vs_real_only": (mean_mae - base_mae) / base_mae * 100,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print(f"Guardado: {SUMMARY_OUTPUT_PATH}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
