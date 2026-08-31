"""
Common downstream Ridge, real test, and metrics (Parts 9, 10, 11).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

build_mixtures = importlib.import_module("common_pipeline.03_utility.build_mixtures")
features_target = importlib.import_module("common_pipeline.03_utility.features_target")
utility_run = importlib.import_module("common_pipeline.03_utility.utility_run")

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

REAL_ONLY_METHOD = "REAL_ONLY"


def evaluate_one(X_train_raw: np.ndarray, y_train: np.ndarray, scaler, X_test_scaled: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    """Ridge(alpha=1.0) identico para todos -- sin busqueda de hiperparametros."""
    X_train_scaled = scaler.transform(X_train_raw)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    return rmse, mae


def build_result_tables(
    real_windows: np.ndarray,
    test_windows: np.ndarray,
    synthetic_pools: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = fit_downstream_scaler(real_windows)
    X_test, y_test = build_features_and_target(test_windows)
    X_test_scaled = scaler.transform(X_test)

    # Real-only is one experiment, not one repeated row per synthetic method.
    X_real, y_real = build_features_and_target(real_windows)
    real_rmse, real_mae = evaluate_one(X_real, y_real, scaler, X_test_scaled, y_test)
    raw_rows = [
        {
            "method": REAL_ONLY_METHOD,
            "ratio": 0.0,
            "subsampling_seed": None,
            "rmse": real_rmse,
            "mae": real_mae,
        }
    ]
    for method, synthetic_pool in sorted(synthetic_pools.items()):
        for ratio in (value for value in RATIOS if value > 0):
            for seed in SUBSAMPLING_SEEDS:
                mix_windows = build_mixture(real_windows, synthetic_pool, ratio, seed)
                X_train, y_train = build_features_and_target(mix_windows)
                rmse, mae = evaluate_one(X_train, y_train, scaler, X_test_scaled, y_test)
                raw_rows.append(
                    {
                        "method": method,
                        "ratio": ratio,
                        "subsampling_seed": seed,
                        "rmse": rmse,
                        "mae": mae,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    summary_rows = [
        {
            "method": REAL_ONLY_METHOD,
            "ratio": 0.0,
            "mean_rmse": real_rmse,
            "std_rmse": 0.0,
            "mean_mae": real_mae,
            "std_mae": 0.0,
            "delta_rmse_vs_real_only": 0.0,
            "delta_rmse_pct_vs_real_only": 0.0,
            "delta_mae_vs_real_only": 0.0,
            "delta_mae_pct_vs_real_only": 0.0,
        }
    ]
    for method in sorted(synthetic_pools):
        for ratio in (value for value in RATIOS if value > 0):
            subset = raw_df[(raw_df["method"] == method) & (raw_df["ratio"] == ratio)]
            mean_rmse = subset["rmse"].mean()
            std_rmse = subset["rmse"].std(ddof=0)
            mean_mae = subset["mae"].mean()
            std_mae = subset["mae"].std(ddof=0)
            summary_rows.append(
                {
                    "method": method,
                    "ratio": ratio,
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse,
                    "mean_mae": mean_mae,
                    "std_mae": std_mae,
                    "delta_rmse_vs_real_only": mean_rmse - real_rmse,
                    "delta_rmse_pct_vs_real_only": (mean_rmse - real_rmse) / real_rmse * 100,
                    "delta_mae_vs_real_only": mean_mae - real_mae,
                    "delta_mae_pct_vs_real_only": (mean_mae - real_mae) / real_mae * 100,
                }
            )
    return raw_df, pd.DataFrame(summary_rows)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=utility_run.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--results-root", type=Path, default=utility_run.RESULTS_ROOT)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = utility_run.existing_run_dir(
        results_root=args.results_root,
        registry_path=args.registry_path,
        run_id=args.run_id,
    )
    real_windows = load_real_visible_windows()
    test_windows = load_real_test_windows()
    manifest = utility_run.load_run(run_dir)
    methods = sorted(manifest.get("calibrated_pools", {}))
    synthetic_pools = {
        method: load_valid_calibrated_pool(method, run_dir=run_dir)
        for method in methods
    }
    raw_df, summary_df = build_result_tables(real_windows, test_windows, synthetic_pools)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = tables_dir / "downstream_results_raw.csv"
    summary_output_path = tables_dir / "downstream_results_summary.csv"
    raw_df.to_csv(raw_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)
    utility_run.update_run_manifest(
        run_dir,
        {
            "downstream_results_raw": utility_run.run_relative_path(run_dir, raw_output_path),
            "downstream_results_summary": utility_run.run_relative_path(run_dir, summary_output_path),
            "real_only": {
                "method_id": REAL_ONLY_METHOD,
                "rmse": float(raw_df.iloc[0]["rmse"]),
                "mae": float(raw_df.iloc[0]["mae"]),
            },
        },
    )
    print(f"Guardado: {raw_output_path} ({len(raw_df)} filas)")
    print(f"Guardado: {summary_output_path}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
