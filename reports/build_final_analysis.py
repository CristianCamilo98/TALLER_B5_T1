"""Build the final-quantitative-reporting layer from the STRICT_FINAL snapshot.

Reads exclusively from ``artifacts/final/strict_final_20260902/**`` (the
tracked, byte-exact copy of the frozen scientific results) and writes
derived, purely descriptive CSV tables under ``reports/final_analysis/``.

This script does not run any part of 01_contract/02_fidelity/03_utility, does
not retrain or recompute any model, and does not modify anything under
``artifacts/final/strict_final_20260902/``. Every number here is a
deterministic transformation (rename, join, mean/std/min/max, Pearson/
Spearman correlation) of numbers that already exist in the STRICT_FINAL CSVs.

"Best" and "most stable" below are post-hoc descriptive labels for reporting
only -- they are not used to select, tune, or retrain anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "artifacts" / "final" / "strict_final_20260902"
UTILITY_TABLES = SNAPSHOT / "utility" / "tables"
FIDELITY_TABLES = SNAPSHOT / "fidelity" / "tables"
OUT_DIR = Path(__file__).resolve().parent / "final_analysis"

# Internal method_id -> presentation label. method_id is preserved as a
# column in every output table; it must never be used as the primary label
# shown to a reader.
METHOD_NAME = {
    "bootstrap_jitter": "Bootstrap + Jitter",
    "cristian": "WGAN-GP",
    "daniel": "DDPM",
    "marco": "TimeVAE",
    "david": "Normalizing Flow",
}
METHOD_ORDER = ["bootstrap_jitter", "cristian", "daniel", "marco", "david"]
NEURAL_METHOD_IDS = ["cristian", "daniel", "marco", "david"]
BASELINE_METHOD_ID = "bootstrap_jitter"
RATIOS = [0.25, 0.50, 0.75]
CHANNELS = ["log_return", "log_high_low_range", "log1p_volume"]


def method_name(method_id: str) -> str:
    return METHOD_NAME[method_id]


def _method_sort_key(method_id: str) -> int:
    return METHOD_ORDER.index(method_id)


def load_downstream_raw() -> pd.DataFrame:
    return pd.read_csv(UTILITY_TABLES / "downstream_results_raw.csv")


def load_downstream_summary() -> pd.DataFrame:
    return pd.read_csv(UTILITY_TABLES / "downstream_results_summary.csv")


def real_only_reference(summary: pd.DataFrame) -> tuple[float, float]:
    """Read REAL_ONLY RMSE/MAE directly from the STRICT_FINAL CSV (never hardcoded)."""

    real_rows = summary[summary["method"] == "REAL_ONLY"]
    if len(real_rows) != 1:
        raise ValueError(
            f"Expected exactly one REAL_ONLY row in downstream_results_summary.csv, found {len(real_rows)}"
        )
    row = real_rows.iloc[0]
    return float(row["mean_rmse"]), float(row["mean_mae"])


def build_master_utility_table(summary: pd.DataFrame, real_only_rmse: float, real_only_mae: float) -> pd.DataFrame:
    rows = summary[summary["method"] != "REAL_ONLY"].copy()
    if len(rows) != 15:
        raise ValueError(f"Expected 15 method x ratio rows, found {len(rows)}")

    rows["method_id"] = rows["method"]
    rows["method_name"] = rows["method_id"].map(method_name)
    rows["synthetic_ratio"] = rows["ratio"]
    rows["rmse_mean"] = rows["mean_rmse"]
    rows["rmse_std"] = rows["std_rmse"]
    rows["mae_mean"] = rows["mean_mae"]
    rows["mae_std"] = rows["std_mae"]
    rows["real_only_rmse"] = real_only_rmse
    rows["real_only_mae"] = real_only_mae
    rows["rmse_improvement_abs"] = real_only_rmse - rows["rmse_mean"]
    rows["rmse_improvement_pct"] = (real_only_rmse - rows["rmse_mean"]) / real_only_rmse * 100.0
    rows["mae_improvement_abs"] = real_only_mae - rows["mae_mean"]
    rows["mae_improvement_pct"] = (real_only_mae - rows["mae_mean"]) / real_only_mae * 100.0

    rows["_order"] = rows["method_id"].map(_method_sort_key)
    rows = rows.sort_values(["_order", "synthetic_ratio"]).drop(columns=["_order"])

    columns = [
        "method_id",
        "method_name",
        "synthetic_ratio",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "real_only_rmse",
        "real_only_mae",
        "rmse_improvement_abs",
        "rmse_improvement_pct",
        "mae_improvement_abs",
        "mae_improvement_pct",
    ]
    return rows[columns].reset_index(drop=True)


def _population_std(series: pd.Series) -> float:
    """ddof=0, matching downstream_ridge.py's own std convention (population std)."""

    return float(series.std(ddof=0))


def build_seed_stability(raw: pd.DataFrame) -> pd.DataFrame:
    rows = raw[raw["method"] != "REAL_ONLY"].copy()
    grouped = rows.groupby(["method", "ratio"], as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", _population_std),
        rmse_min=("rmse", "min"),
        rmse_max=("rmse", "max"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", _population_std),
        mae_min=("mae", "min"),
        mae_max=("mae", "max"),
        n_seeds=("subsampling_seed", "count"),
    )
    if len(grouped) != 15:
        raise ValueError(f"Expected 15 method x ratio groups, found {len(grouped)}")
    if not (grouped["n_seeds"] == 3).all():
        raise ValueError("Expected exactly 3 seeds per method x ratio")

    grouped["method_id"] = grouped["method"]
    grouped["method_name"] = grouped["method_id"].map(method_name)
    grouped["synthetic_ratio"] = grouped["ratio"]
    grouped["rmse_range"] = grouped["rmse_max"] - grouped["rmse_min"]
    grouped["rmse_cv"] = grouped["rmse_std"] / grouped["rmse_mean"]
    grouped["mae_range"] = grouped["mae_max"] - grouped["mae_min"]

    grouped["_order"] = grouped["method_id"].map(_method_sort_key)
    grouped = grouped.sort_values(["_order", "synthetic_ratio"]).drop(columns=["_order"])

    columns = [
        "method_id",
        "method_name",
        "synthetic_ratio",
        "rmse_mean",
        "rmse_std",
        "rmse_min",
        "rmse_max",
        "rmse_range",
        "rmse_cv",
        "mae_mean",
        "mae_std",
        "mae_min",
        "mae_max",
        "mae_range",
        "n_seeds",
    ]
    return grouped[columns].reset_index(drop=True)


def build_model_summary(master: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method_id in METHOD_ORDER:
        sub = master[master["method_id"] == method_id]
        stab_sub = stability[stability["method_id"] == method_id]

        best_rmse_row = sub.loc[sub["rmse_mean"].idxmin()]
        best_mae_row = sub.loc[sub["mae_mean"].idxmin()]
        most_stable_row = stab_sub.loc[stab_sub["rmse_std"].idxmin()]

        records.append(
            {
                "method_id": method_id,
                "method_name": method_name(method_id),
                "best_rmse_ratio": float(best_rmse_row["synthetic_ratio"]),
                "best_rmse_mean": float(best_rmse_row["rmse_mean"]),
                "best_rmse_std": float(best_rmse_row["rmse_std"]),
                "best_rmse_improvement_pct": float(best_rmse_row["rmse_improvement_pct"]),
                "best_mae_ratio": float(best_mae_row["synthetic_ratio"]),
                "best_mae_mean": float(best_mae_row["mae_mean"]),
                "best_mae_improvement_pct": float(best_mae_row["mae_improvement_pct"]),
                "mean_rmse_across_ratios": float(sub["rmse_mean"].mean()),
                "mean_rmse_improvement_across_ratios": float(sub["rmse_improvement_pct"].mean()),
                "most_stable_ratio_by_rmse_std": float(most_stable_row["synthetic_ratio"]),
                "post_hoc_description_only": True,
            }
        )
    return pd.DataFrame.from_records(records)


def build_baseline_vs_neural(master: pd.DataFrame) -> pd.DataFrame:
    baseline = master[master["method_id"] == BASELINE_METHOD_ID].set_index("synthetic_ratio")
    records = []
    for method_id in NEURAL_METHOD_IDS:
        neural = master[master["method_id"] == method_id].set_index("synthetic_ratio")
        for ratio in RATIOS:
            b = baseline.loc[ratio]
            n = neural.loc[ratio]
            records.append(
                {
                    "synthetic_ratio": ratio,
                    "neural_method_id": method_id,
                    "neural_method_name": method_name(method_id),
                    "baseline_method_id": BASELINE_METHOD_ID,
                    "baseline_method_name": method_name(BASELINE_METHOD_ID),
                    "baseline_rmse_mean": float(b["rmse_mean"]),
                    "neural_rmse_mean": float(n["rmse_mean"]),
                    "rmse_difference_vs_baseline": float(b["rmse_mean"] - n["rmse_mean"]),
                    "baseline_mae_mean": float(b["mae_mean"]),
                    "neural_mae_mean": float(n["mae_mean"]),
                    "mae_difference_vs_baseline": float(b["mae_mean"] - n["mae_mean"]),
                    "baseline_rmse_improvement_pct": float(b["rmse_improvement_pct"]),
                    "neural_rmse_improvement_pct": float(n["rmse_improvement_pct"]),
                    "rmse_improvement_pct_difference_vs_baseline": float(
                        n["rmse_improvement_pct"] - b["rmse_improvement_pct"]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def build_fidelity_master() -> pd.DataFrame:
    wasserstein = pd.read_csv(FIDELITY_TABLES / "wasserstein.csv")
    correlation_errors = pd.read_csv(FIDELITY_TABLES / "correlation_errors.csv").set_index("method")
    c2st = pd.read_csv(FIDELITY_TABLES / "c2st_results.csv").set_index("method")
    nearest_neighbor = pd.read_csv(FIDELITY_TABLES / "nearest_neighbor.csv").set_index("method")
    return_acf = pd.read_csv(FIDELITY_TABLES / "return_acf.csv")
    abs_return_acf = pd.read_csv(FIDELITY_TABLES / "abs_return_acf.csv")

    wasserstein_wide = wasserstein.pivot(index="method", columns="channel", values="wasserstein_1")
    return_acf_mae = return_acf.groupby("method")["acf_mae"].mean()
    abs_return_acf_mae = abs_return_acf.groupby("method")["acf_mae"].mean()

    records = []
    for method_id in METHOD_ORDER:
        records.append(
            {
                "method_id": method_id,
                "method_name": method_name(method_id),
                "c2st_roc_auc": float(c2st.loc[method_id, "roc_auc"]),
                "mean_correlation_error": float(correlation_errors.loc[method_id, "mean_absolute_off_diagonal_difference"]),
                "nearest_neighbor_mean": float(nearest_neighbor.loc[method_id, "mean"]),
                "nearest_neighbor_median": float(nearest_neighbor.loc[method_id, "median"]),
                "wasserstein_log_return": float(wasserstein_wide.loc[method_id, "log_return"]),
                "wasserstein_log_high_low_range": float(wasserstein_wide.loc[method_id, "log_high_low_range"]),
                "wasserstein_log1p_volume": float(wasserstein_wide.loc[method_id, "log1p_volume"]),
                "return_acf_mae_vs_real": float(return_acf_mae.loc[method_id]),
                "abs_return_acf_mae_vs_real": float(abs_return_acf_mae.loc[method_id]),
            }
        )
    return pd.DataFrame.from_records(records)


FIDELITY_METRICS = [
    "c2st_roc_auc",
    "mean_correlation_error",
    "nearest_neighbor_mean",
    "wasserstein_log_return",
    "wasserstein_log_high_low_range",
    "wasserstein_log1p_volume",
    "return_acf_mae_vs_real",
    "abs_return_acf_mae_vs_real",
]
UTILITY_TARGETS = ["best_rmse_improvement_pct", "mean_rmse_improvement_across_ratios"]


def build_fidelity_vs_utility(
    fidelity_master: pd.DataFrame, master: pd.DataFrame, model_summary: pd.DataFrame
) -> pd.DataFrame:
    ratio_pivot = master.pivot(index="method_id", columns="synthetic_ratio", values="rmse_improvement_pct")
    ratio_pivot = ratio_pivot.rename(
        columns={0.25: "rmse_improvement_25", 0.50: "rmse_improvement_50", 0.75: "rmse_improvement_75"}
    )

    merged = fidelity_master.merge(model_summary, on=["method_id", "method_name"], how="inner")
    merged = merged.merge(ratio_pivot, on="method_id", how="inner")
    if len(merged) != 5:
        raise ValueError(f"Expected 5 methods in fidelity_vs_utility, found {len(merged)}")

    columns = (
        ["method_id", "method_name"]
        + FIDELITY_METRICS
        + [
            "rmse_improvement_25",
            "rmse_improvement_50",
            "rmse_improvement_75",
            "best_rmse_improvement_pct",
            "mean_rmse_improvement_across_ratios",
        ]
    )
    return merged[columns].reset_index(drop=True)


def build_fidelity_utility_correlations(fidelity_vs_utility: pd.DataFrame) -> pd.DataFrame:
    n = len(fidelity_vs_utility)
    records = []
    for metric in FIDELITY_METRICS:
        for target in UTILITY_TARGETS:
            x = fidelity_vs_utility[metric].to_numpy(dtype=float)
            y = fidelity_vs_utility[target].to_numpy(dtype=float)
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_rho, spearman_p = stats.spearmanr(x, y)
            records.append(
                {
                    "fidelity_metric": metric,
                    "utility_target": target,
                    "n": n,
                    "pearson_r": float(pearson_r),
                    "pearson_p_value": float(pearson_p),
                    "spearman_rho": float(spearman_rho),
                    "spearman_p_value": float(spearman_p),
                    "exploratory_only": True,
                    "statistical_inference": False,
                }
            )
    return pd.DataFrame.from_records(records)


def build_sanity_check(master: pd.DataFrame, real_only_rmse: float) -> pd.DataFrame:
    rows = master.copy()
    rows["real_only_rmse_check"] = real_only_rmse
    rows["synthetic_rmse"] = rows["rmse_mean"]
    rows["rmse_ratio_vs_real_only"] = rows["synthetic_rmse"] / real_only_rmse
    rows["cross_check_improvement_pct"] = (1.0 - rows["rmse_ratio_vs_real_only"]) * 100.0
    rows["matches_master_table"] = np.isclose(
        rows["cross_check_improvement_pct"], rows["rmse_improvement_pct"], atol=1e-9
    )

    columns = [
        "method_id",
        "method_name",
        "synthetic_ratio",
        "real_only_rmse_check",
        "synthetic_rmse",
        "rmse_ratio_vs_real_only",
        "rmse_improvement_pct",
        "cross_check_improvement_pct",
        "matches_master_table",
    ]
    return rows[columns].reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_downstream_summary()
    raw = load_downstream_raw()
    real_only_rmse, real_only_mae = real_only_reference(summary)

    master = build_master_utility_table(summary, real_only_rmse, real_only_mae)
    stability = build_seed_stability(raw)
    model_summary = build_model_summary(master, stability)
    baseline_vs_neural = build_baseline_vs_neural(master)
    fidelity_master = build_fidelity_master()
    fidelity_vs_utility = build_fidelity_vs_utility(fidelity_master, master, model_summary)
    correlations = build_fidelity_utility_correlations(fidelity_vs_utility)
    sanity_check = build_sanity_check(master, real_only_rmse)

    # Structural sanity checks (arithmetic, not scientific).
    if len(raw[raw["method"] == "REAL_ONLY"]) != 1:
        raise ValueError("Duplicate or missing REAL_ONLY row in downstream_results_raw.csv")
    if not sanity_check["matches_master_table"].all():
        raise ValueError("Sanity check formula mismatch detected against master_utility_table")
    numeric_cols_to_check = [
        c
        for c in master.columns
        if c not in {"method_id", "method_name"}
    ]
    if master[numeric_cols_to_check].isna().any().any():
        raise ValueError("Unexpected NaN in master_utility_table")

    outputs = {
        "master_utility_table.csv": master,
        "seed_stability.csv": stability,
        "model_summary.csv": model_summary,
        "baseline_vs_neural.csv": baseline_vs_neural,
        "fidelity_master.csv": fidelity_master,
        "fidelity_vs_utility.csv": fidelity_vs_utility,
        "fidelity_utility_correlations.csv": correlations,
        "sanity_check_improvements.csv": sanity_check,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT_DIR / filename, index=False, float_format="%.12g")
        print(f"wrote {OUT_DIR / filename} ({len(frame)} rows)")

    print(f"\nREAL_ONLY rmse={real_only_rmse:.6f} mae={real_only_mae:.6f}")
    print("Sanity check: all rows match master table formula ->", bool(sanity_check["matches_master_table"].all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
