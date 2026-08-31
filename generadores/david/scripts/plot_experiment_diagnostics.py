#!/usr/bin/env python3
"""Plot David experiment diagnostics.

The figures are explanatory artifacts: they do not change the official output.
They help show why ``temporal_jitter_0p40_rho0p85`` was selected and where the
tradeoffs start to become risky.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FIDELITY_MODULE = REPO_ROOT / "common_pipeline" / "02_fidelity"
for module_path in (REPO_ROOT, FIDELITY_MODULE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from fidelity_core import (  # noqa: E402
    CHANNEL_ORDER,
    WINDOW_LENGTH,
    load_real_reference,
    mean_window_acf,
    reconstruct_windows,
)

validate_physical = importlib.import_module("common_pipeline.03_utility.validate_physical")

EXPERIMENT_ROOT = REPO_ROOT / "generadores" / "david" / "experiments"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
FIGURES_DIR = EXPERIMENT_ROOT / "figures"
EXPERIMENT_OUTPUTS_DIR = EXPERIMENT_ROOT / "outputs"
OFFICIAL_DAVID_PATH = (
    REPO_ROOT / "generadores" / "david" / "outputs" / "bootstrap_jitter_seed42_normalized.parquet"
)
COMMON_BOOTSTRAP_PATH = (
    REPO_ROOT / "common_pipeline" / "01_contract" / "outputs" / "bootstrap_jitter_seed42_normalized.parquet"
)
SELECTED_METHOD = "temporal_jitter_0p40_rho0p85"
BASELINE_METHOD = "baseline_jitter_0p05"

FAMILY_LABELS = {
    "bootstrap_independent": "Bootstrap + ruido blanco",
    "bootstrap_temporal": "Bootstrap + ruido temporal",
    "regime_bootstrap_temporal": "Regimen + ruido temporal",
    "regime_neighbor_mixup_temporal": "Regimen + mixup",
    "pca_gmm": "PCA + GMM",
}

FAMILY_COLORS = {
    "bootstrap_independent": "#4c78a8",
    "bootstrap_temporal": "#f58518",
    "regime_bootstrap_temporal": "#54a24b",
    "regime_neighbor_mixup_temporal": "#b279a2",
    "pca_gmm": "#e45756",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_features_flat(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path, columns=["features_flat"])
    return reconstruct_windows(frame["features_flat"], name=path.name).astype(np.float64, copy=False)


def parse_noise_and_rho(method: str) -> tuple[float | None, float | None]:
    match = re.search(r"jitter_(?P<noise>\d+p?\d*)_rho(?P<rho>\d+p?\d*)", method)
    if not match:
        return None, None
    noise = float(match.group("noise").replace("p", "."))
    rho = float(match.group("rho").replace("p", "."))
    return noise, rho


def _annotate_key_points(axis: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    annotations = (
        (BASELINE_METHOD, "baseline 0.05", (6, 8), "left"),
        (SELECTED_METHOD, "seleccionado 0.40", (8, -16), "left"),
        ("temporal_jitter_0p50_rho0p85", "0.50: invalida 0.48%", (8, 8), "left"),
        ("pca_gmm_32c12", "PCA-GMM", (8, 8), "left"),
    )
    for method, label, offset, horizontal_alignment in annotations:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        row = subset.iloc[0]
        axis.annotate(
            label,
            xy=(row[x_col], row[y_col]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha=horizontal_alignment,
        )


def plot_pareto(comparison: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(9, 6))
    for family, subset in comparison.groupby("family"):
        axis.scatter(
            subset["c2st_auc"],
            subset["best_mean_rmse"],
            s=50 + subset["invalid_rate"] * 3500,
            color=FAMILY_COLORS.get(family, "#777777"),
            label=FAMILY_LABELS.get(family, family),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.7,
        )

    selected = comparison[comparison["method"] == SELECTED_METHOD]
    if not selected.empty:
        axis.scatter(
            selected["c2st_auc"],
            selected["best_mean_rmse"],
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
    _annotate_key_points(axis, comparison, "c2st_auc", "best_mean_rmse")
    axis.set_title("Tradeoff fidelity/utility de las variantes de David")
    axis.set_xlabel("C2ST AUC contra donor_validation (menor, hacia 0.5, es mejor)")
    axis.set_ylabel("Mejor RMSE downstream (menor es mejor)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    return _save(fig, figures_dir / "candidate_pareto_c2st_rmse.png")


def plot_temporal_sweep(comparison: pd.DataFrame, figures_dir: Path) -> Path:
    sweep = comparison[comparison["family"].eq("bootstrap_temporal")].copy()
    parsed = sweep["method"].map(parse_noise_and_rho)
    sweep["noise_scale"] = [item[0] for item in parsed]
    sweep["rho"] = [item[1] for item in parsed]
    sweep = sweep[sweep["rho"].eq(0.85)].sort_values("noise_scale")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = (
        ("c2st_auc", "C2ST AUC", "menor hacia 0.5"),
        ("mean_wasserstein", "Wasserstein medio", "menor"),
        ("best_mean_rmse", "Mejor RMSE downstream", "menor"),
        ("return_acf_mae", "Return ACF MAE", "menor"),
    )
    for axis, (column, title, hint) in zip(axes.ravel(), panels):
        axis.plot(sweep["noise_scale"], sweep[column], marker="o", color="#f58518")
        axis.axvline(0.40, color="black", linestyle="--", linewidth=1, alpha=0.75)
        invalid = sweep[sweep["invalid_rate"].gt(0)]
        if not invalid.empty:
            axis.scatter(invalid["noise_scale"], invalid[column], color="#e45756", s=70, zorder=4)
        axis.set_title(f"{title} ({hint})")
        axis.set_xlabel("noise_scale")
        axis.grid(True, alpha=0.25)
    fig.suptitle("Sweep de ruido temporal AR(1), rho=0.85", y=1.02)
    return _save(fig, figures_dir / "temporal_noise_sweep.png")


def plot_utility_curves(utility: pd.DataFrame, figures_dir: Path) -> Path:
    keep = [
        BASELINE_METHOD,
        "temporal_jitter_0p20_rho0p85",
        "temporal_jitter_0p30_rho0p85",
        SELECTED_METHOD,
        "temporal_jitter_0p50_rho0p85",
        "regime_mixup_jitter_0p01_rho0p85",
        "pca_gmm_32c12",
    ]
    fig, axis = plt.subplots(figsize=(9, 6))
    for method in keep:
        subset = utility[utility["method"].eq(method)].sort_values("ratio")
        if subset.empty:
            continue
        linewidth = 2.8 if method == SELECTED_METHOD else 1.6
        alpha = 1.0 if method in (SELECTED_METHOD, BASELINE_METHOD) else 0.74
        axis.errorbar(
            subset["ratio"],
            subset["mean_rmse"],
            yerr=subset["std_rmse"],
            marker="o",
            linewidth=linewidth,
            capsize=3,
            alpha=alpha,
            label=method,
        )
    axis.set_title("Utility downstream por ratio sintetico")
    axis.set_xlabel("Ratio sintetico en train")
    axis.set_ylabel("RMSE")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    return _save(fig, figures_dir / "candidate_utility_curves.png")


def plot_marginals(real_validation: np.ndarray, figures_dir: Path) -> Path:
    baseline = _load_features_flat(COMMON_BOOTSTRAP_PATH)
    selected = _load_features_flat(OFFICIAL_DAVID_PATH)
    datasets = {
        "real_validation": real_validation,
        "baseline 0.05": baseline,
        "David 0.40 rho0.85": selected,
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for channel_index, axis in enumerate(axes):
        for label, windows in datasets.items():
            values = windows[:, :, channel_index].reshape(-1)
            axis.hist(values, bins=80, density=True, histtype="step", linewidth=1.4, label=label)
        axis.set_title(CHANNEL_ORDER[channel_index])
        axis.grid(True, alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Marginales en espacio global_channel_normalized", y=1.04)
    return _save(fig, figures_dir / "marginal_distributions_baseline_vs_selected.png")


def plot_distribution_bands(real_validation: np.ndarray, figures_dir: Path) -> Path:
    baseline = _load_features_flat(COMMON_BOOTSTRAP_PATH)
    selected = _load_features_flat(OFFICIAL_DAVID_PATH)
    datasets = {
        "real_validation": (real_validation, "#333333"),
        "baseline 0.05": (baseline, "#4c78a8"),
        "David 0.40 rho0.85": (selected, "#f58518"),
    }
    time = np.arange(WINDOW_LENGTH)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for channel_index, axis in enumerate(axes):
        for label, (windows, color) in datasets.items():
            values = windows[:, :, channel_index]
            p10, p50, p90 = np.percentile(values, [10, 50, 90], axis=0)
            axis.plot(time, p50, label=label, color=color, linewidth=1.7)
            axis.fill_between(time, p10, p90, color=color, alpha=0.11)
        axis.set_ylabel(CHANNEL_ORDER[channel_index])
        axis.grid(True, alpha=0.22)
    axes[-1].set_xlabel("Dia dentro de la ventana")
    axes[0].legend(fontsize=8, ncol=3)
    fig.suptitle("Bandas temporales p10-p90 y mediana", y=0.995)
    return _save(fig, figures_dir / "temporal_distribution_bands.png")


def plot_acf_curves(real_validation: np.ndarray, figures_dir: Path, *, absolute: bool) -> Path:
    baseline = _load_features_flat(COMMON_BOOTSTRAP_PATH)
    selected = _load_features_flat(OFFICIAL_DAVID_PATH)
    datasets = {
        "real_validation": real_validation,
        "baseline 0.05": baseline,
        "David 0.40 rho0.85": selected,
    }

    fig, axis = plt.subplots(figsize=(9, 5))
    for label, windows in datasets.items():
        acf, _ = mean_window_acf(windows, absolute=absolute)
        axis.plot(np.arange(1, len(acf) + 1), acf, marker="o", linewidth=1.7, label=label)
    axis.set_title("ACF de abs(return)" if absolute else "ACF de return")
    axis.set_xlabel("Lag")
    axis.set_ylabel("ACF media por ventana")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    filename = "acf_abs_return_curves.png" if absolute else "acf_return_curves.png"
    return _save(fig, figures_dir / filename)


def plot_physical_margin(comparison: pd.DataFrame, results_dir: Path, figures_dir: Path) -> Path:
    sweep = comparison[comparison["family"].eq("bootstrap_temporal")].copy()
    parsed = sweep["method"].map(parse_noise_and_rho)
    sweep["noise_scale"] = [item[0] for item in parsed]
    sweep["rho"] = [item[1] for item in parsed]
    sweep = sweep[sweep["rho"].eq(0.85)].sort_values("noise_scale")
    mu, sigma = validate_physical.load_calibration()

    rows = []
    for row in sweep.itertuples():
        path = EXPERIMENT_OUTPUTS_DIR / f"{row.method}_seed42_normalized.parquet"
        if not path.exists():
            continue
        windows = _load_features_flat(path)
        calibrated = validate_physical.calibrate(windows, mu, sigma)
        min_range = calibrated[:, :, 1].min(axis=1)
        min_volume = calibrated[:, :, 2].min(axis=1)
        rows.append(
            {
                "noise_scale": row.noise_scale,
                "invalid_rate": row.invalid_rate,
                "range_p01": np.percentile(min_range, 1),
                "range_p05": np.percentile(min_range, 5),
                "volume_p01": np.percentile(min_volume, 1),
                "volume_p05": np.percentile(min_volume, 5),
            }
        )
    margins = pd.DataFrame(rows)
    margins.to_csv(results_dir / "david_temporal_physical_margins.csv", index=False, float_format="%.12g")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for axis, prefix, title in (
        (axes[0], "range", "Min log_high_low_range por ventana"),
        (axes[1], "volume", "Min log1p_volume por ventana"),
    ):
        axis.plot(margins["noise_scale"], margins[f"{prefix}_p01"], marker="o", label="p01")
        axis.plot(margins["noise_scale"], margins[f"{prefix}_p05"], marker="o", label="p05")
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.axvline(0.40, color="#f58518", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("noise_scale")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("valor calibrado")
    axes[0].legend(fontsize=8)
    return _save(fig, figures_dir / "physical_margin_temporal_sweep.png")


def write_decision_table(comparison: pd.DataFrame, results_dir: Path) -> Path:
    selected_rows = comparison[
        comparison["method"].isin(
            [
                BASELINE_METHOD,
                "temporal_jitter_0p25_rho0p85",
                "temporal_jitter_0p30_rho0p85",
                SELECTED_METHOD,
                "temporal_jitter_0p50_rho0p85",
                "pca_gmm_32c12",
            ]
        )
    ].copy()
    selected_rows = selected_rows[
        [
            "method",
            "family",
            "c2st_auc",
            "mean_wasserstein",
            "return_acf_mae",
            "abs_return_acf_mae",
            "invalid_rate",
            "best_ratio",
            "best_mean_rmse",
            "best_mean_mae",
        ]
    ].sort_values("best_mean_rmse")
    path = results_dir / "david_decision_shortlist.csv"
    selected_rows.to_csv(path, index=False, float_format="%.12g")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    comparison = pd.read_csv(args.results_dir / "david_candidate_comparison.csv")
    utility = pd.read_csv(args.results_dir / "david_candidate_utility_summary.csv")
    _, real_validation, _ = load_real_reference(
        REPO_ROOT / "data/features/windows/donor_train.parquet",
        REPO_ROOT / "data/features/windows/donor_validation.parquet",
    )

    generated = [
        plot_pareto(comparison, args.figures_dir),
        plot_temporal_sweep(comparison, args.figures_dir),
        plot_utility_curves(utility, args.figures_dir),
        plot_marginals(real_validation, args.figures_dir),
        plot_distribution_bands(real_validation, args.figures_dir),
        plot_acf_curves(real_validation, args.figures_dir, absolute=False),
        plot_acf_curves(real_validation, args.figures_dir, absolute=True),
        plot_physical_margin(comparison, args.results_dir, args.figures_dir),
        write_decision_table(comparison, args.results_dir),
    ]
    print("Generated David diagnostics:")
    for path in generated:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
