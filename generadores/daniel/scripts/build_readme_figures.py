"""Build the DDPM-only reporting figures embedded in the module README.

This script reads frozen experiment inputs and final reporting tables. It does
not train the DDPM, generate synthetic windows, or run any common-pipeline stage.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


REPO_ROOT = Path(__file__).resolve().parents[3]
DANIEL_ROOT = Path(__file__).resolve().parents[1]
UTILITY_TABLE = REPO_ROOT / "reports" / "final_analysis" / "master_utility_table.csv"
STABILITY_TABLE = REPO_ROOT / "reports" / "final_analysis" / "seed_stability.csv"
FIDELITY_MASTER = REPO_ROOT / "reports" / "final_analysis" / "fidelity_master.csv"
FIDELITY_TABLES = (
    REPO_ROOT
    / "artifacts"
    / "final"
    / "strict_final_20260902"
    / "fidelity"
    / "tables"
)
DONOR_VALIDATION = REPO_ROOT / "data" / "features" / "windows" / "donor_validation.parquet"
DDPM_OUTPUT = DANIEL_ROOT / "outputs" / "diffusion_seed42_normalized.parquet"
NORMALIZER = DANIEL_ROOT / "evidence" / "seed42" / "normalizer.json"
COMMON_FIDELITY_CODE = REPO_ROOT / "common_pipeline" / "02_fidelity"
RAW_RESULTS = (
    REPO_ROOT
    / "artifacts"
    / "final"
    / "strict_final_20260902"
    / "utility"
    / "tables"
    / "downstream_results_raw.csv"
)
OUTPUT_DIR = DANIEL_ROOT / "figures"

RATIOS = (0.25, 0.50, 0.75)
SEEDS = (42, 123, 2026)
NAVY = "#17324D"
BLUE = "#2E6F9E"
TEAL = "#2A9D8F"
LIGHT_BLUE = "#9DC8E2"
GRAY = "#697681"
GRID = "#D9E1E8"
REAL_LABEL = "Held-out real donors"
DDPM_LABEL = "DDPM"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": NAVY,
            "axes.edgecolor": "#AAB6C1",
            "xtick.color": "#43515C",
            "ytick.color": "#43515C",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "ddpm-readme-figures-v1",
        }
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def _select_ddpm_utility() -> tuple[pd.DataFrame, float]:
    frame = pd.read_csv(UTILITY_TABLE)
    _require_columns(
        frame,
        {
            "method_id",
            "synthetic_ratio",
            "rmse_mean",
            "rmse_std",
            "real_only_rmse",
        },
        UTILITY_TABLE,
    )
    ddpm = frame.loc[frame["method_id"].eq("daniel")].copy()
    ddpm = ddpm.sort_values("synthetic_ratio").reset_index(drop=True)
    if ddpm["synthetic_ratio"].tolist() != list(RATIOS):
        raise ValueError("Expected exactly the DDPM ratios 0.25, 0.50, and 0.75")
    real_values = ddpm["real_only_rmse"].drop_duplicates().to_numpy(dtype=float)
    if len(real_values) != 1:
        raise ValueError("Expected one common REAL_ONLY RMSE reference")
    return ddpm, float(real_values[0])


def _select_ddpm_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_RESULTS)
    summary = pd.read_csv(STABILITY_TABLE)
    _require_columns(raw, {"method", "ratio", "subsampling_seed", "rmse"}, RAW_RESULTS)
    _require_columns(
        summary,
        {"method_id", "synthetic_ratio", "rmse_mean", "rmse_std", "n_seeds"},
        STABILITY_TABLE,
    )

    ddpm_raw = raw.loc[raw["method"].eq("daniel")].copy()
    ddpm_raw["subsampling_seed"] = ddpm_raw["subsampling_seed"].astype(int)
    ddpm_raw = ddpm_raw.sort_values(["ratio", "subsampling_seed"]).reset_index(drop=True)
    observed_pairs = set(zip(ddpm_raw["ratio"], ddpm_raw["subsampling_seed"]))
    expected_pairs = {(ratio, seed) for ratio in RATIOS for seed in SEEDS}
    if observed_pairs != expected_pairs or len(ddpm_raw) != len(expected_pairs):
        raise ValueError("Expected one DDPM result for every ratio/seed pair")

    ddpm_summary = summary.loc[summary["method_id"].eq("daniel")].copy()
    ddpm_summary = ddpm_summary.sort_values("synthetic_ratio").reset_index(drop=True)
    if ddpm_summary["synthetic_ratio"].tolist() != list(RATIOS):
        raise ValueError("Expected exactly three DDPM stability rows")
    if not ddpm_summary["n_seeds"].eq(3).all():
        raise ValueError("Expected three downstream seeds per DDPM ratio")
    return ddpm_raw, ddpm_summary


def _stack_windows(frame: pd.DataFrame, *, source: Path) -> np.ndarray:
    _require_columns(frame, {"features_flat"}, source)
    rows = [np.asarray(row) for row in frame["features_flat"]]
    if not rows or {row.size for row in rows} != {195}:
        raise ValueError(f"{source} must contain non-empty 195-value windows")
    windows = np.stack(rows).reshape(-1, 65, 3)
    if not np.isfinite(windows).all():
        raise ValueError(f"{source} contains non-finite values")
    return windows


def _load_fidelity_windows() -> tuple[np.ndarray, np.ndarray]:
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))
    if normalizer.get("fit_split") != "donor_train":
        raise ValueError("Expected the canonical donor_train normalizer")
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    std = np.asarray(normalizer["std"], dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Expected three canonical channel statistics")

    real_frame = pd.read_parquet(DONOR_VALIDATION, columns=["features_flat"])
    real = _stack_windows(real_frame, source=DONOR_VALIDATION).astype(np.float64)
    real = ((real - mean) / std).astype(np.float32)
    if real.shape != (380, 65, 3):
        raise ValueError("Expected 380 held-out donor validation windows")

    synthetic_frame = pd.read_parquet(
        DDPM_OUTPUT,
        columns=["source_model", "training_seed", "features_flat"],
    )
    if set(synthetic_frame["source_model"]) != {"diffusion_ddpm"}:
        raise ValueError("Official pool must identify source_model=diffusion_ddpm")
    if set(synthetic_frame["training_seed"]) != {42}:
        raise ValueError("Official pool must identify training_seed=42")
    synthetic_pool = _stack_windows(synthetic_frame, source=DDPM_OUTPUT).astype(
        np.float32, copy=False
    )
    if synthetic_pool.shape != (5000, 65, 3):
        raise ValueError("Expected the official 5,000-window DDPM pool")

    index_path = FIDELITY_TABLES / "evaluation_subset_indices.csv"
    indices = pd.read_csv(index_path)["row_position"].to_numpy(dtype=np.int64)
    if len(indices) != 380 or len(np.unique(indices)) != 380:
        raise ValueError("Expected 380 unique official evaluation row positions")
    if indices.min() < 0 or indices.max() >= len(synthetic_pool):
        raise ValueError("Official evaluation indices exceed the DDPM pool")
    return real, synthetic_pool[indices]


def _save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Software": "Matplotlib; generated from frozen experiment inputs and final tables"
        },
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "Matplotlib"},
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    normalized_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
    svg_path.write_text(normalized_svg, encoding="utf-8", newline="\n")
    plt.close(fig)


def build_marginal_figure(real: np.ndarray, synthetic: np.ndarray) -> None:
    channels = ("log_return", "log_high_low_range", "log1p_volume")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    for index, (axis, channel) in enumerate(zip(axes, channels)):
        real_values = real[:, :, index].reshape(-1)
        synthetic_values = synthetic[:, :, index].reshape(-1)
        combined = np.concatenate((real_values, synthetic_values))
        bins = np.linspace(float(combined.min()), float(combined.max()), 65)
        axis.hist(
            real_values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.28,
            color=GRAY,
            edgecolor=GRAY,
            linewidth=1.2,
            label=REAL_LABEL,
        )
        axis.hist(
            synthetic_values,
            bins=bins,
            density=True,
            histtype="step",
            color=TEAL,
            linewidth=1.8,
            label=DDPM_LABEL,
        )
        axis.set_title(channel, color=NAVY, fontsize=11.5)
        axis.set_xlabel("Canonical normalized value")
        if index == 0:
            axis.set_ylabel("Density")
            axis.legend(frameon=False, fontsize=9)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("DDPM Marginal Fidelity", fontsize=17, color=NAVY, y=0.98)
    fig.text(
        0.5,
        0.90,
        "Held-out real donor validation vs DDPM synthetic windows",
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.015,
        "380 windows per source · all 24,700 values per channel · no clipping or outlier removal",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.065, right=0.985, wspace=0.25)
    _save(fig, "ddpm_marginal_distributions")


def build_tsne_figure() -> None:
    source = FIDELITY_TABLES / "joint_tsne_coordinates.csv"
    frame = pd.read_csv(source)
    _require_columns(frame, {"method", "source_row", "tsne_1", "tsne_2"}, source)
    selected = frame.loc[frame["method"].isin(["real", "daniel"])].copy()
    counts = selected.groupby("method").size().to_dict()
    if counts != {"daniel": 380, "real": 380}:
        raise ValueError(f"Unexpected frozen t-SNE counts: {counts}")

    fig, axis = plt.subplots(figsize=(8.8, 6.2))
    for method, label, color, marker in (
        ("real", REAL_LABEL, GRAY, "o"),
        ("daniel", DDPM_LABEL, TEAL, "^"),
    ):
        subset = selected.loc[selected["method"].eq(method)]
        axis.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            s=25,
            alpha=0.68,
            color=color,
            marker=marker,
            edgecolors="none",
            label=label,
        )
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.grid(color=GRID, linewidth=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="best")
    fig.suptitle("DDPM t-SNE Diagnostic", fontsize=17, color=NAVY, y=0.98)
    fig.text(
        0.5,
        0.915,
        "Held-out real donors and DDPM within the frozen common embedding",
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.018,
        "Qualitative projection of the frozen common fidelity embedding.\n"
        "t-SNE is diagnostic only and is not used as a quantitative score.",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.11, right=0.97)
    _save(fig, "ddpm_tsne_fidelity")


def _frozen_c2st(real: np.ndarray, synthetic: np.ndarray):
    sys.path.insert(0, str(COMMON_FIDELITY_CODE))
    try:
        from fidelity_core import c2st_out_of_fold
    finally:
        sys.path.pop(0)

    result = c2st_out_of_fold(real, synthetic, random_state=42)
    source = FIDELITY_TABLES / "c2st_results.csv"
    table = pd.read_csv(source)
    _require_columns(table, {"method", "roc_auc", "prediction_scope", "cv"}, source)
    row = table.loc[table["method"].eq("daniel")]
    if len(row) != 1 or row.iloc[0]["prediction_scope"] != "out_of_fold":
        raise ValueError("Expected one frozen out-of-fold DDPM C2ST result")
    frozen_auc = float(row.iloc[0]["roc_auc"])
    if not np.isclose(result.roc_auc, frozen_auc, rtol=0.0, atol=5e-12):
        raise RuntimeError(
            f"Reproduced C2ST AUC {result.roc_auc:.15f} does not match "
            f"STRICT_FINAL {frozen_auc:.15f}"
        )
    return result, frozen_auc


def build_c2st_figure(real: np.ndarray, synthetic: np.ndarray) -> tuple[float, float]:
    result, frozen_auc = _frozen_c2st(real, synthetic)
    counts = np.asarray(
        [
            [
                np.sum((result.labels == actual) & (result.predictions == predicted))
                for predicted in (0, 1)
            ]
            for actual in (0, 1)
        ],
        dtype=np.int64,
    )
    percentages = counts / counts.sum(axis=1, keepdims=True)

    fig, axis = plt.subplots(figsize=(7.8, 6.2))
    image = axis.imshow(percentages, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = [REAL_LABEL, DDPM_LABEL]
    axis.set_xticks([0, 1], labels)
    axis.set_yticks([0, 1], labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    for row in range(2):
        for column in range(2):
            color = "white" if percentages[row, column] > 0.55 else NAVY
            axis.text(
                column,
                row,
                f"{counts[row, column]}\n({percentages[row, column]:.1%})",
                ha="center",
                va="center",
                color=color,
                fontsize=14,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Row percentage")
    fig.suptitle("DDPM C2ST Confusion Matrix", fontsize=17, color=NAVY, y=0.98)
    fig.text(
        0.5,
        0.915,
        "5-fold out-of-fold C2ST predictions",
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.025,
        f"Frozen-protocol ROC-AUC = {result.roc_auc:.4f} · counts and row percentages",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.21, right=0.91)
    _save(fig, "ddpm_c2st_confusion_matrix")
    return result.roc_auc, frozen_auc


def build_c2st_roc_figure(real: np.ndarray, synthetic: np.ndarray) -> tuple[float, float]:
    result, frozen_auc = _frozen_c2st(real, synthetic)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        result.labels,
        result.probabilities,
        pos_label=1,
    )

    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color=TEAL,
        linewidth=2.6,
        label=f"DDPM vs real — AUC = {result.roc_auc:.3f}",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=GRAY,
        linewidth=1.5,
        linestyle="--",
        label="Random — AUC = 0.500",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    fig.suptitle("DDPM — C2ST ROC", fontsize=17, color=NAVY, y=0.98)
    fig.text(
        0.5,
        0.90,
        "Held-out real donors vs DDPM synthetic · 5-fold OOF",
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.025,
        "AUC closer to 0.5 means real and synthetic are harder to distinguish.",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.11, right=0.97)
    _save(fig, "ddpm_c2st_roc")
    return result.roc_auc, frozen_auc


def build_acf_figure() -> tuple[float, float]:
    return_source = FIDELITY_TABLES / "return_acf.csv"
    absolute_source = FIDELITY_TABLES / "abs_return_acf.csv"
    metric_source = FIDELITY_MASTER
    return_table = pd.read_csv(return_source)
    absolute_table = pd.read_csv(absolute_source)
    metrics = pd.read_csv(metric_source)
    required_acf = {"method", "lag", "real_acf", "synthetic_acf", "acf_mae"}
    _require_columns(return_table, required_acf, return_source)
    _require_columns(absolute_table, required_acf, absolute_source)
    _require_columns(
        metrics,
        {"method_id", "return_acf_mae_vs_real", "abs_return_acf_mae_vs_real"},
        metric_source,
    )
    return_ddpm = return_table.loc[return_table["method"].eq("daniel")].sort_values("lag")
    absolute_ddpm = absolute_table.loc[absolute_table["method"].eq("daniel")].sort_values("lag")
    metric_row = metrics.loc[metrics["method_id"].eq("daniel")]
    if len(return_ddpm) != 20 or len(absolute_ddpm) != 20 or len(metric_row) != 1:
        raise ValueError("Expected 20 frozen ACF lags and one DDPM metric row")
    return_mae = float(metric_row.iloc[0]["return_acf_mae_vs_real"])
    absolute_mae = float(metric_row.iloc[0]["abs_return_acf_mae_vs_real"])
    if not np.isclose(return_ddpm["acf_mae"].iloc[0], return_mae, atol=5e-12):
        raise ValueError("Return ACF MAE differs between frozen reporting tables")
    if not np.isclose(absolute_ddpm["acf_mae"].iloc[0], absolute_mae, atol=5e-12):
        raise ValueError("Absolute-return ACF MAE differs between frozen reporting tables")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharex=True)
    for axis, table, title, mae in (
        (axes[0], return_ddpm, "A. Return ACF", return_mae),
        (axes[1], absolute_ddpm, "B. Absolute-return ACF", absolute_mae),
    ):
        axis.plot(
            table["lag"],
            table["real_acf"],
            color=GRAY,
            marker="o",
            markersize=4,
            linewidth=1.7,
            label=REAL_LABEL,
        )
        axis.plot(
            table["lag"],
            table["synthetic_acf"],
            color=TEAL,
            marker="^",
            markersize=4,
            linewidth=1.7,
            label=DDPM_LABEL,
        )
        axis.axhline(0.0, color="#9AA7B2", linewidth=0.8)
        axis.set_title(title, color=NAVY, fontsize=11.5)
        axis.set_xlabel("Lag")
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.grid(color=GRID, linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.97,
            0.93,
            f"MAE = {mae:.4f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=NAVY,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": GRID},
        )
    axes[0].set_ylabel("Mean within-window ACF")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("DDPM Temporal Dependence", fontsize=17, color=NAVY, y=0.98)
    fig.text(
        0.5,
        0.905,
        "Mean within-window ACF · held-out donor validation vs DDPM",
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.08, right=0.98, wspace=0.20)
    _save(fig, "ddpm_temporal_acf")
    return return_mae, absolute_mae


def build_utility_figure() -> None:
    ddpm, real_only_rmse = _select_ddpm_utility()
    labels = ["Real-only", "25%", "50%", "75%"]
    means = np.concatenate(([real_only_rmse], ddpm["rmse_mean"].to_numpy(dtype=float)))
    synthetic_std = ddpm["rmse_std"].to_numpy(dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.6, 5.7))
    colors = [GRAY, LIGHT_BLUE, BLUE, TEAL]
    bars = ax.bar(x, means, width=0.62, color=colors, edgecolor="white", linewidth=1.2)
    ax.errorbar(
        x[1:],
        means[1:],
        yerr=synthetic_std,
        fmt="none",
        ecolor=NAVY,
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        zorder=3,
    )
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            color=NAVY,
            fontweight="bold",
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, max(means) * 1.14)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.99,
        0.96,
        "Error bars: ±1 SD across downstream seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GRAY,
        fontsize=9,
    )
    fig.suptitle("DDPM Downstream Utility", fontsize=17, color=NAVY, y=0.98)
    fig.text(
        0.5,
        0.915,
        "RMSE on future NVDA test · lower is better",
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.10, right=0.97)
    _save(fig, "ddpm_rmse_vs_synthetic_share")


def build_stability_figure() -> None:
    raw, summary = _select_ddpm_stability()
    x = np.arange(len(RATIOS))
    seed_colors = {42: "#355C7D", 123: "#C06C84", 2026: "#2A9D8F"}

    fig, ax = plt.subplots(figsize=(9.6, 5.7))
    for seed in SEEDS:
        seed_rows = raw.loc[raw["subsampling_seed"].eq(seed)].sort_values("ratio")
        ax.plot(
            x,
            seed_rows["rmse"].to_numpy(dtype=float),
            marker="o",
            markersize=7,
            linewidth=1.6,
            color=seed_colors[seed],
            label=f"Seed {seed}",
            alpha=0.9,
        )

    means = summary["rmse_mean"].to_numpy(dtype=float)
    std = summary["rmse_std"].to_numpy(dtype=float)
    ax.errorbar(
        x,
        means,
        yerr=std,
        fmt="D-",
        markersize=7,
        linewidth=2.2,
        color=NAVY,
        markerfacecolor="white",
        markeredgewidth=1.8,
        capsize=5,
        label="Mean ±1 SD",
        zorder=5,
    )

    ax.set_xticks(x, ["25%", "50%", "75%"])
    ax.set_xlabel("Synthetic share")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("DDPM Downstream Stability", fontsize=17, color=NAVY, y=0.98)
    fig.text(
        0.5,
        0.915,
        "Downstream mixture/subsampling seeds, not DDPM training seeds.",
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.10, right=0.97)
    _save(fig, "ddpm_downstream_stability")


def main() -> None:
    _configure_style()
    real, synthetic = _load_fidelity_windows()
    build_utility_figure()
    build_stability_figure()
    build_marginal_figure(real, synthetic)
    build_tsne_figure()
    reproduced_auc, frozen_auc = build_c2st_figure(real, synthetic)
    roc_auc, roc_frozen_auc = build_c2st_roc_figure(real, synthetic)
    if reproduced_auc != roc_auc or frozen_auc != roc_frozen_auc:
        raise RuntimeError("C2ST confusion-matrix and ROC guards disagree")
    return_mae, absolute_mae = build_acf_figure()
    print(f"Wrote DDPM README figures to {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print(
        "C2ST guard: "
        f"reproduced_auc={reproduced_auc:.15f}, "
        f"strict_final_auc={frozen_auc:.15f}"
    )
    print(
        "ACF metrics from fidelity_master.csv: "
        f"return_mae={return_mae:.12f}, abs_return_mae={absolute_mae:.12f}"
    )


if __name__ == "__main__":
    main()
