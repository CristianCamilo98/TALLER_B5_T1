"""Build the professional reporting figure layer from the derived analysis tables.

Reads exclusively from ``reports/final_analysis/*.csv`` and, only where the
derived tables do not carry enough detail (channel-level marginal quantiles),
from ``artifacts/final/strict_final_20260902/fidelity/tables/**``. Writes PNG
(300 dpi) and, for the presentation set, SVG figures under
``reports/final_analysis/figures/``.

This script does not run any part of 01_contract/02_fidelity/03_utility, does
not retrain or recompute any model, and does not modify anything under
``artifacts/final/strict_final_20260902/`` or the derived CSVs in
``reports/final_analysis/``. It only reads numbers that already exist and
renders them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent / "final_analysis"
SNAPSHOT = REPO_ROOT / "artifacts" / "final" / "strict_final_20260902"
FIG_DIR = ANALYSIS_DIR / "figures"
PRESENTATION_DIR = FIG_DIR / "presentation"
GITHUB_DIR = FIG_DIR / "github"

# --------------------------------------------------------------------------
# Naming: models only, never people. Fixed presentation order everywhere.
# --------------------------------------------------------------------------
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
RATIO_LABELS = {0.25: "25%", 0.50: "50%", 0.75: "75%"}

# --------------------------------------------------------------------------
# Validated categorical palette (dataviz skill reference instance), assigned
# in the fixed method order above -- never cycled, never re-ranked.
# --------------------------------------------------------------------------
METHOD_COLOR = {
    "bootstrap_jitter": "#2a78d6",  # slot 1 blue
    "cristian": "#eb6834",  # slot 2 orange
    "daniel": "#1baf7a",  # slot 3 aqua
    "marco": "#eda100",  # slot 4 yellow
    "david": "#e87ba4",  # slot 5 magenta
}
METHOD_MARKER = {
    "bootstrap_jitter": "o",
    "cristian": "s",
    "daniel": "^",
    "marco": "D",
    "david": "v",
}
REAL_ONLY_COLOR = "#52514e"  # secondary ink -- deliberately outside the categorical set
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS_LINE,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.facecolor": SURFACE,
    }
)

created_files: list[Path] = []


def _sanitize_svg_whitespace(path: Path) -> None:
    """Strip trailing whitespace from every line of a matplotlib-generated SVG.

    Purely textual: rstrip() per line plus a normalized trailing newline. Does
    not touch element content, attribute values, geometry, styles, labels, or
    dimensions -- only whitespace sitting after the last non-space character
    on each line (matplotlib's SVG backend leaves this behind in some
    generated `<path d="...">` attributes).
    """

    text = path.read_text(encoding="utf-8")
    sanitized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    if sanitized != text:
        path.write_text(sanitized, encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str, *, presentation: bool, github: bool, svg: bool = True) -> None:
    if presentation:
        path_png = PRESENTATION_DIR / f"{stem}.png"
        fig.savefig(path_png, bbox_inches="tight")
        created_files.append(path_png)
        if svg:
            path_svg = PRESENTATION_DIR / f"{stem}.svg"
            fig.savefig(path_svg, bbox_inches="tight")
            _sanitize_svg_whitespace(path_svg)
            created_files.append(path_svg)
    if github:
        path_png = GITHUB_DIR / f"{stem}.png"
        fig.savefig(path_png, bbox_inches="tight")
        created_files.append(path_png)


def method_label(method_id: str) -> str:
    return METHOD_NAME[method_id]


def ordered(df: pd.DataFrame, id_col: str = "method_id") -> pd.DataFrame:
    order_index = {m: i for i, m in enumerate(METHOD_ORDER)}
    return df.assign(_order=df[id_col].map(order_index)).sort_values("_order").drop(columns="_order")


def legend_handles(method_ids: list[str], *, marker: bool = False) -> list[Line2D]:
    handles = []
    for mid in method_ids:
        if marker:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=METHOD_MARKER[mid],
                    color=METHOD_COLOR[mid],
                    linestyle="none",
                    markersize=9,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=method_label(mid),
                )
            )
        else:
            handles.append(Line2D([0], [0], color=METHOD_COLOR[mid], lw=3, label=method_label(mid)))
    return handles


# --------------------------------------------------------------------------
# Data loading (reports/final_analysis/*.csv, read-only)
# --------------------------------------------------------------------------
def load_master() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_DIR / "master_utility_table.csv")


def load_seed_stability() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_DIR / "seed_stability.csv")


def load_baseline_vs_neural() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_DIR / "baseline_vs_neural.csv")


def load_fidelity_master() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_DIR / "fidelity_master.csv")


def load_fidelity_vs_utility() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_DIR / "fidelity_vs_utility.csv")


def real_only_reference() -> tuple[float, float]:
    summary = pd.read_csv(SNAPSHOT / "utility" / "tables" / "downstream_results_summary.csv")
    row = summary[summary["method"] == "REAL_ONLY"].iloc[0]
    return float(row["mean_rmse"]), float(row["mean_mae"])


# --------------------------------------------------------------------------
# Figure 1 -- RMSE by synthetic share (+ zoom variant)
# --------------------------------------------------------------------------
def fig_rmse_by_share(master: pd.DataFrame, real_only_rmse: float) -> None:
    def draw(ax: plt.Axes, *, zoomed: bool) -> None:
        for mid in METHOD_ORDER:
            sub = master[master["method_id"] == mid].sort_values("synthetic_ratio")
            ax.errorbar(
                sub["synthetic_ratio"] * 100,
                sub["rmse_mean"],
                yerr=sub["rmse_std"],
                marker=METHOD_MARKER[mid],
                color=METHOD_COLOR[mid],
                linewidth=2,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.6,
                capsize=4,
                label=method_label(mid),
            )
        if not zoomed:
            ax.axhline(real_only_rmse, color=REAL_ONLY_COLOR, linestyle="--", linewidth=1.6, zorder=1)
            ax.text(
                26,
                real_only_rmse,
                f"Real-only reference ≈ {real_only_rmse:.3f}",
                color=REAL_ONLY_COLOR,
                fontsize=10.5,
                va="bottom",
            )
            ax.set_ylim(bottom=0)
        else:
            ax.text(
                0.02,
                0.97,
                f"Real-only reference ≈ {real_only_rmse:.3f} RMSE (off chart — see full-range figure)",
                transform=ax.transAxes,
                color=REAL_ONLY_COLOR,
                fontsize=10.5,
                va="top",
                ha="left",
            )
        ax.set_xticks([25, 50, 75])
        ax.set_xticklabels(["25%", "50%", "75%"])
        ax.set_xlim(15, 85)
        ax.set_xlabel("Synthetic data share")
        ax.set_ylabel("RMSE")
        ax.set_title("Forecast RMSE vs Synthetic Data Share")

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    draw(ax, zoomed=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Method")
    fig.text(0.02, -0.02, "Lower is better · error bars = ±1 SD across 3 seeds", fontsize=10, color=INK_MUTED)
    save_figure(fig, "01_rmse_by_synthetic_share", presentation=True, github=False)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    draw(ax, zoomed=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Method")
    fig.text(0.02, -0.02, "Lower is better · error bars = ±1 SD across 3 seeds", fontsize=10, color=INK_MUTED)
    save_figure(fig, "rmse_by_synthetic_share", presentation=False, github=True, svg=False)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    draw(ax, zoomed=True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Method")
    fig.text(
        0.02,
        -0.02,
        "Lower is better · error bars = ±1 SD across 3 seeds · zoomed to the 5-method range",
        fontsize=10,
        color=INK_MUTED,
    )
    save_figure(fig, "01b_rmse_by_synthetic_share_zoom", presentation=True, github=False)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 -- RMSE improvement heatmap
# --------------------------------------------------------------------------
def fig_improvement_heatmap(master: pd.DataFrame) -> None:
    pivot = master.pivot(index="method_id", columns="synthetic_ratio", values="rmse_improvement_pct")
    pivot = pivot.reindex(METHOD_ORDER)[RATIOS]
    values = pivot.to_numpy()

    # Fixed color scale so small real differences are not visually exaggerated.
    # Falls back to a data-driven range only if a value ever fell outside it.
    if values.min() >= 70 and values.max() <= 85:
        vmin, vmax = 70.0, 85.0
    else:
        vmin, vmax = values.min() - 1, values.max() + 1
    mid_value = (vmin + vmax) / 2

    def draw(ax: plt.Axes) -> plt.cm.ScalarMappable:
        im = ax.imshow(values, cmap=SEQUENTIAL_BLUE, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(RATIOS)))
        ax.set_xticklabels([RATIO_LABELS[r] for r in RATIOS])
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels([method_label(m) for m in METHOD_ORDER])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                text_color = "white" if values[i, j] > mid_value else INK_PRIMARY
                ax.text(
                    j,
                    i,
                    f"+{values[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight="bold",
                )
        return im

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    im = draw(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("RMSE improvement vs Real-Only (%)")
    ax.set_title("RMSE Improvement vs Real-Only")
    fig.text(0.02, -0.03, "Positive values indicate lower forecasting error.", fontsize=10, color=INK_MUTED)
    save_figure(fig, "02_rmse_improvement_heatmap", presentation=True, github=False)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    im = draw(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("RMSE improvement vs Real-Only (%)")
    ax.set_title("RMSE Improvement vs Real-Only")
    fig.text(0.02, -0.03, "Positive values indicate lower forecasting error.", fontsize=10, color=INK_MUTED)
    save_figure(fig, "rmse_improvement_heatmap", presentation=False, github=True, svg=False)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 -- Performance at 75%  (RMSE, then MAE variant)
# --------------------------------------------------------------------------
def _performance_at_ratio(
    master: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    real_only_value: float,
    stem: str,
    title: str,
    presentation: bool = True,
    github: bool = False,
    github_stem: str | None = None,
) -> None:
    sub = master[np.isclose(master["synthetic_ratio"], 0.75)].copy()
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    improvement_col = f"{metric}_improvement_pct"
    sub = sub.sort_values(mean_col, ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    y_pos = np.arange(len(sub))
    colors = [METHOD_COLOR[m] for m in sub["method_id"]]
    ax.barh(y_pos, sub[mean_col], xerr=sub[std_col], color=colors, capsize=4, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([method_label(m) for m in sub["method_id"]])
    ax.invert_yaxis()  # best (lowest) at top
    for y, (value, std, improvement) in enumerate(zip(sub[mean_col], sub[std_col], sub[improvement_col])):
        ax.text(value + std + 0.004, y, f"{value:.3f}  (+{improvement:.1f}%)", va="center", fontsize=11, color=INK_PRIMARY)
    ax.set_xlim(right=sub[mean_col].max() + sub[std_col].max() + 0.095)
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.text(
        0.98,
        -0.14,
        f"Real-only reference {metric_label.split()[0]} ≈ {real_only_value:.3f} (not shown to scale)",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        color=INK_MUTED,
    )
    save_figure(fig, stem, presentation=presentation, github=False)
    if github and github_stem:
        fig.savefig(GITHUB_DIR / f"{github_stem}.png", bbox_inches="tight")
        created_files.append(GITHUB_DIR / f"{github_stem}.png")
    plt.close(fig)


def fig_performance_at_75pct(master: pd.DataFrame, real_only_rmse: float, real_only_mae: float) -> None:
    _performance_at_ratio(
        master,
        metric="rmse",
        metric_label="RMSE (75% synthetic share)",
        real_only_value=real_only_rmse,
        stem="03_performance_at_75pct",
        title="Forecast RMSE at 75% Synthetic Share",
        presentation=True,
        github=True,
        github_stem="performance_at_75pct",
    )
    _performance_at_ratio(
        master,
        metric="mae",
        metric_label="MAE (75% synthetic share)",
        real_only_value=real_only_mae,
        stem="03b_mae_at_75pct",
        title="Forecast MAE at 75% Synthetic Share",
        presentation=True,
        github=False,
    )


def fig_mae_by_share(master: pd.DataFrame, real_only_mae: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for mid in METHOD_ORDER:
        sub = master[master["method_id"] == mid].sort_values("synthetic_ratio")
        ax.errorbar(
            sub["synthetic_ratio"] * 100,
            sub["mae_mean"],
            yerr=sub["mae_std"],
            marker=METHOD_MARKER[mid],
            color=METHOD_COLOR[mid],
            linewidth=2,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            capsize=4,
            label=method_label(mid),
        )
    ax.axhline(real_only_mae, color=REAL_ONLY_COLOR, linestyle="--", linewidth=1.6, zorder=1)
    ax.text(26, real_only_mae, f"Real-only reference ≈ {real_only_mae:.3f}", color=REAL_ONLY_COLOR, fontsize=10.5, va="bottom")
    ax.set_ylim(bottom=0)
    ax.set_xticks([25, 50, 75])
    ax.set_xticklabels(["25%", "50%", "75%"])
    ax.set_xlim(15, 85)
    ax.set_xlabel("Synthetic data share")
    ax.set_ylabel("MAE")
    ax.set_title("Forecast MAE vs Synthetic Data Share")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Method")
    fig.text(0.02, -0.02, "Lower is better · error bars = ±1 SD across 3 seeds", fontsize=10, color=INK_MUTED)
    save_figure(fig, "mae_by_synthetic_share", presentation=False, github=True, svg=False)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4 -- Seed stability
# --------------------------------------------------------------------------
def fig_seed_stability(stability: pd.DataFrame) -> None:
    ratio_shade = {0.25: "#9ec5f4", 0.50: "#2a78d6", 0.75: "#0d366b"}
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    n_methods = len(METHOD_ORDER)
    offsets = np.linspace(-0.22, 0.22, len(RATIOS))
    for mi, mid in enumerate(METHOD_ORDER):
        for ri, ratio in enumerate(RATIOS):
            row = stability[(stability["method_id"] == mid) & np.isclose(stability["synthetic_ratio"], ratio)].iloc[0]
            x = mi + offsets[ri]
            ax.plot([x, x], [row["rmse_min"], row["rmse_max"]], color=ratio_shade[ratio], linewidth=2, solid_capstyle="round")
            ax.plot(
                x,
                row["rmse_mean"],
                marker="o",
                markersize=7,
                color=ratio_shade[ratio],
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([method_label(m) for m in METHOD_ORDER])
    ax.set_ylabel("RMSE")
    ax.set_title("Seed Stability by Method and Synthetic Share")
    handles = [Line2D([0], [0], color=ratio_shade[r], lw=3, marker="o", markersize=7, label=RATIO_LABELS[r]) for r in RATIOS]
    ax.legend(handles=handles, title="Synthetic share", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.text(
        0.02,
        -0.02,
        "Dot = mean RMSE across 3 downstream mixture/subsampling seeds (42, 123, 2026) · line = min–max range",
        fontsize=10,
        color=INK_MUTED,
    )
    save_figure(fig, "04_seed_stability", presentation=True, github=False)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for mi, mid in enumerate(METHOD_ORDER):
        for ri, ratio in enumerate(RATIOS):
            row = stability[(stability["method_id"] == mid) & np.isclose(stability["synthetic_ratio"], ratio)].iloc[0]
            x = mi + offsets[ri]
            ax.plot([x, x], [row["rmse_min"], row["rmse_max"]], color=ratio_shade[ratio], linewidth=2, solid_capstyle="round")
            ax.plot(x, row["rmse_mean"], marker="o", markersize=7, color=ratio_shade[ratio], markeredgecolor="white", markeredgewidth=0.7)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([method_label(m) for m in METHOD_ORDER])
    ax.set_ylabel("RMSE")
    ax.set_title("Seed Stability by Method and Synthetic Share")
    handles = [Line2D([0], [0], color=ratio_shade[r], lw=3, marker="o", markersize=7, label=RATIO_LABELS[r]) for r in RATIOS]
    ax.legend(handles=handles, title="Synthetic share", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.text(0.02, -0.02, "Dot = mean RMSE across 3 downstream mixture/subsampling seeds (42, 123, 2026) · line = min–max range", fontsize=10, color=INK_MUTED)
    save_figure(fig, "seed_stability", presentation=False, github=True, svg=False)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5 -- Neural vs simple baseline
# --------------------------------------------------------------------------
def fig_baseline_vs_neural(baseline_vs_neural: pd.DataFrame) -> None:
    n_neural = len(NEURAL_METHOD_IDS)
    width = 0.8 / n_neural
    x_base = np.arange(len(RATIOS))
    y_min = baseline_vs_neural["rmse_improvement_pct_difference_vs_baseline"].min()
    y_max = baseline_vs_neural["rmse_improvement_pct_difference_vs_baseline"].max()
    pad = (y_max - y_min) * 0.12

    def draw(ax: plt.Axes) -> None:
        for i, mid in enumerate(NEURAL_METHOD_IDS):
            sub = baseline_vs_neural[baseline_vs_neural["neural_method_id"] == mid].sort_values("synthetic_ratio")
            x = x_base + (i - (n_neural - 1) / 2) * width
            values = sub["rmse_improvement_pct_difference_vs_baseline"].to_numpy()
            ax.bar(x, values, width=width * 0.9, color=METHOD_COLOR[mid], label=method_label(mid))
            for xi, value in zip(x, values):
                offset = pad * 0.18 if value >= 0 else -pad * 0.18
                va = "bottom" if value >= 0 else "top"
                ax.text(xi, value + offset, f"{value:+.2f}", ha="center", va=va, fontsize=8.5, color=INK_PRIMARY)
        ax.axhline(0, color=AXIS_LINE, linewidth=1.4)
        ax.text(
            1.0,
            0,
            "0 = same as\nsimple baseline",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9.5,
            color=INK_MUTED,
        )
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xticks(x_base)
        ax.set_xticklabels([RATIO_LABELS[r] for r in RATIOS])
        ax.set_xlabel("Synthetic data share")
        ax.set_ylabel("RMSE improvement difference\nvs Bootstrap + Jitter (pp)")
        ax.set_title("Do Neural Generators Beat the Simple Baseline?")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, title="Method")

    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    draw(ax)
    fig.text(
        0.02,
        -0.03,
        "Positive = neural model improves more than Bootstrap + Jitter (pp) · Negative = simple baseline wins",
        fontsize=10,
        color=INK_MUTED,
    )
    save_figure(fig, "05_neural_vs_simple_baseline", presentation=True, github=False)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    draw(ax)
    fig.text(
        0.02,
        -0.03,
        "Positive = neural model improves more than Bootstrap + Jitter (pp) · Negative = simple baseline wins",
        fontsize=10,
        color=INK_MUTED,
    )
    save_figure(fig, "neural_vs_simple_baseline", presentation=False, github=True, svg=False)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 6 -- C2ST fidelity
# --------------------------------------------------------------------------
def fig_c2st(fidelity: pd.DataFrame) -> None:
    fid = ordered(fidelity)
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    x = np.arange(len(fid))
    colors = [METHOD_COLOR[m] for m in fid["method_id"]]
    ax.bar(x, fid["c2st_roc_auc"], color=colors, width=0.6)
    ax.axhline(0.5, color=REAL_ONLY_COLOR, linestyle="--", linewidth=1.6)
    ax.text(
        1.01,
        0.5,
        "0.50 = hard to distinguish\nfrom held-out real donors",
        transform=ax.get_yaxis_transform(),
        color=REAL_ONLY_COLOR,
        fontsize=9.5,
        ha="left",
        va="center",
    )
    for xi, value in zip(x, fid["c2st_roc_auc"]):
        ax.text(xi, value + 0.015, f"{value:.3f}", ha="center", fontsize=11, color=INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in fid["method_id"]])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("C2ST ROC-AUC")
    ax.set_title("Synthetic-vs-Real Donor Distinguishability")
    fig.text(
        0.02,
        -0.03,
        "Closer to 0.50 indicates higher distributional similarity under this test (not a global quality score).",
        fontsize=10,
        color=INK_MUTED,
    )
    save_figure(fig, "06_c2st_fidelity", presentation=True, github=False)
    fig.savefig(GITHUB_DIR / "c2st_fidelity.png", bbox_inches="tight")
    created_files.append(GITHUB_DIR / "c2st_fidelity.png")
    plt.close(fig)


def fig_fidelity_summary(fidelity: pd.DataFrame) -> None:
    fid = ordered(fidelity)
    panels = [
        ("c2st_roc_auc", "C2ST ROC-AUC"),
        ("mean_correlation_error", "Mean correlation error"),
        ("return_acf_mae_vs_real", "Return ACF MAE"),
        ("abs_return_acf_mae_vs_real", "Abs-return ACF MAE"),
        ("wasserstein_log_return", "Wasserstein: log_return"),
        ("wasserstein_log_high_low_range", "Wasserstein: log_high_low_range"),
        ("wasserstein_log1p_volume", "Wasserstein: log1p_volume"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
    axes_flat = axes.flatten()
    x = np.arange(len(fid))
    colors = [METHOD_COLOR[m] for m in fid["method_id"]]
    for ax, (col, label) in zip(axes_flat, panels):
        ax.bar(x, fid[col], color=colors, width=0.65)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([method_label(m) for m in fid["method_id"]], rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel("value (lower = closer to real)", fontsize=8.5)
    axes_flat[-1].axis("off")
    handles = legend_handles(METHOD_ORDER)
    axes_flat[-1].legend(handles=handles, loc="center", title="Method", fontsize=10)
    fig.suptitle("Fidelity Metrics by Method — No Composite Score", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.02, "Each panel is an independent metric on its own scale; no metric is aggregated into a ranking.", fontsize=10, color=INK_MUTED)
    save_figure(fig, "fidelity_metric_summary", presentation=False, github=True, svg=False)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 7 -- Fidelity vs utility (exploratory scatter)
# --------------------------------------------------------------------------
FIDELITY_VS_UTILITY_LABEL_OFFSET = {
    "bootstrap_jitter": (10, 8),
    "cristian": (10, -16),
    "daniel": (10, 8),
    "marco": (-10, -18),
    "david": (-14, 12),
}
FIDELITY_VS_UTILITY_LABEL_HA = {
    "bootstrap_jitter": "left",
    "cristian": "left",
    "daniel": "left",
    "marco": "right",
    "david": "right",
}


def fig_fidelity_vs_utility(fvu: pd.DataFrame) -> None:
    fvu = ordered(fvu)
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    for _, row in fvu.iterrows():
        mid = row["method_id"]
        ax.scatter(
            row["c2st_roc_auc"],
            row["best_rmse_improvement_pct"],
            s=140,
            color=METHOD_COLOR[mid],
            marker=METHOD_MARKER[mid],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            method_label(mid),
            (row["c2st_roc_auc"], row["best_rmse_improvement_pct"]),
            textcoords="offset points",
            xytext=FIDELITY_VS_UTILITY_LABEL_OFFSET[mid],
            ha=FIDELITY_VS_UTILITY_LABEL_HA[mid],
            fontsize=10.5,
            color=INK_PRIMARY,
        )
    ax.set_xlabel("C2ST ROC-AUC (fidelity vs held-out real donors)")
    ax.set_ylabel("Best observed RMSE improvement vs Real-Only (%)")
    ax.set_title("Fidelity vs Downstream Utility")
    ax.text(
        0.02,
        0.02,
        "Exploratory · n=5 methods · descriptive only, no causal claim, no fitted trend line",
        transform=ax.transAxes,
        fontsize=10,
        color=INK_MUTED,
        va="bottom",
    )
    save_figure(fig, "07_fidelity_vs_utility", presentation=True, github=False)
    fig.savefig(GITHUB_DIR / "fidelity_vs_utility.png", bbox_inches="tight")
    created_files.append(GITHUB_DIR / "fidelity_vs_utility.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Optional figure -- log_return marginal (quantile range, central 99%)
# --------------------------------------------------------------------------
def fig_log_return_distribution() -> None:
    stats = pd.read_csv(SNAPSHOT / "fidelity" / "tables" / "marginal_statistics.csv")
    stats = stats[stats["channel"] == "log_return"].set_index("method")

    rows_order = ["real"] + METHOD_ORDER
    labels = ["Held-out real donors"] + [method_label(m) for m in METHOD_ORDER]
    colors = [REAL_ONLY_COLOR] + [METHOD_COLOR[m] for m in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for i, (key, label, color) in enumerate(zip(rows_order, labels, colors)):
        row = stats.loc[key]
        y = len(rows_order) - 1 - i
        ax.plot([row["p01"], row["p99"]], [y, y], color=color, linewidth=2, solid_capstyle="round", alpha=0.55)
        ax.plot([row["p25"], row["p75"]], [y, y], color=color, linewidth=7, solid_capstyle="round")
        ax.plot(row["median"], y, marker="o", color="white", markeredgecolor=color, markeredgewidth=2, markersize=8, zorder=3)

    ax.set_yticks(range(len(rows_order)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("log_return")
    ax.set_title("log_return Marginal Distribution — Central 99% View")
    fig.text(
        0.02,
        -0.03,
        "Thick bar = interquartile range (p25–p75) · thin line = central 99% (p01–p99) · dot = median. Underlying calculations use the full data.",
        fontsize=9.5,
        color=INK_MUTED,
    )
    save_figure(fig, "fidelity_log_return_distribution", presentation=False, github=True, svg=False)
    plt.close(fig)


def main() -> int:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    GITHUB_DIR.mkdir(parents=True, exist_ok=True)

    master = load_master()
    stability = load_seed_stability()
    baseline_vs_neural = load_baseline_vs_neural()
    fidelity = load_fidelity_master()
    fvu = load_fidelity_vs_utility()
    real_only_rmse, real_only_mae = real_only_reference()

    fig_rmse_by_share(master, real_only_rmse)
    fig_improvement_heatmap(master)
    fig_performance_at_75pct(master, real_only_rmse, real_only_mae)
    fig_mae_by_share(master, real_only_mae)
    fig_seed_stability(stability)
    fig_baseline_vs_neural(baseline_vs_neural)
    fig_c2st(fidelity)
    fig_fidelity_summary(fidelity)
    fig_fidelity_vs_utility(fvu)
    fig_log_return_distribution()

    print(f"Created {len(created_files)} figure files:")
    for path in created_files:
        print(" -", path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
