"""Build the DDPM-only reporting figures embedded in the module README.

This script reads the frozen final reporting tables. It does not train the
DDPM, generate synthetic windows, or run any common-pipeline stage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DANIEL_ROOT = Path(__file__).resolve().parents[1]
UTILITY_TABLE = REPO_ROOT / "reports" / "final_analysis" / "master_utility_table.csv"
STABILITY_TABLE = REPO_ROOT / "reports" / "final_analysis" / "seed_stability.csv"
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


def _save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib; generated from frozen final reporting tables"},
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
    build_utility_figure()
    build_stability_figure()
    print(f"Wrote DDPM README figures to {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
