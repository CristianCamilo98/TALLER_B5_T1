"""
Common visualizations (Part 12). Lee unicamente los CSV ya producidos --
nunca recalcula nada, para que las graficas no puedan divergir silenciosamente
de las tablas ya reportadas.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

utility_run = importlib.import_module("common_pipeline.03_utility.utility_run")
TABLES_DIR: Path
FIGURES_DIR: Path

RATIO_LABELS = {0.0: "0%", 0.25: "25%", 0.50: "50%", 0.75: "75%"}


def plot_physical_rejection_rates():
    df = pd.read_csv(TABLES_DIR / "physical_validity.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["method"], df["invalid_rate"] * 100, color="tab:red")
    ax.set_ylabel("invalid (%)")
    ax.set_title("Physical rejection rate by method")
    ax.set_ylim(0, max(5, df["invalid_rate"].max() * 100 * 1.5))
    for i, v in enumerate(df["invalid_rate"] * 100):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "physical_rejection_rates.png", dpi=120)
    plt.close(fig)
    print("Guardado: physical_rejection_rates.png")


def _plot_metric_vs_ratio(summary, metric, filename, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted(summary["method"].unique()):
        sub = summary[summary["method"] == method].sort_values("ratio")
        ax.errorbar(sub["ratio"] * 100, sub[f"mean_{metric}"], yerr=sub[f"std_{metric}"],
                     marker="o", capsize=4, label=method)
    ax.set_xlabel("% sintetico en la mezcla")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. proporcion sintetica, por metodo")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"Guardado: {filename}")


def plot_rmse_improvement_heatmap(summary):
    pivot = summary.pivot(index="method", columns="ratio", values="delta_rmse_pct_vs_real_only")
    pivot = pivot[[c for c in [0.0, 0.25, 0.50, 0.75] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([RATIO_LABELS[c] for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("ratio sintetico")
    ax.set_title("% cambio en RMSE vs. real-only (negativo = mejora)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="% cambio RMSE")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rmse_improvement_heatmap.png", dpi=120)
    plt.close(fig)
    print("Guardado: rmse_improvement_heatmap.png")


def plot_utility_summary(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in sorted(summary["method"].unique()):
        sub = summary[summary["method"] == method].sort_values("ratio")
        axes[0].plot(sub["ratio"] * 100, sub["mean_rmse"], marker="o", label=method)
    axes[0].set_xlabel("% sintetico"); axes[0].set_ylabel("RMSE medio")
    axes[0].set_title("RMSE absoluto"); axes[0].legend()

    positive_ratios = summary[summary["ratio"] > 0]
    best_ratio = positive_ratios.loc[positive_ratios.groupby("method")["mean_rmse"].idxmin()]
    axes[1].bar(best_ratio["method"], best_ratio["delta_rmse_pct_vs_real_only"], color="tab:blue")
    axes[1].set_ylabel("% mejora RMSE vs. real-only (mejor ratio)")
    axes[1].set_title("Mejor mejora obtenida, por metodo")
    for i, (_, row) in enumerate(best_ratio.iterrows()):
        axes[1].text(i, row["delta_rmse_pct_vs_real_only"], f"{row['delta_rmse_pct_vs_real_only']:.1f}%\n(ratio {RATIO_LABELS[row['ratio']]})", ha="center", va="top")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "utility_summary.png", dpi=120)
    plt.close(fig)
    print("Guardado: utility_summary.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=utility_run.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--results-root", type=Path, default=utility_run.RESULTS_ROOT)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None):
    global TABLES_DIR, FIGURES_DIR
    args = build_parser().parse_args(argv)
    run_dir = utility_run.existing_run_dir(
        results_root=args.results_root,
        registry_path=args.registry_path,
        run_id=args.run_id,
    )
    TABLES_DIR = run_dir / "tables"
    FIGURES_DIR = run_dir / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_physical_rejection_rates()
    summary = pd.read_csv(TABLES_DIR / "downstream_results_summary.csv")
    _plot_metric_vs_ratio(summary, "rmse", "rmse_vs_synthetic_ratio.png", "RMSE")
    _plot_metric_vs_ratio(summary, "mae", "mae_vs_synthetic_ratio.png", "MAE")
    plot_rmse_improvement_heatmap(summary)
    plot_utility_summary(summary)


if __name__ == "__main__":
    main()
