"""Sanity-control plots for final NVDA-like pools; not common fidelity."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .validation import CHANNEL_ORDER


def rejection_annotations(summary: pd.DataFrame) -> list[str]:
    required = {"seed", "n_accepted", "n_rejected", "rejection_rate"}
    if not required.issubset(summary.columns):
        raise ValueError(f"Rejection summary lacks columns: {sorted(required - set(summary))}")
    return [
        f"{int(row.n_accepted)} accepted / {int(row.n_rejected)} rejected\n"
        f"{float(row.rejection_rate):.1%}"
        for row in summary.itertuples(index=False)
    ]


def generate_final_pool_figures(
    summary: pd.DataFrame,
    pools: dict[int, np.ndarray],
    visible_daily: pd.DataFrame,
    output_directory: Path | str,
) -> dict[str, Path]:
    if set(pools) != {42, 123, 2026}:
        raise ValueError("Plots require pools for seeds 42, 123, and 2026")
    for values in pools.values():
        if values.ndim != 3 or tuple(values.shape[1:]) != (65, 3):
            raise ValueError("Every plotted pool must have shape (N, 65, 3)")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    figure, axis = plt.subplots(figsize=(8, 5))
    seeds = summary["seed"].astype(str).tolist()
    rates = summary["rejection_rate"].to_numpy(dtype=np.float64)
    axis.bar(seeds, rates, color="tab:blue", alpha=0.35)
    # Zero-height bars are correct but visually disappear. Markers and exact
    # labels communicate 0% without inventing positive rejection rates.
    axis.scatter(seeds, rates, color="tab:blue", marker="o", zorder=3)
    for seed, rate, label in zip(
        seeds, rates, rejection_annotations(summary), strict=True
    ):
        axis.annotate(
            label,
            xy=(seed, rate),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if np.all(rates == 0):
        axis.set_ylim(-0.002, 0.025)
    axis.set_xlabel("Training/sampling seed")
    axis.set_ylabel("Whole-window rejection rate")
    axis.set_title("Final Diffusion pool rejection rates")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output / "diffusion_final_pool_rejection_rates.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths["rejection_rates"] = path

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_ORDER, strict=True)):
        flattened = [pools[seed][:, :, index].reshape(-1) for seed in sorted(pools)]
        lower = min(float(values.min()) for values in flattened)
        upper = max(float(values.max()) for values in flattened)
        bins = np.linspace(lower, upper, 81)
        for seed, values in zip(sorted(pools), flattened, strict=True):
            axis.hist(values, bins=bins, density=True, histtype="step", label=f"seed {seed}")
        axis.set_title(channel)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[-1].legend()
    figure.suptitle("NVDA-like channel distributions by frozen Diffusion seed")
    figure.tight_layout()
    path = output / "diffusion_nvda_like_channel_distributions_by_seed.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths["channel_distributions"] = path

    visible_returns = visible_daily["log_return"].to_numpy(dtype=np.float64)
    synthetic_returns = [pools[seed][:, :, 0].reshape(-1) for seed in sorted(pools)]
    all_returns = [visible_returns, *synthetic_returns]
    lower = min(float(values.min()) for values in all_returns)
    upper = max(float(values.max()) for values in all_returns)
    bins = np.linspace(lower, upper, 101)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.hist(
        visible_returns,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="NVDA visible daily",
    )
    for seed, values in zip(sorted(pools), synthetic_returns, strict=True):
        axis.hist(values, bins=bins, density=True, histtype="step", label=f"seed {seed}")
    axis.set_xlabel("log_return")
    axis.set_ylabel("Density")
    axis.set_title("NVDA visible vs NVDA-like Diffusion returns (sanity only)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = output / "diffusion_nvda_like_return_distribution_vs_visible.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths["return_vs_visible"] = path
    return paths
