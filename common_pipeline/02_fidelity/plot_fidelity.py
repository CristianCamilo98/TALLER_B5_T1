"""Generate every common-fidelity figure from one evaluation run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from fidelity_core import (  # noqa: E402
    CHANNEL_ORDER,
    apply_common_subset,
    load_real_reference,
    load_synthetic_pool,
)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _method_colors(methods: list[str]) -> dict[str, object]:
    palette = plt.get_cmap("tab10")
    return {method: palette(index % 10) for index, method in enumerate(methods)}


def load_plot_inputs(
    repository_root: Path, results_dir: Path
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Reload exact evaluation inputs; plotting never changes their scale."""

    manifest = json.loads(
        (results_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    _, real, statistics = load_real_reference(
        repository_root / manifest["nearest_neighbor_reference"],
        repository_root / manifest["real_reference"],
    )
    indices = pd.read_csv(
        results_dir / "tables/evaluation_subset_indices.csv"
    )["row_position"].to_numpy(dtype=np.int64)
    pools = {
        method: load_synthetic_pool(
            repository_root / metadata["path"],
            method=method,
            expected_normalizer=statistics,
        ).windows
        for method, metadata in manifest["synthetic_methods"].items()
    }
    return real, apply_common_subset(pools, indices), manifest


def plot_marginals(
    real: np.ndarray,
    synthetic: dict[str, np.ndarray],
    figures_dir: Path,
) -> list[Path]:
    """Use one set of bins per channel across real and every method."""

    datasets = {"real": real, **synthetic}
    methods = list(datasets)
    colors = _method_colors(methods)
    paths = []
    for channel_index, channel in enumerate(CHANNEL_ORDER):
        values = {
            method: windows[:, :, channel_index].reshape(-1)
            for method, windows in datasets.items()
        }
        # Shared edges are essential: method-specific bins can visually hide
        # differences in center, dispersion, and tails.
        shared = np.concatenate(list(values.values()))
        bins = np.histogram_bin_edges(shared, bins=60)
        figure, axis = plt.subplots(figsize=(9, 5))
        for method in methods:
            axis.hist(
                values[method],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=colors[method],
                label=method,
            )
        axis.set_title(f"Common normalized marginal: {channel}")
        axis.set_xlabel(channel)
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(alpha=0.2)
        path = figures_dir / f"marginals_{channel}.png"
        _save(figure, path)
        paths.append(path)
    return paths


def plot_acf(table: pd.DataFrame, path: Path, *, title: str) -> None:
    methods = table["method"].drop_duplicates().tolist()
    colors = _method_colors(["real", *methods])
    figure, axis = plt.subplots(figsize=(9, 5))
    first = table.loc[table["method"] == methods[0]]
    axis.plot(
        first["lag"], first["real_acf"], marker="o", label="real", color=colors["real"]
    )
    for method in methods:
        subset = table.loc[table["method"] == method]
        axis.plot(
            subset["lag"],
            subset["synthetic_acf"],
            marker="o",
            markersize=3,
            label=method,
            color=colors[method],
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set(title=title, xlabel="Lag", ylabel="Mean within-window ACF")
    axis.set_xticks(range(1, 21))
    axis.legend()
    axis.grid(alpha=0.2)
    _save(figure, path)


def plot_correlations(table: pd.DataFrame, path: Path) -> None:
    methods = table["method"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        1, len(methods), figsize=(4.1 * len(methods), 4), squeeze=False
    )
    image = None
    for axis, method in zip(axes[0], methods):
        subset = table.loc[table["method"] == method]
        matrix = (
            subset.pivot(
                index="row_channel", columns="column_channel", values="correlation"
            )
            .reindex(index=CHANNEL_ORDER, columns=CHANNEL_ORDER)
            .to_numpy()
        )
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_title(method)
        axis.set_xticks(range(3), CHANNEL_ORDER, rotation=45, ha="right")
        axis.set_yticks(range(3), CHANNEL_ORDER)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Pearson r")
    figure.suptitle("Common channel correlations (fixed scale -1 to +1)")
    figure.subplots_adjust(bottom=0.25, top=0.83, wspace=0.4)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_nearest_neighbor(table: pd.DataFrame, path: Path) -> None:
    methods = table["method"].tolist()
    positions = np.arange(len(methods))
    means = table["mean"].to_numpy()
    lower = means - table["p05"].to_numpy()
    upper = table["p95"].to_numpy() - means
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(positions, means, yerr=[lower, upper], fmt="o", capsize=5)
    axis.scatter(positions, table["median"], marker="x", color="black", label="Median")
    axis.set_xticks(positions, methods, rotation=25, ha="right")
    axis.set_ylabel("Euclidean distance to nearest donor-train window")
    axis.set_title("Nearest-neighbour diagnostic (mean and p05-p95)")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    _save(figure, path)


def plot_c2st(table: pd.DataFrame, path: Path) -> None:
    methods = table["method"].tolist()
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(methods, table["roc_auc"], color=plt.get_cmap("tab10").colors[: len(methods)])
    axis.axhline(0.5, color="black", linestyle="--", label="Random reference (0.50)")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Out-of-fold ROC-AUC")
    axis.set_title("Common Logistic Regression C2ST")
    axis.tick_params(axis="x", rotation=25)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.legend()
    _save(figure, path)


def plot_joint_tsne(table: pd.DataFrame, path: Path) -> None:
    methods = table["method"].drop_duplicates().tolist()
    colors = _method_colors(methods)
    figure, axis = plt.subplots(figsize=(9, 7))
    for method in methods:
        subset = table.loc[table["method"] == method]
        axis.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            s=11,
            alpha=0.65,
            label=method,
            color=colors[method],
        )
    axis.set_title("Joint t-SNE: real and all synthetic methods")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(markerscale=1.8)
    axis.grid(alpha=0.15)
    _save(figure, path)


def plot_scorecard(tables_dir: Path, path: Path) -> None:
    """Show metrics side-by-side without inventing a composite score."""

    wasserstein = pd.read_csv(tables_dir / "wasserstein.csv")
    return_acf = pd.read_csv(tables_dir / "return_acf.csv")
    abs_acf = pd.read_csv(tables_dir / "abs_return_acf.csv")
    correlations = pd.read_csv(tables_dir / "correlation_errors.csv")
    c2st = pd.read_csv(tables_dir / "c2st_results.csv")
    nearest = pd.read_csv(tables_dir / "nearest_neighbor.csv")
    nearest = nearest.loc[nearest["method"] != "real_validation"]
    methods = c2st["method"].tolist()
    positions = np.arange(len(methods))
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))

    for channel in CHANNEL_ORDER:
        subset = wasserstein.loc[wasserstein["channel"] == channel].set_index("method")
        axes[0, 0].plot(positions, subset.loc[methods, "wasserstein_1"], marker="o", label=channel)
    axes[0, 0].set_title("Wasserstein-1 by channel (lower is closer)")
    axes[0, 0].legend(fontsize=8)

    return_values = return_acf.groupby("method")["acf_mae"].first().reindex(methods)
    abs_values = abs_acf.groupby("method")["acf_mae"].first().reindex(methods)
    axes[0, 1].plot(positions, return_values, marker="o", label="return")
    axes[0, 1].plot(positions, abs_values, marker="o", label="abs(return)")
    axes[0, 1].set_title("ACF MAE lags 1-20 (lower is closer)")
    axes[0, 1].legend()

    corr_values = correlations.set_index("method").loc[
        methods, "mean_absolute_off_diagonal_difference"
    ]
    axes[0, 2].bar(positions, corr_values)
    axes[0, 2].set_title("Mean correlation error (lower is closer)")

    axes[1, 0].bar(positions, c2st.set_index("method").loc[methods, "roc_auc"])
    axes[1, 0].axhline(0.5, color="black", linestyle="--")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_title("C2ST ROC-AUC (0.5 is hard to distinguish)")

    nn_values = nearest.set_index("method").loc[methods]
    axes[1, 1].plot(positions, nn_values["mean"], marker="o", label="mean")
    axes[1, 1].plot(positions, nn_values["median"], marker="x", label="median")
    axes[1, 1].set_title("Nearest-train distance (diagnostic only)")
    axes[1, 1].legend()

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.5,
        0.5,
        "No composite score\nMetrics retain their own units\nand interpretations.",
        ha="center",
        va="center",
        fontsize=13,
    )
    for axis in axes.ravel()[:5]:
        axis.set_xticks(positions, methods, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Common fidelity scorecard — separate diagnostics, no ranking score")
    _save(figure, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=MODULE_DIR.parents[1])
    parser.add_argument("--results-dir", type=Path, default=MODULE_DIR / "results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = args.repo_root.resolve()
    results_dir = args.results_dir.resolve()
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    real, synthetic, _ = load_plot_inputs(repository_root, results_dir)

    generated = plot_marginals(real, synthetic, figures_dir)
    return_acf = pd.read_csv(tables_dir / "return_acf.csv")
    abs_return_acf = pd.read_csv(tables_dir / "abs_return_acf.csv")
    plot_acf(return_acf, figures_dir / "return_acf.png", title="Return ACF, lags 1-20")
    plot_acf(
        abs_return_acf,
        figures_dir / "abs_return_acf.png",
        title="Absolute-return ACF, lags 1-20",
    )
    plot_correlations(
        pd.read_csv(tables_dir / "channel_correlations.csv"),
        figures_dir / "channel_correlations.png",
    )
    plot_nearest_neighbor(
        pd.read_csv(tables_dir / "nearest_neighbor.csv"),
        figures_dir / "nearest_neighbor.png",
    )
    plot_c2st(
        pd.read_csv(tables_dir / "c2st_results.csv"),
        figures_dir / "c2st_auc.png",
    )
    plot_joint_tsne(
        pd.read_csv(tables_dir / "joint_tsne_coordinates.csv"),
        figures_dir / "joint_tsne.png",
    )
    plot_scorecard(tables_dir, figures_dir / "fidelity_scorecard.png")
    generated.extend(
        figures_dir / filename
        for filename in (
            "return_acf.png",
            "abs_return_acf.png",
            "channel_correlations.png",
            "nearest_neighbor.png",
            "c2st_auc.png",
            "joint_tsne.png",
            "fidelity_scorecard.png",
        )
    )
    print(f"Generated {len(generated)} common figures in {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
