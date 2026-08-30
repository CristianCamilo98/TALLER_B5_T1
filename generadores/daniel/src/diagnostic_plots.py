"""Plots driven exclusively by normalized diagnostic arrays and tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from .diagnostics import as_numpy_windows
from .validation import CHANNEL_ORDER


def _shared_bins(first: np.ndarray, second: np.ndarray, count: int = 80) -> np.ndarray:
    minimum = float(min(first.min(), second.min()))
    maximum = float(max(first.max(), second.max()))
    if minimum == maximum:
        maximum = minimum + 1.0
    return np.linspace(minimum, maximum, count + 1)


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate_diagnostic_figures(
    real_validation,
    synthetic,
    *,
    acf_return: pd.DataFrame,
    acf_abs_return: pd.DataFrame,
    correlations: dict,
    validation_nn: np.ndarray,
    synthetic_nn: np.ndarray,
    output_directory: Path | str,
    prefix: str = "diffusion_seed42_normalized",
) -> dict[str, str]:
    real = as_numpy_windows(real_validation)
    generated = as_numpy_windows(synthetic)
    output = Path(output_directory)
    paths = {
        "return_distribution": output / f"{prefix}_return_distribution.png",
        "abs_return_acf": output / f"{prefix}_abs_return_acf.png",
        "return_acf": output / f"{prefix}_return_acf.png",
        "channel_distributions": output / f"{prefix}_channel_distributions.png",
        "correlations": output / f"{prefix}_correlations.png",
        "nearest_neighbor": output / f"{prefix}_nearest_neighbor.png",
    }

    real_return = real[:, :, 0].reshape(-1)
    synthetic_return = generated[:, :, 0].reshape(-1)
    bins = _shared_bins(real_return, synthetic_return)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(real_return, bins=bins, density=True, alpha=0.55, label="Donor validation")
    axis.hist(synthetic_return, bins=bins, density=True, alpha=0.55, label="Diffusion")
    axis.set_title("Normalized log-return distribution")
    axis.set_xlabel("Normalized log_return")
    axis.set_ylabel("Density")
    axis.legend()
    _save(figure, paths["return_distribution"])

    for table, key, title in (
        (acf_return, "return_acf", "Normalized log-return ACF"),
        (acf_abs_return, "abs_return_acf", "Normalized absolute log-return ACF"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(table["lag"], table["real_validation_acf"], marker="o", label="Donor validation")
        axis.plot(table["lag"], table["synthetic_acf"], marker="o", label="Diffusion")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Lag")
        axis.set_ylabel("Mean within-window ACF")
        axis.set_xticks(table["lag"])
        axis.legend()
        _save(figure, paths[key])

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_ORDER, strict=True)):
        real_values = real[:, :, index].reshape(-1)
        synthetic_values = generated[:, :, index].reshape(-1)
        bins = _shared_bins(real_values, synthetic_values)
        axis.hist(real_values, bins=bins, density=True, alpha=0.55, label="Validation")
        axis.hist(synthetic_values, bins=bins, density=True, alpha=0.55, label="Synthetic")
        axis.set_title(channel)
        axis.set_xlabel("Normalized value")
        if index == 0:
            axis.set_ylabel("Density")
        axis.legend()
    figure.suptitle("Normalized channel distributions")
    _save(figure, paths["channel_distributions"])

    matrices = (
        (correlations["real"], "Donor validation", -1.0, 1.0),
        (correlations["synthetic"], "Diffusion", -1.0, 1.0),
        (
            correlations["absolute_difference"],
            "Absolute difference",
            0.0,
            float(max(correlations["absolute_difference"].max(), 1e-12)),
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, (matrix, title, vmin, vmax) in zip(axes, matrices, strict=True):
        image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap="coolwarm")
        axis.set_xticks(range(3), CHANNEL_ORDER, rotation=35, ha="right")
        axis.set_yticks(range(3), CHANNEL_ORDER)
        axis.set_title(title)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Normalized cross-channel correlations")
    _save(figure, paths["correlations"])

    bins = _shared_bins(validation_nn, synthetic_nn, count=60)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(validation_nn, bins=bins, density=True, alpha=0.55, label="Validation → train")
    axis.hist(synthetic_nn, bins=bins, density=True, alpha=0.55, label="Synthetic → train")
    axis.set_title("Nearest-neighbour distance to donor train")
    axis.set_xlabel("Euclidean distance in flattened normalized space")
    axis.set_ylabel("Density")
    axis.legend()
    _save(figure, paths["nearest_neighbor"])

    return {name: path.as_posix() for name, path in paths.items()}
