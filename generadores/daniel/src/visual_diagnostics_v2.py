"""Pedagogical normalized-space diagnostics for Daniel's frozen Diffusion model.

These analyses are generator-local diagnostics, not the future common fidelity
implementation and not evidence of downstream forecasting utility.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .validation import CHANNEL_ORDER


BALANCED_COUNT = 380
DIAGNOSTIC_SEED = 42


def _as_windows(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or tuple(array.shape[1:]) != (65, 3):
        raise ValueError(f"{name} must have shape (N, 65, 3), got {array.shape}")
    if array.dtype not in (np.float32, np.float64) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite float32/float64 values")
    return array


def deterministic_balanced_samples(
    real: np.ndarray,
    synthetic_pool: np.ndarray,
    *,
    count: int = BALANCED_COUNT,
    seed: int = DIAGNOSTIC_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all requested real rows and a deterministic synthetic subset."""

    real_array = _as_windows(real, name="real")
    synthetic_array = _as_windows(synthetic_pool, name="synthetic_pool")
    if len(real_array) != count or len(synthetic_array) < count:
        raise ValueError("Balanced diagnostic requires 380 real and >=380 synthetic rows")
    indices = np.sort(
        np.random.default_rng(seed).choice(len(synthetic_array), size=count, replace=False)
    )
    return real_array.copy(), synthetic_array[indices].copy(), indices


def flatten_windows(values: np.ndarray) -> np.ndarray:
    """Convert `(N, 65, 3)` windows to the `(N, 195)` classifier contract."""

    array = _as_windows(values, name="windows")
    return array.reshape(len(array), 65 * 3)


def shared_histogram_bins(
    real_values: np.ndarray, synthetic_values: np.ndarray, *, n_bins: int = 80
) -> np.ndarray:
    """Use one set of bin edges so visual density comparisons are fair."""

    combined = np.concatenate(
        [np.asarray(real_values).reshape(-1), np.asarray(synthetic_values).reshape(-1)]
    )
    lower, upper = float(combined.min()), float(combined.max())
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    return np.linspace(lower, upper, n_bins + 1)


def marginal_statistics(real: np.ndarray, synthetic: np.ndarray) -> list[dict]:
    real_array = _as_windows(real, name="real")
    synthetic_array = _as_windows(synthetic, name="synthetic")
    return [
        {
            "channel": channel,
            "real_mean": float(real_array[:, :, index].mean()),
            "real_std": float(real_array[:, :, index].std(ddof=0)),
            "synthetic_mean": float(synthetic_array[:, :, index].mean()),
            "synthetic_std": float(synthetic_array[:, :, index].std(ddof=0)),
        }
        for index, channel in enumerate(CHANNEL_ORDER)
    ]


def plot_marginals(real: np.ndarray, synthetic: np.ndarray, output: Path | str) -> list[dict]:
    """Plot channel marginals; this intentionally ignores temporal dependence."""

    real_array = _as_windows(real, name="real")
    synthetic_array = _as_windows(synthetic, name="synthetic")
    stats = marginal_statistics(real_array, synthetic_array)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_ORDER, strict=True)):
        real_values = real_array[:, :, index].reshape(-1)
        synthetic_values = synthetic_array[:, :, index].reshape(-1)
        # Shared bins reveal shifts in centre, dispersion, and tails without a
        # plotting artefact caused by independently chosen histogram edges.
        # Marginals do NOT measure temporal order or cross-channel dependence.
        bins = shared_histogram_bins(real_values, synthetic_values)
        axis.hist(real_values, bins=bins, density=True, alpha=0.45, label="Real validation")
        axis.hist(
            synthetic_values,
            bins=bins,
            density=True,
            alpha=0.45,
            label="Diffusion synthetic",
        )
        row = stats[index]
        axis.set_title(channel)
        axis.text(
            0.02,
            0.98,
            f"real μ={row['real_mean']:.3f}, σ={row['real_std']:.3f}\n"
            f"synth μ={row['synthetic_mean']:.3f}, σ={row['synthetic_std']:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[-1].legend()
    figure.suptitle("Normalized donor validation vs Diffusion seed 42 marginals")
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return stats


def logistic_c2st(
    real: np.ndarray, synthetic: np.ndarray, *, random_state: int = DIAGNOSTIC_SEED
) -> dict:
    """Evaluate linear distinguishability exclusively with out-of-fold predictions."""

    # Each financial observation is a 65x3 window. LogisticRegression expects
    # a 2D feature matrix, so each observation becomes 195 ordered features.
    real_flat = flatten_windows(real)
    synthetic_flat = flatten_windows(synthetic)
    if len(real_flat) != len(synthetic_flat):
        raise ValueError("C2ST classes must be balanced")
    features = np.concatenate([real_flat, synthetic_flat], axis=0)
    labels = np.concatenate(
        [np.zeros(len(real_flat), dtype=np.int64), np.ones(len(synthetic_flat), dtype=np.int64)]
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    # The classifier is not part of the generator. Each held-out fold is scored
    # by a model that never fitted those rows, preventing in-sample AUC inflation.
    for fold_id, (train_indices, test_indices) in enumerate(folds.split(features, labels)):
        estimator = clone(pipeline)
        estimator.fit(features[train_indices], labels[train_indices])
        probabilities[test_indices] = estimator.predict_proba(features[test_indices])[:, 1]
        fold_ids[test_indices] = fold_id
    if np.isnan(probabilities).any() or (fold_ids < 0).any():
        raise RuntimeError("Every C2ST row must receive exactly one out-of-fold prediction")
    predictions = (probabilities >= 0.5).astype(np.int64)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "thresholds": thresholds,
        "probabilities": probabilities,
        "labels": labels,
        "fold_ids": fold_ids,
        "n_splits": 5,
        "evaluation": "out_of_fold",
    }


def plot_logistic_c2st(result: dict, output: Path | str) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        result["false_positive_rate"],
        result["true_positive_rate"],
        linewidth=2,
        label=f"OOF ROC-AUC = {result['roc_auc']:.3f}",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random = 0.50")
    # AUC near 0.50 means this linear classifier has little distinguishing
    # ability. Higher AUC indicates systematic differences, while AUC=0.50
    # never proves that the full real and synthetic distributions are equal.
    axis.text(
        0.98,
        0.04,
        f"Accuracy = {result['accuracy']:.3f}\n5-fold stratified OOF",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "grey"},
    )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Logistic real-vs-synthetic diagnostic")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def tsne_projection(
    real: np.ndarray, synthetic: np.ndarray, *, random_state: int = DIAGNOSTIC_SEED
) -> dict:
    real_flat = flatten_windows(real)
    synthetic_flat = flatten_windows(synthetic)
    if len(real_flat) != len(synthetic_flat):
        raise ValueError("t-SNE comparison requires balanced real/synthetic counts")
    features = np.concatenate([real_flat, synthetic_flat], axis=0)
    labels = np.concatenate(
        [np.zeros(len(real_flat), dtype=np.int64), np.ones(len(synthetic_flat), dtype=np.int64)]
    )
    standardized = StandardScaler().fit_transform(features)
    pca_components = min(30, standardized.shape[1], standardized.shape[0] - 1)
    # PCA first makes the nonlinear projection more stable and efficient by
    # reducing the 195-dimensional windows to a moderate linear representation.
    reduced = PCA(n_components=pca_components, random_state=random_state).fit_transform(
        standardized
    )
    started = perf_counter()
    # t-SNE maps the common 30D representation to 2D only for visualization.
    # Clear separation may suggest distribution shift, but visual overlap does
    # not prove equality. The embedding depends on its parameters and seed and
    # must never be converted into a model ranking.
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        max_iter=1000,
    ).fit_transform(reduced)
    return {
        "embedding": embedding,
        "labels": labels,
        "pca": {"n_components": pca_components, "random_state": random_state},
        "tsne": {
            "n_components": 2,
            "perplexity": 30,
            "init": "pca",
            "learning_rate": "auto",
            "random_state": random_state,
            "max_iter": 1000,
        },
        "runtime_seconds": perf_counter() - started,
    }


def plot_tsne(result: dict, output: Path | str) -> None:
    embedding = result["embedding"]
    labels = result["labels"]
    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.scatter(
        embedding[labels == 0, 0],
        embedding[labels == 0, 1],
        s=18,
        alpha=0.65,
        label="Real donor validation",
    )
    axis.scatter(
        embedding[labels == 1, 0],
        embedding[labels == 1, 1],
        s=18,
        alpha=0.55,
        label="Diffusion synthetic",
    )
    axis.set_title("Joint normalized-space t-SNE (visual diagnostic only)")
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.legend()
    axis.grid(alpha=0.15)
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
