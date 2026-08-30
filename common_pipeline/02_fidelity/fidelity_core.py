"""Shared, model-agnostic fidelity evaluation primitives.

All comparisons operate in the common donor-train-normalized space.  This
module deliberately contains no NVDA calibration and performs no clipping,
sign correction, winsorization, or re-standardization of synthetic outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CHANNEL_ORDER = (
    "log_return",
    "log_high_low_range",
    "log1p_volume",
)
WINDOW_LENGTH = 65
N_CHANNELS = len(CHANNEL_ORDER)
FEATURE_COUNT = WINDOW_LENGTH * N_CHANNELS
DONOR_TRAIN_COUNT = 4910
DONOR_VALIDATION_COUNT = 380
SYNTHETIC_POOL_COUNT = 5000
EVALUATION_SUBSET_SIZE = 380
EVALUATION_SUBSET_SEED = 42
MAX_ACF_LAG = 20
STD_THRESHOLD = 1e-8


@dataclass(frozen=True)
class GlobalChannelStatistics:
    """Float64 donor-train statistics for the common three-channel scaler."""

    mean: np.ndarray
    std: np.ndarray
    raw_std: np.ndarray


@dataclass(frozen=True)
class SyntheticPool:
    """Validated common synthetic output and its provenance."""

    method: str
    path: Path
    windows: np.ndarray
    training_seed: int
    space: str
    channel_order: tuple[str, ...]
    metadata_evidence: str


@dataclass(frozen=True)
class C2STResult:
    """Out-of-fold predictions and summary metrics from the common C2ST."""

    roc_auc: float
    accuracy: float
    probabilities: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    n_splits: int


def validate_windows(
    windows: np.ndarray,
    *,
    name: str,
    expected_count: int | None = None,
) -> np.ndarray:
    """Validate only the normalized tensor contract, never physical signs."""

    array = np.asarray(windows)
    if array.ndim != 3 or array.shape[1:] != (WINDOW_LENGTH, N_CHANNELS):
        raise ValueError(
            f"{name} must have shape (N, {WINDOW_LENGTH}, {N_CHANNELS}), "
            f"got {array.shape}"
        )
    if expected_count is not None and array.shape[0] != expected_count:
        raise ValueError(f"{name} must contain {expected_count} windows")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def reconstruct_windows(
    features_flat: Iterable[Sequence[float]],
    *,
    name: str,
    expected_count: int | None = None,
) -> np.ndarray:
    """Reconstruct session-major 195-value rows as ``(N, 65, 3)``."""

    rows = [np.asarray(row) for row in features_flat]
    if not rows:
        raise ValueError(f"{name} contains no windows")
    lengths = {row.size for row in rows}
    if lengths != {FEATURE_COUNT}:
        raise ValueError(f"{name} must contain exactly {FEATURE_COUNT} values per row")
    if any(not np.issubdtype(row.dtype, np.number) for row in rows):
        raise TypeError(f"{name} features_flat must be numeric")
    array = np.stack(rows, axis=0).reshape(-1, WINDOW_LENGTH, N_CHANNELS)
    return validate_windows(array, name=name, expected_count=expected_count)


def load_canonical_windows(path: Path | str, *, expected_count: int) -> np.ndarray:
    """Load one canonical donor split without importing generator code."""

    source = Path(path)
    expected_stem = {
        DONOR_TRAIN_COUNT: "donor_train",
        DONOR_VALIDATION_COUNT: "donor_validation",
    }.get(expected_count)
    if expected_stem is not None and source.stem != expected_stem:
        raise ValueError(
            "Common fidelity accepts only donor_train and donor_validation as real inputs"
        )
    frame = pd.read_parquet(source, columns=["features_flat"])
    return reconstruct_windows(
        frame["features_flat"], name=source.name, expected_count=expected_count
    ).astype(np.float64, copy=False)


def fit_global_channel_normalizer(train_windows: np.ndarray) -> GlobalChannelStatistics:
    """Fit the common float64, ddof=0 scaler over window and time axes."""

    train = validate_windows(train_windows, name="donor_train").astype(
        np.float64, copy=False
    )
    mean = np.mean(train, axis=(0, 1), dtype=np.float64)
    raw_std = np.std(train, axis=(0, 1), ddof=0, dtype=np.float64)
    std = np.where(raw_std < STD_THRESHOLD, 1.0, raw_std)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Global channel statistics contain NaN or Inf")
    return GlobalChannelStatistics(mean=mean, std=std, raw_std=raw_std)


def transform_global_channel(
    windows: np.ndarray, statistics: GlobalChannelStatistics
) -> np.ndarray:
    """Apply frozen train statistics and expose the common float32 space."""

    values = validate_windows(windows, name="windows").astype(np.float64, copy=False)
    transformed = (values - statistics.mean) / statistics.std
    transformed = transformed.astype(np.float32)
    return validate_windows(transformed, name="normalized_windows")


def load_real_reference(
    donor_train_path: Path | str,
    donor_validation_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, GlobalChannelStatistics]:
    """Fit on donor train and transform train/validation without validation refit."""

    train = load_canonical_windows(donor_train_path, expected_count=DONOR_TRAIN_COUNT)
    validation = load_canonical_windows(
        donor_validation_path, expected_count=DONOR_VALIDATION_COUNT
    )
    statistics = fit_global_channel_normalizer(train)
    return (
        transform_global_channel(train, statistics),
        transform_global_channel(validation, statistics),
        statistics,
    )


def _unique_scalar(frame: pd.DataFrame, column: str) -> object | None:
    if column not in frame.columns:
        return None

    def canonical(value: object) -> object:
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    values = [canonical(value) for value in frame[column].tolist()]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"Synthetic column {column!r} must contain one unique value")
    return first


def _sidecar_metadata(path: Path) -> dict:
    candidates = (
        path.with_name(f"{path.stem}_manifest.json"),
        path.with_suffix(".json"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def load_synthetic_pool(
    path: Path | str,
    *,
    method: str,
    expected_count: int = SYNTHETIC_POOL_COUNT,
    expected_normalizer: GlobalChannelStatistics | None = None,
) -> SyntheticPool:
    """Load one published normalized pool while tolerating current schemas."""

    source = Path(path)
    frame = pd.read_parquet(source)
    if "features_flat" not in frame.columns:
        raise ValueError(f"{source} lacks features_flat")
    windows = reconstruct_windows(
        frame["features_flat"], name=source.name, expected_count=expected_count
    )
    sidecar = _sidecar_metadata(source)

    seed_value = _unique_scalar(frame, "training_seed")
    if seed_value is None:
        seed_value = _unique_scalar(frame, "seed")
    if seed_value is None:
        seed_value = sidecar.get("training_seed", sidecar.get("seed"))
    if seed_value is None:
        match = re.search(r"seed[_-]?(\d+)", source.stem, flags=re.IGNORECASE)
        seed_value = int(match.group(1)) if match else None
    if int(seed_value) != 42:
        raise ValueError(f"{source} must be the published training-seed-42 pool")

    explicit_space = _unique_scalar(frame, "space")
    sidecar_space = sidecar.get("space")
    if explicit_space is not None:
        if explicit_space != "global_channel_normalized":
            raise ValueError(f"{source} is not in global_channel_normalized space")
        space = str(explicit_space)
        evidence = "parquet:space"
    elif sidecar_space is not None:
        normalized_text = str(sidecar_space).lower()
        if "z-score" not in normalized_text and "global_channel_normalized" not in normalized_text:
            raise ValueError(f"{source} sidecar does not certify normalized space")
        space = "global_channel_normalized"
        evidence = "sidecar:space"
    elif "normalized" in source.stem.lower():
        space = "global_channel_normalized"
        evidence = "published normalized filename plus canonical 195-value contract"
    else:
        raise ValueError(f"Cannot certify normalized space for {source}")

    explicit_channels = _unique_scalar(frame, "channel_order")
    sidecar_channels = sidecar.get("channels", sidecar.get("channel_order"))
    channel_order = tuple(explicit_channels or sidecar_channels or CHANNEL_ORDER)
    if channel_order != CHANNEL_ORDER:
        raise ValueError(f"{source} uses incompatible channel order {channel_order}")

    sidecar_mean = sidecar.get("scaler_mean")
    sidecar_std = sidecar.get("scaler_std")
    if (sidecar_mean is None) != (sidecar_std is None):
        raise ValueError(f"{source} sidecar contains an incomplete scaler contract")
    if sidecar_mean is not None and expected_normalizer is not None:
        published_mean = np.asarray(sidecar_mean, dtype=np.float64)
        published_std = np.asarray(sidecar_std, dtype=np.float64)
        if published_mean.shape != (N_CHANNELS,) or published_std.shape != (N_CHANNELS,):
            raise ValueError(f"{source} sidecar scaler must contain three channels")
        if not np.allclose(
            published_mean, expected_normalizer.mean, rtol=1e-10, atol=1e-12
        ) or not np.allclose(
            published_std, expected_normalizer.std, rtol=1e-10, atol=1e-12
        ):
            mean_delta = float(np.max(np.abs(published_mean - expected_normalizer.mean)))
            std_delta = float(np.max(np.abs(published_std - expected_normalizer.std)))
            raise ValueError(
                f"{source} sidecar scaler is not the common float64 donor-train "
                f"contract (max mean delta={mean_delta:.12g}, "
                f"max std delta={std_delta:.12g})"
            )
        evidence = f"{evidence}; sidecar scaler matches common float64 contract"

    return SyntheticPool(
        method=method,
        path=source,
        windows=windows,
        training_seed=42,
        space=space,
        channel_order=channel_order,
        metadata_evidence=evidence,
    )


def discover_neural_outputs(repository_root: Path | str) -> dict[str, Path]:
    """Discover one published normalized Parquet per generator owner."""

    root = Path(repository_root)
    discovered: dict[str, Path] = {}
    for output_dir in sorted((root / "generadores").glob("*/outputs")):
        candidates = sorted(output_dir.glob("*normalized*.parquet"))
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one normalized Parquet in {output_dir}, found {len(candidates)}"
            )
        discovered[output_dir.parent.name] = candidates[0]
    return discovered


def common_subset_indices(
    pool_size: int = SYNTHETIC_POOL_COUNT,
    *,
    subset_size: int = EVALUATION_SUBSET_SIZE,
    evaluation_subset_seed: int = EVALUATION_SUBSET_SEED,
) -> np.ndarray:
    """Generate the one shared set of row positions used by every method."""

    if subset_size <= 0 or subset_size > pool_size:
        raise ValueError("subset_size must be between 1 and pool_size")
    return np.random.default_rng(evaluation_subset_seed).choice(
        pool_size, size=subset_size, replace=False
    )


def apply_common_subset(
    pools: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    """Apply identical row positions, in identical order, to all methods."""

    selected: dict[str, np.ndarray] = {}
    positions = np.asarray(indices, dtype=np.int64)
    for method, windows in pools.items():
        array = validate_windows(windows, name=method)
        if positions.min() < 0 or positions.max() >= array.shape[0]:
            raise IndexError(f"Subset indices exceed pool {method!r}")
        selected[method] = array[positions]
    return selected


def marginal_statistics(datasets: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Compute pooled marginal summaries using fixed SciPy definitions."""

    rows: list[dict[str, object]] = []
    for method, windows in datasets.items():
        array = validate_windows(windows, name=method).astype(np.float64, copy=False)
        for index, channel in enumerate(CHANNEL_ORDER):
            values = array[:, :, index].reshape(-1)
            percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
            rows.append(
                {
                    "method": method,
                    "channel": channel,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "median": float(percentiles[3]),
                    "p01": float(percentiles[0]),
                    "p05": float(percentiles[1]),
                    "p25": float(percentiles[2]),
                    "p75": float(percentiles[4]),
                    "p95": float(percentiles[5]),
                    "p99": float(percentiles[6]),
                    "skewness": float(skew(values, bias=False)),
                    "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
                }
            )
    return pd.DataFrame(rows)


def wasserstein_table(
    real: np.ndarray, synthetic: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Compute empirical Wasserstein-1 independently for every channel."""

    reference = validate_windows(real, name="real")
    rows: list[dict[str, object]] = []
    for method, windows in synthetic.items():
        generated = validate_windows(windows, name=method)
        distances = [
            float(
                wasserstein_distance(
                    reference[:, :, index].reshape(-1),
                    generated[:, :, index].reshape(-1),
                )
            )
            for index in range(N_CHANNELS)
        ]
        mean_distance = float(np.mean(distances))
        for channel, distance in zip(CHANNEL_ORDER, distances):
            rows.append(
                {
                    "method": method,
                    "channel": channel,
                    "wasserstein_1": distance,
                    "mean_wasserstein_across_channels": mean_distance,
                }
            )
    return pd.DataFrame(rows)


def mean_window_acf(
    windows: np.ndarray,
    *,
    absolute: bool = False,
    max_lag: int = MAX_ACF_LAG,
) -> tuple[np.ndarray, np.ndarray]:
    """Average biased ACFs computed inside, never across, window boundaries."""

    array = validate_windows(windows, name="acf_windows").astype(np.float64, copy=False)
    if not 1 <= max_lag < WINDOW_LENGTH:
        raise ValueError("max_lag must be between 1 and 64")
    values = array[:, :, 0]
    if absolute:
        values = np.abs(values)
    centered = values - values.mean(axis=1, keepdims=True)
    denominator = np.sum(centered**2, axis=1)
    acf_values: list[float] = []
    valid_counts: list[int] = []
    for lag in range(1, max_lag + 1):
        valid = denominator > 0.0
        if not np.any(valid):
            raise ValueError("All windows have zero temporal variance")
        numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        acf_values.append(float(np.mean(numerator[valid] / denominator[valid])))
        valid_counts.append(int(np.count_nonzero(valid)))
    return np.asarray(acf_values), np.asarray(valid_counts)


def acf_table(
    real: np.ndarray,
    synthetic: Mapping[str, np.ndarray],
    *,
    absolute: bool,
    max_lag: int = MAX_ACF_LAG,
) -> pd.DataFrame:
    """Create comparable ACF curves and per-method lag-1..20 MAE."""

    real_acf, real_valid = mean_window_acf(real, absolute=absolute, max_lag=max_lag)
    rows: list[dict[str, object]] = []
    for method, windows in synthetic.items():
        method_acf, method_valid = mean_window_acf(
            windows, absolute=absolute, max_lag=max_lag
        )
        differences = method_acf - real_acf
        mae = float(np.mean(np.abs(differences)))
        for lag_index in range(max_lag):
            rows.append(
                {
                    "method": method,
                    "lag": lag_index + 1,
                    "real_acf": float(real_acf[lag_index]),
                    "synthetic_acf": float(method_acf[lag_index]),
                    "difference": float(differences[lag_index]),
                    "absolute_difference": float(abs(differences[lag_index])),
                    "acf_mae": mae,
                    "real_valid_windows": int(real_valid[lag_index]),
                    "synthetic_valid_windows": int(method_valid[lag_index]),
                }
            )
    return pd.DataFrame(rows)


def channel_correlation_matrix(windows: np.ndarray) -> np.ndarray:
    """Pearson matrix after flattening only sample and time dimensions."""

    flattened = validate_windows(windows, name="correlation_windows").reshape(
        -1, N_CHANNELS
    )
    matrix = np.corrcoef(flattened.astype(np.float64, copy=False), rowvar=False)
    if matrix.shape != (N_CHANNELS, N_CHANNELS) or not np.isfinite(matrix).all():
        raise ValueError("Channel correlation matrix is not finite 3x3")
    return matrix


def correlation_tables(
    real: np.ndarray, synthetic: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all matrices plus absolute off-diagonal errors versus real."""

    matrices = {"real": channel_correlation_matrix(real)}
    matrices.update(
        {method: channel_correlation_matrix(windows) for method, windows in synthetic.items()}
    )
    matrix_rows: list[dict[str, object]] = []
    for method, matrix in matrices.items():
        for row_index, row_channel in enumerate(CHANNEL_ORDER):
            for column_index, column_channel in enumerate(CHANNEL_ORDER):
                matrix_rows.append(
                    {
                        "method": method,
                        "row_channel": row_channel,
                        "column_channel": column_channel,
                        "correlation": float(matrix[row_index, column_index]),
                    }
                )
    real_matrix = matrices["real"]
    off_diagonal = ~np.eye(N_CHANNELS, dtype=bool)
    error_rows = []
    for method in synthetic:
        absolute_difference = np.abs(matrices[method] - real_matrix)
        error_rows.append(
            {
                "method": method,
                "mean_absolute_off_diagonal_difference": float(
                    absolute_difference[off_diagonal].mean()
                ),
                "max_absolute_off_diagonal_difference": float(
                    absolute_difference[off_diagonal].max()
                ),
            }
        )
    return pd.DataFrame(matrix_rows), pd.DataFrame(error_rows)


def nearest_train_distances(
    queries: np.ndarray,
    donor_train: np.ndarray,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Euclidean 195-D nearest-neighbour distances with bounded memory."""

    query = validate_windows(queries, name="queries").reshape(len(queries), FEATURE_COUNT)
    reference = validate_windows(donor_train, name="donor_train").reshape(
        len(donor_train), FEATURE_COUNT
    )
    query = query.astype(np.float64, copy=False)
    reference = reference.astype(np.float64, copy=False)
    reference_norm = np.sum(reference**2, axis=1)
    distances = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk_size):
        block = query[start : start + chunk_size]
        squared = (
            np.sum(block**2, axis=1, keepdims=True)
            + reference_norm[None, :]
            - 2.0 * block @ reference.T
        )
        distances[start : start + len(block)] = np.sqrt(
            np.maximum(np.min(squared, axis=1), 0.0)
        )
    return distances


def nearest_neighbor_table(
    donor_train: np.ndarray,
    real_validation: np.ndarray,
    synthetic: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Summarize validation/synthetic distance to normalized donor train."""

    datasets = {"real_validation": real_validation, **synthetic}
    rows = []
    for method, windows in datasets.items():
        values = nearest_train_distances(windows, donor_train)
        rows.append(
            {
                "method": method,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=0)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        )
    return pd.DataFrame(rows)


def c2st_out_of_fold(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    random_state: int = EVALUATION_SUBSET_SEED,
) -> C2STResult:
    """Run the frozen linear C2ST using only five-fold out-of-fold predictions."""

    real_array = validate_windows(real, name="c2st_real")
    synthetic_array = validate_windows(synthetic, name="c2st_synthetic")
    if len(real_array) != len(synthetic_array):
        raise ValueError("C2ST requires balanced real and synthetic counts")
    features = np.concatenate((real_array, synthetic_array), axis=0).reshape(
        -1, FEATURE_COUNT
    )
    labels = np.concatenate(
        (np.zeros(len(real_array), dtype=np.int8), np.ones(len(synthetic_array), dtype=np.int8))
    )
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(
        classifier, features, labels, cv=folds, method="predict_proba"
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    return C2STResult(
        roc_auc=float(roc_auc_score(labels, probabilities)),
        accuracy=float(accuracy_score(labels, predictions)),
        probabilities=probabilities,
        predictions=predictions,
        labels=labels,
        n_splits=5,
    )


def c2st_table(
    real: np.ndarray, synthetic: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Apply the identical balanced OOF C2ST to each synthetic method."""

    rows = []
    for method, windows in synthetic.items():
        result = c2st_out_of_fold(real, windows)
        rows.append(
            {
                "method": method,
                "real_count": len(real),
                "synthetic_count": len(windows),
                "features": FEATURE_COUNT,
                "cv": "StratifiedKFold(n_splits=5,shuffle=True,random_state=42)",
                "prediction_scope": "out_of_fold",
                "roc_auc": result.roc_auc,
                "accuracy": result.accuracy,
            }
        )
    return pd.DataFrame(rows)


def assemble_joint_embedding_input(
    real: np.ndarray,
    synthetic: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble one joint matrix, labels, and within-source row identifiers."""

    datasets = {"real": validate_windows(real, name="tsne_real"), **synthetic}
    counts = {method: len(windows) for method, windows in datasets.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Joint t-SNE requires equal counts, got {counts}")
    matrix = np.concatenate(list(datasets.values()), axis=0).reshape(-1, FEATURE_COUNT)
    labels = np.concatenate(
        [np.repeat(method, len(windows)) for method, windows in datasets.items()]
    )
    row_ids = np.concatenate([np.arange(len(windows)) for windows in datasets.values()])
    return matrix.astype(np.float64, copy=False), labels, row_ids


def joint_tsne_coordinates(
    real: np.ndarray,
    synthetic: Mapping[str, np.ndarray],
    *,
    random_state: int = EVALUATION_SUBSET_SEED,
    pca_components: int = 30,
    perplexity: float = 30.0,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """Fit one StandardScaler -> PCA -> t-SNE embedding across all methods."""

    matrix, labels, row_ids = assemble_joint_embedding_input(real, synthetic)
    if pca_components > min(matrix.shape[0] - 1, matrix.shape[1]):
        raise ValueError("pca_components exceeds the valid joint data dimension")
    standardized = StandardScaler().fit_transform(matrix)
    reduced = PCA(n_components=pca_components, random_state=random_state).fit_transform(
        standardized
    )
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
    ).fit_transform(reduced)
    return pd.DataFrame(
        {
            "method": labels,
            "source_row": row_ids,
            "tsne_1": coordinates[:, 0],
            "tsne_2": coordinates[:, 1],
        }
    )
