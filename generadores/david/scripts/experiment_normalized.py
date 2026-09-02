#!/usr/bin/env python3
"""Generate and evaluate improved normalized candidates for David.

LEGACY EXPERIMENT
NOT PART OF FINAL NORMALIZING FLOW
DO NOT USE FOR MODEL SELECTION
DO NOT PROMOTE OUTPUTS

This module belongs to the historical temporal-jitter / candidate-search
experiment that predates the official RealNVP Normalizing Flow. It is kept
only as historical evidence. The official David output lives in
``generadores/david/outputs`` and is produced exclusively by
``train_normalizing_flow.py`` + ``generate_normalized.py``. This script
writes experimental Parquets under ``generadores/david/experiments`` so the
common contract discovery never mistakes them for deliverables, and it has
no capability to write or overwrite the official output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_MODULE = REPO_ROOT / "common_pipeline" / "01_contract"
FIDELITY_MODULE = REPO_ROOT / "common_pipeline" / "02_fidelity"
for module_path in (REPO_ROOT, CONTRACT_MODULE, FIDELITY_MODULE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from constants import (  # noqa: E402
    CANONICAL_MEAN,
    CANONICAL_STD,
    CHANNEL_ORDER,
    DONOR_TRAIN_PATH,
    EXPECTED_ROWS,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from io_utils import flatten_window, sha256_file, write_json  # noqa: E402
from normalizer import load_donor_train_normalized  # noqa: E402

from fidelity_core import (  # noqa: E402
    acf_table,
    apply_common_subset,
    c2st_table,
    common_subset_indices,
    correlation_tables,
    load_real_reference,
    nearest_neighbor_table,
    wasserstein_table,
)

features_target = importlib.import_module("common_pipeline.03_utility.features_target")
build_mixtures = importlib.import_module("common_pipeline.03_utility.build_mixtures")
validate_physical = importlib.import_module("common_pipeline.03_utility.validate_physical")
downstream_ridge = importlib.import_module("common_pipeline.03_utility.downstream_ridge")

EXPERIMENT_ROOT = REPO_ROOT / "generadores" / "david" / "experiments"
EXPERIMENT_OUTPUTS_DIR = EXPERIMENT_ROOT / "outputs"
EXPERIMENT_RESULTS_DIR = EXPERIMENT_ROOT / "results"
OFFICIAL_OUTPUT = REPO_ROOT / "generadores" / "david" / "outputs" / "normalizing_flow_seed42_normalized.parquet"
SEED = 42


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    noise_scale: float = 0.0
    rho: float = 0.0
    pca_components: int = 32
    gmm_components: int = 12


@dataclass(frozen=True)
class CandidateArtifact:
    spec: CandidateSpec
    path: Path
    sha256: str
    windows: np.ndarray
    seed: int


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("baseline_jitter_0p05", "bootstrap_independent", noise_scale=0.05),
    CandidateSpec("bootstrap_jitter_0p02", "bootstrap_independent", noise_scale=0.02),
    CandidateSpec("bootstrap_jitter_0p075", "bootstrap_independent", noise_scale=0.075),
    CandidateSpec("bootstrap_jitter_0p10", "bootstrap_independent", noise_scale=0.10),
    CandidateSpec("bootstrap_jitter_0p125", "bootstrap_independent", noise_scale=0.125),
    CandidateSpec("bootstrap_jitter_0p15", "bootstrap_independent", noise_scale=0.15),
    CandidateSpec("correlated_jitter_0p10", "bootstrap_temporal", noise_scale=0.10, rho=0.0),
    CandidateSpec("correlated_jitter_0p125", "bootstrap_temporal", noise_scale=0.125, rho=0.0),
    CandidateSpec("temporal_jitter_0p03_rho0p85", "bootstrap_temporal", noise_scale=0.03, rho=0.85),
    CandidateSpec("temporal_jitter_0p05_rho0p85", "bootstrap_temporal", noise_scale=0.05, rho=0.85),
    CandidateSpec("temporal_jitter_0p075_rho0p85", "bootstrap_temporal", noise_scale=0.075, rho=0.85),
    CandidateSpec("temporal_jitter_0p10_rho0p85", "bootstrap_temporal", noise_scale=0.10, rho=0.85),
    CandidateSpec("temporal_jitter_0p10_rho0p65", "bootstrap_temporal", noise_scale=0.10, rho=0.65),
    CandidateSpec("temporal_jitter_0p10_rho0p95", "bootstrap_temporal", noise_scale=0.10, rho=0.95),
    CandidateSpec("temporal_jitter_0p125_rho0p85", "bootstrap_temporal", noise_scale=0.125, rho=0.85),
    CandidateSpec("temporal_jitter_0p15_rho0p85", "bootstrap_temporal", noise_scale=0.15, rho=0.85),
    CandidateSpec("temporal_jitter_0p175_rho0p85", "bootstrap_temporal", noise_scale=0.175, rho=0.85),
    CandidateSpec("temporal_jitter_0p20_rho0p85", "bootstrap_temporal", noise_scale=0.20, rho=0.85),
    CandidateSpec("temporal_jitter_0p25_rho0p85", "bootstrap_temporal", noise_scale=0.25, rho=0.85),
    CandidateSpec("temporal_jitter_0p30_rho0p85", "bootstrap_temporal", noise_scale=0.30, rho=0.85),
    CandidateSpec("temporal_jitter_0p40_rho0p85", "bootstrap_temporal", noise_scale=0.40, rho=0.85),
    CandidateSpec("temporal_jitter_0p50_rho0p85", "bootstrap_temporal", noise_scale=0.50, rho=0.85),
    CandidateSpec("regime_temporal_jitter_0p03_rho0p85", "regime_bootstrap_temporal", noise_scale=0.03, rho=0.85),
    CandidateSpec("regime_temporal_jitter_0p05_rho0p85", "regime_bootstrap_temporal", noise_scale=0.05, rho=0.85),
    CandidateSpec("regime_temporal_jitter_0p075_rho0p85", "regime_bootstrap_temporal", noise_scale=0.075, rho=0.85),
    CandidateSpec("regime_temporal_jitter_0p10_rho0p85", "regime_bootstrap_temporal", noise_scale=0.10, rho=0.85),
    CandidateSpec("regime_mixup_jitter_0p01_rho0p85", "regime_neighbor_mixup_temporal", noise_scale=0.01, rho=0.85),
    CandidateSpec("regime_mixup_jitter_0p02_rho0p85", "regime_neighbor_mixup_temporal", noise_scale=0.02, rho=0.85),
    CandidateSpec("regime_mixup_jitter_0p03_rho0p85", "regime_neighbor_mixup_temporal", noise_scale=0.03, rho=0.85),
    CandidateSpec("pca_gmm_24c8", "pca_gmm", pca_components=24, gmm_components=8),
    CandidateSpec("pca_gmm_32c12", "pca_gmm", pca_components=32, gmm_components=12),
)


def candidate_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in CANDIDATES)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _safe_corr(windows: np.ndarray) -> np.ndarray:
    flat = windows.reshape(-1, N_CHANNELS).astype(np.float64, copy=False)
    corr = np.corrcoef(flat, rowvar=False)
    values, vectors = np.linalg.eigh(corr)
    values = np.clip(values, 1e-6, None)
    psd = (vectors * values) @ vectors.T
    scale = np.sqrt(np.diag(psd))
    return psd / np.outer(scale, scale)


def _draw_temporal_noise(
    rng: np.random.Generator,
    *,
    shape: tuple[int, int, int],
    channel_corr: np.ndarray,
    noise_scale: float,
    rho: float,
) -> np.ndarray:
    if noise_scale == 0.0:
        return np.zeros(shape, dtype=np.float32)
    cov = channel_corr * (noise_scale**2)
    innovations = rng.multivariate_normal(
        mean=np.zeros(N_CHANNELS),
        cov=cov,
        size=(shape[0], shape[1]),
    ).astype(np.float32)
    if rho <= 0.0:
        return innovations

    noise = np.empty_like(innovations)
    noise[:, 0, :] = innovations[:, 0, :]
    innovation_weight = float(np.sqrt(1.0 - rho**2))
    for step in range(1, shape[1]):
        noise[:, step, :] = rho * noise[:, step - 1, :] + innovation_weight * innovations[:, step, :]
    return noise


def _summary_features(windows: np.ndarray) -> np.ndarray:
    returns = windows[:, :, 0].astype(np.float64, copy=False)
    ranges = windows[:, :, 1].astype(np.float64, copy=False)
    volumes = windows[:, :, 2].astype(np.float64, copy=False)
    summary = np.stack(
        [
            returns.std(axis=1, ddof=0),
            np.mean(np.abs(returns), axis=1),
            returns.sum(axis=1),
            ranges.mean(axis=1),
            volumes.mean(axis=1),
            volumes.std(axis=1, ddof=0),
        ],
        axis=1,
    )
    center = summary.mean(axis=0)
    scale = np.where(summary.std(axis=0, ddof=0) < 1e-8, 1.0, summary.std(axis=0, ddof=0))
    return (summary - center) / scale


def _quantile_bins(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    thresholds = np.quantile(values, quantiles)
    return np.searchsorted(thresholds, values, side="right").astype(np.int64)


def _regime_labels(windows: np.ndarray) -> np.ndarray:
    returns = windows[:, :, 0].astype(np.float64, copy=False)
    ranges = windows[:, :, 1].astype(np.float64, copy=False)
    volumes = windows[:, :, 2].astype(np.float64, copy=False)
    vol_score = 0.5 * returns.std(axis=1, ddof=0) + 0.5 * ranges.mean(axis=1)
    trend_score = returns.sum(axis=1)
    volume_score = volumes.mean(axis=1)
    vol_bin = _quantile_bins(vol_score, (1 / 3, 2 / 3))
    trend_bin = _quantile_bins(trend_score, (1 / 3, 2 / 3))
    volume_bin = _quantile_bins(volume_score, (0.5,))
    return vol_bin * 6 + trend_bin * 2 + volume_bin


def _draw_regime_indices(
    rng: np.random.Generator,
    labels: np.ndarray,
    *,
    n_windows: int,
) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    chosen_regimes = rng.choice(unique, size=n_windows, replace=True, p=probabilities)
    indices = np.empty(n_windows, dtype=np.int64)
    for regime in unique:
        positions = np.flatnonzero(chosen_regimes == regime)
        donor_positions = np.flatnonzero(labels == regime)
        if len(positions):
            indices[positions] = rng.choice(donor_positions, size=len(positions), replace=True)
    return indices


def _nearest_by_regime(windows: np.ndarray, labels: np.ndarray, *, n_neighbors: int = 12) -> list[np.ndarray]:
    summary = _summary_features(windows)
    neighbors: list[np.ndarray] = [np.array([idx], dtype=np.int64) for idx in range(len(windows))]
    for regime in np.unique(labels):
        member_indices = np.flatnonzero(labels == regime)
        if len(member_indices) < 2:
            continue
        local_k = min(n_neighbors, len(member_indices))
        nn = NearestNeighbors(n_neighbors=local_k)
        nn.fit(summary[member_indices])
        local_neighbors = nn.kneighbors(return_distance=False)
        for row_position, neighbor_positions in enumerate(local_neighbors):
            global_index = member_indices[row_position]
            candidates = member_indices[neighbor_positions]
            candidates = candidates[candidates != global_index]
            neighbors[global_index] = candidates if len(candidates) else np.array([global_index], dtype=np.int64)
    return neighbors


def _bootstrap_independent(
    donor: np.ndarray,
    spec: CandidateSpec,
    *,
    seed: int,
    n_windows: int,
) -> np.ndarray:
    rng = _rng(seed)
    indices = rng.integers(0, len(donor), size=n_windows)
    bootstrap = donor[indices].astype(np.float32, copy=True)
    noise = rng.normal(0.0, spec.noise_scale, size=bootstrap.shape).astype(np.float32)
    return bootstrap + noise


def _bootstrap_temporal(
    donor: np.ndarray,
    spec: CandidateSpec,
    *,
    seed: int,
    n_windows: int,
) -> np.ndarray:
    rng = _rng(seed)
    indices = rng.integers(0, len(donor), size=n_windows)
    bootstrap = donor[indices].astype(np.float32, copy=True)
    noise = _draw_temporal_noise(
        rng,
        shape=bootstrap.shape,
        channel_corr=_safe_corr(donor),
        noise_scale=spec.noise_scale,
        rho=spec.rho,
    )
    return bootstrap + noise


def _regime_bootstrap_temporal(
    donor: np.ndarray,
    spec: CandidateSpec,
    *,
    seed: int,
    n_windows: int,
) -> np.ndarray:
    rng = _rng(seed)
    labels = _regime_labels(donor)
    indices = _draw_regime_indices(rng, labels, n_windows=n_windows)
    bootstrap = donor[indices].astype(np.float32, copy=True)
    noise = _draw_temporal_noise(
        rng,
        shape=bootstrap.shape,
        channel_corr=_safe_corr(donor),
        noise_scale=spec.noise_scale,
        rho=spec.rho,
    )
    return bootstrap + noise


def _regime_neighbor_mixup_temporal(
    donor: np.ndarray,
    spec: CandidateSpec,
    *,
    seed: int,
    n_windows: int,
) -> np.ndarray:
    rng = _rng(seed)
    labels = _regime_labels(donor)
    base_indices = _draw_regime_indices(rng, labels, n_windows=n_windows)
    neighbors = _nearest_by_regime(donor, labels)
    mate_indices = np.array([rng.choice(neighbors[index]) for index in base_indices], dtype=np.int64)
    lambdas = rng.beta(8.0, 2.0, size=n_windows)
    lambdas = np.where(rng.random(n_windows) < 0.5, lambdas, 1.0 - lambdas)
    mixed = (
        lambdas.reshape(-1, 1, 1) * donor[base_indices]
        + (1.0 - lambdas).reshape(-1, 1, 1) * donor[mate_indices]
    ).astype(np.float32)
    noise = _draw_temporal_noise(
        rng,
        shape=mixed.shape,
        channel_corr=_safe_corr(donor),
        noise_scale=spec.noise_scale,
        rho=spec.rho,
    )
    return mixed + noise


def _pca_gmm(
    donor: np.ndarray,
    spec: CandidateSpec,
    *,
    seed: int,
    n_windows: int,
) -> np.ndarray:
    flat = donor.reshape(len(donor), -1).astype(np.float64, copy=False)
    pca = PCA(n_components=spec.pca_components, random_state=seed)
    latent = pca.fit_transform(flat)
    gmm = GaussianMixture(
        n_components=spec.gmm_components,
        covariance_type="full",
        reg_covar=1e-4,
        max_iter=500,
        n_init=2,
        random_state=seed,
    )
    gmm.fit(latent)
    samples, _ = gmm.sample(n_windows)
    generated = pca.inverse_transform(samples).reshape(n_windows, WINDOW_LENGTH, N_CHANNELS)
    return generated.astype(np.float32)


def generate_candidate_windows(
    donor: np.ndarray,
    spec: CandidateSpec,
    *,
    seed: int = SEED,
    n_windows: int = EXPECTED_ROWS,
) -> np.ndarray:
    generators = {
        "bootstrap_independent": _bootstrap_independent,
        "bootstrap_temporal": _bootstrap_temporal,
        "regime_bootstrap_temporal": _regime_bootstrap_temporal,
        "regime_neighbor_mixup_temporal": _regime_neighbor_mixup_temporal,
        "pca_gmm": _pca_gmm,
    }
    return generators[spec.family](donor, spec, seed=seed, n_windows=n_windows)


def make_canonical_frame(
    windows: np.ndarray,
    *,
    source_model: str,
    training_seed: int = SEED,
) -> pd.DataFrame:
    records = []
    for synthetic_id, window in enumerate(windows):
        records.append(
            {
                "synthetic_id": synthetic_id,
                "source_model": source_model,
                "training_seed": training_seed,
                "space": GLOBAL_NORMALIZED_SPACE,
                "window_length": WINDOW_LENGTH,
                "n_channels": N_CHANNELS,
                "channel_order": list(CHANNEL_ORDER),
                "features_flat": flatten_window(window),
            }
        )
    return pd.DataFrame.from_records(records)


def write_candidate(
    donor: np.ndarray,
    spec: CandidateSpec,
    *,
    output_dir: Path = EXPERIMENT_OUTPUTS_DIR,
    seed: int = SEED,
    n_windows: int = EXPECTED_ROWS,
) -> CandidateArtifact:
    windows = generate_candidate_windows(donor, spec, seed=seed, n_windows=n_windows)
    frame = make_canonical_frame(windows, source_model=spec.name, training_seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec.name}_seed{seed}_normalized.parquet"
    frame.to_parquet(path, index=False)
    digest = sha256_file(path)
    write_json(
        path.with_suffix(".provenance.json"),
        {
            "source_model": spec.name,
            "family": spec.family,
            "algorithm": spec.family,
            "seed": seed,
            "training_seed": seed,
            "noise_scale": spec.noise_scale,
            "rho": spec.rho,
            "pca_components": spec.pca_components if spec.family == "pca_gmm" else None,
            "gmm_components": spec.gmm_components if spec.family == "pca_gmm" else None,
            "space": GLOBAL_NORMALIZED_SPACE,
            "channel_order": list(CHANNEL_ORDER),
            "mean": list(CANONICAL_MEAN),
            "std": list(CANONICAL_STD),
            "donor_train_path": "data/features/windows/donor_train.parquet",
            "donor_train_sha256": sha256_file(DONOR_TRAIN_PATH),
            "n_windows": n_windows,
            "logical_shape": [n_windows, WINDOW_LENGTH, N_CHANNELS],
            "parquet_sha256": digest,
        },
    )
    return CandidateArtifact(spec=spec, path=path, sha256=digest, windows=windows, seed=seed)


def _load_candidate_windows(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path, columns=["features_flat"])
    return np.stack(
        [
            np.asarray(row, dtype=np.float32).reshape(WINDOW_LENGTH, N_CHANNELS)
            for row in frame["features_flat"]
        ],
        axis=0,
    )


def _fidelity_summary(artifacts: list[CandidateArtifact]) -> pd.DataFrame:
    donor_train, real_validation, statistics = load_real_reference(
        REPO_ROOT / "data/features/windows/donor_train.parquet",
        REPO_ROOT / "data/features/windows/donor_validation.parquet",
    )
    pools = {artifact.spec.name: artifact.windows for artifact in artifacts}
    subset_indices = common_subset_indices()
    subset = apply_common_subset(pools, subset_indices)
    wasserstein = wasserstein_table(real_validation, subset)
    c2st = c2st_table(real_validation, subset)
    return_acf = acf_table(real_validation, subset, absolute=False)
    abs_return_acf = acf_table(real_validation, subset, absolute=True)
    nearest = nearest_neighbor_table(donor_train, real_validation, subset)
    _, corr_errors = correlation_tables(real_validation, subset)

    rows = []
    for artifact in artifacts:
        method = artifact.spec.name
        method_wass = wasserstein[wasserstein["method"] == method]
        method_c2st = c2st[c2st["method"] == method].iloc[0]
        method_return_acf = return_acf[return_acf["method"] == method]["acf_mae"].iloc[0]
        method_abs_acf = abs_return_acf[abs_return_acf["method"] == method]["acf_mae"].iloc[0]
        method_nn = nearest[nearest["method"] == method].iloc[0]
        method_corr = corr_errors[corr_errors["method"] == method].iloc[0]
        rows.append(
            {
                "method": method,
                "family": artifact.spec.family,
                "sha256": artifact.sha256,
                "path": artifact.path.relative_to(REPO_ROOT).as_posix(),
                "c2st_auc": float(method_c2st["roc_auc"]),
                "c2st_distance_from_0p5": abs(float(method_c2st["roc_auc"]) - 0.5),
                "c2st_accuracy": float(method_c2st["accuracy"]),
                "mean_wasserstein": float(method_wass["mean_wasserstein_across_channels"].iloc[0]),
                "return_acf_mae": float(method_return_acf),
                "abs_return_acf_mae": float(method_abs_acf),
                "corr_mae": float(method_corr["mean_absolute_off_diagonal_difference"]),
                "nearest_train_median": float(method_nn["median"]),
                "nearest_train_mean": float(method_nn["mean"]),
                "normalizer_mean_0": float(statistics.mean[0]),
            }
        )
    return pd.DataFrame(rows)


def _utility_summary(artifacts: list[CandidateArtifact]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu, sigma = validate_physical.load_calibration()
    real_windows = features_target.load_real_visible_windows()
    scaler = build_mixtures.fit_downstream_scaler(real_windows)
    test_windows = features_target.load_real_test_windows()
    x_test, y_test = features_target.build_features_and_target(test_windows)
    x_test_scaled = scaler.transform(x_test)

    physical_rows = []
    raw_rows = []
    for artifact in artifacts:
        calibrated = validate_physical.calibrate(artifact.windows, mu, sigma)
        valid_mask = np.array([validate_physical.validate_window(window) for window in calibrated])
        valid_pool = calibrated[valid_mask]
        physical_rows.append(
            {
                "method": artifact.spec.name,
                "generated": len(calibrated),
                "valid": int(valid_mask.sum()),
                "invalid": int((~valid_mask).sum()),
                "invalid_rate": float((~valid_mask).mean()),
            }
        )

        for ratio in build_mixtures.RATIOS:
            n_synth = build_mixtures.n_synthetic_for_ratio(build_mixtures.N_REAL, ratio)
            if n_synth == 0:
                x_train, y_train = features_target.build_features_and_target(real_windows)
                rmse, mae = downstream_ridge.evaluate_one(x_train, y_train, scaler, x_test_scaled, y_test)
                raw_rows.append(
                    {
                        "method": artifact.spec.name,
                        "ratio": ratio,
                        "subsampling_seed": None,
                        "rmse": rmse,
                        "mae": mae,
                    }
                )
                continue
            for seed in build_mixtures.SUBSAMPLING_SEEDS:
                mix = build_mixtures.build_mixture(real_windows, valid_pool, ratio, seed)
                x_train, y_train = features_target.build_features_and_target(mix)
                rmse, mae = downstream_ridge.evaluate_one(x_train, y_train, scaler, x_test_scaled, y_test)
                raw_rows.append(
                    {
                        "method": artifact.spec.name,
                        "ratio": ratio,
                        "subsampling_seed": seed,
                        "rmse": rmse,
                        "mae": mae,
                    }
                )

    raw = pd.DataFrame(raw_rows)
    physical = pd.DataFrame(physical_rows)
    summary_rows = []
    for method in raw["method"].unique():
        real_only = raw[(raw["method"] == method) & (raw["ratio"] == 0.0)].iloc[0]
        base_rmse = float(real_only["rmse"])
        base_mae = float(real_only["mae"])
        for ratio in build_mixtures.RATIOS:
            subset = raw[(raw["method"] == method) & (raw["ratio"] == ratio)]
            mean_rmse = float(subset["rmse"].mean())
            mean_mae = float(subset["mae"].mean())
            summary_rows.append(
                {
                    "method": method,
                    "ratio": ratio,
                    "mean_rmse": mean_rmse,
                    "std_rmse": float(subset["rmse"].std(ddof=0)) if len(subset) > 1 else 0.0,
                    "mean_mae": mean_mae,
                    "std_mae": float(subset["mae"].std(ddof=0)) if len(subset) > 1 else 0.0,
                    "delta_rmse_pct_vs_real_only": (mean_rmse - base_rmse) / base_rmse * 100.0,
                    "delta_mae_pct_vs_real_only": (mean_mae - base_mae) / base_mae * 100.0,
                }
            )
    return physical, pd.DataFrame(summary_rows), raw


def compare_artifacts(artifacts: list[CandidateArtifact], *, results_dir: Path = EXPERIMENT_RESULTS_DIR) -> pd.DataFrame:
    fidelity = _fidelity_summary(artifacts)
    physical, utility, raw = _utility_summary(artifacts)
    best_utility = (
        utility.sort_values(["method", "mean_rmse"])
        .groupby("method", as_index=False)
        .first()
        .rename(
            columns={
                "ratio": "best_ratio",
                "mean_rmse": "best_mean_rmse",
                "std_rmse": "best_std_rmse",
                "mean_mae": "best_mean_mae",
                "std_mae": "best_std_mae",
                "delta_rmse_pct_vs_real_only": "best_delta_rmse_pct_vs_real_only",
                "delta_mae_pct_vs_real_only": "best_delta_mae_pct_vs_real_only",
            }
        )
    )
    comparison = (
        fidelity.merge(physical, on="method", how="left")
        .merge(
            best_utility[
                [
                    "method",
                    "best_ratio",
                    "best_mean_rmse",
                    "best_std_rmse",
                    "best_mean_mae",
                    "best_std_mae",
                    "best_delta_rmse_pct_vs_real_only",
                    "best_delta_mae_pct_vs_real_only",
                ]
            ],
            on="method",
            how="left",
        )
        .sort_values(["c2st_distance_from_0p5", "mean_wasserstein", "best_mean_rmse"])
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    fidelity.to_csv(results_dir / "david_candidate_fidelity.csv", index=False, float_format="%.12g")
    physical.to_csv(results_dir / "david_candidate_physical.csv", index=False, float_format="%.12g")
    utility.to_csv(results_dir / "david_candidate_utility_summary.csv", index=False, float_format="%.12g")
    raw.to_csv(results_dir / "david_candidate_utility_raw.csv", index=False, float_format="%.12g")
    comparison.to_csv(results_dir / "david_candidate_comparison.csv", index=False, float_format="%.12g")
    return comparison


def _selected_specs(names: list[str] | None) -> tuple[CandidateSpec, ...]:
    if not names:
        return CANDIDATES
    by_name = {spec.name: spec for spec in CANDIDATES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}. Available: {sorted(by_name)}")
    return tuple(by_name[name] for name in names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", choices=candidate_names())
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_OUTPUTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_RESULTS_DIR)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse candidate Parquets already present in the experiment output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    donor = load_donor_train_normalized(DONOR_TRAIN_PATH)
    specs = _selected_specs(args.candidate)

    artifacts: list[CandidateArtifact] = []
    for spec in specs:
        path = args.output_dir / f"{spec.name}_seed42_normalized.parquet"
        if args.skip_generation:
            windows = _load_candidate_windows(path)
            digest = sha256_file(path)
            artifact = CandidateArtifact(spec=spec, path=path, sha256=digest, windows=windows, seed=SEED)
        else:
            artifact = write_candidate(donor, spec, output_dir=args.output_dir)
        artifacts.append(artifact)
        print(f"{spec.name}: {artifact.path} sha256={artifact.sha256}")

    comparison = compare_artifacts(artifacts, results_dir=args.results_dir)
    print("\nCandidate comparison:")
    print(
        comparison[
            [
                "method",
                "c2st_auc",
                "mean_wasserstein",
                "return_acf_mae",
                "abs_return_acf_mae",
                "invalid_rate",
                "best_ratio",
                "best_mean_rmse",
            ]
        ].to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
