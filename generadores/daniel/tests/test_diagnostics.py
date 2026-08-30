from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from generadores.daniel.src.diagnostic_plots import generate_diagnostic_figures
from generadores.daniel.src.diagnostics import (
    DIAGNOSTIC_MANIFEST_FIELDS,
    acf_comparison,
    channel_statistics,
    cross_channel_correlations,
    diversity_diagnostics,
    load_sample_pool,
    mean_window_acf,
    memorization_diagnostics,
    nearest_neighbor_distances,
    save_sample_pool,
    wasserstein_1d,
    write_diagnostic_manifest,
)
from generadores.daniel.src.diffusion import GaussianDiffusion
from generadores.daniel.src.network import TemporalDenoiser
from generadores.daniel.src.sampler import DDPMSampler
from generadores.daniel.src.validation import CHANNEL_ORDER


def _windows(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, 65, 3)).astype(np.float32)


def test_small_pool_exact_count_reproducibility_and_round_trip(tmp_path) -> None:
    torch.manual_seed(5)
    model = TemporalDenoiser(base_channels=8, time_embedding_dim=16)
    sampler = DDPMSampler(GaussianDiffusion(model, steps=4))
    first = sampler.sample(7, seed=42)
    repeated = sampler.sample(7, seed=42)
    different = sampler.sample(7, seed=43)
    assert first.shape == (7, 65, 3)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)
    path = tmp_path / "pool.npz"
    digest = save_sample_pool(path, first, seed=42)
    loaded = load_sample_pool(path)
    assert len(digest) == 64
    assert loaded["samples"].shape == (7, 65, 3)
    assert loaded["seed"] == 42
    assert loaded["channel_order"] == CHANNEL_ORDER
    assert np.array_equal(loaded["samples"], first.numpy())


def test_channel_statistics_and_exact_empirical_wasserstein() -> None:
    real = _windows(4, 1)
    synthetic = _windows(5, 2)
    table = channel_statistics(real, synthetic)
    assert len(table) == 6
    assert set(table["source"]) == {"real_validation", "synthetic"}
    assert set(table["channel"]) == set(CHANNEL_ORDER)
    assert np.isfinite(table.select_dtypes(include="number")).all().all()
    assert wasserstein_1d(np.array([0.0, 1.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)


def test_acf_is_computed_inside_each_window_without_boundary_pairs() -> None:
    windows = np.zeros((2, 65, 3), dtype=np.float64)
    windows[0, :, 0] = np.arange(65, dtype=np.float64)
    windows[1, :, 0] = np.arange(65, dtype=np.float64)[::-1] + 10_000.0
    acf, counts = mean_window_acf(windows, max_lag=2)
    manual = []
    for lag in (1, 2):
        per_window = []
        for window in windows[:, :, 0]:
            centered = window - window.mean()
            per_window.append(
                np.sum(centered[:-lag] * centered[lag:]) / np.sum(centered**2)
            )
        manual.append(np.mean(per_window))
    assert np.allclose(acf, manual)
    assert np.array_equal(counts, [2, 2])


def test_nearest_neighbor_dimensions_memorization_and_diversity() -> None:
    train = np.stack(
        [np.zeros((65, 3), dtype=np.float32), np.ones((65, 3), dtype=np.float32)]
    )
    validation = np.full((1, 65, 3), 0.5, dtype=np.float32)
    synthetic = np.stack(
        [
            np.zeros((65, 3), dtype=np.float32),
            np.full((65, 3), 2.0, dtype=np.float32),
            np.full((65, 3), 2.0, dtype=np.float32),
        ]
    )
    distances = nearest_neighbor_distances(synthetic, train, chunk_size=2)
    assert distances.shape == (3,)
    assert distances[0] == 0.0
    assert distances[1] == pytest.approx(np.sqrt(195.0))
    memory = memorization_diagnostics(synthetic, train, validation)
    assert memory["exact_duplicate_count"] == 1
    assert memory["near_exact_count_including_exact"] == 1
    diversity = diversity_diagnostics(synthetic, subsample_size=3)
    assert diversity["n_unique"] == 2
    assert diversity["duplicate_count"] == 1
    assert all(value > 0.0 for value in diversity["variance_by_channel"].values())


def test_manifest_and_all_plots_are_constructible_from_inputs(tmp_path) -> None:
    real = _windows(4, 10)
    synthetic = _windows(5, 11)
    acf_return = acf_comparison(real, synthetic, max_lag=20)
    acf_abs = acf_comparison(real, synthetic, max_lag=20, absolute=True)
    correlations = cross_channel_correlations(real, synthetic)
    paths = generate_diagnostic_figures(
        real,
        synthetic,
        acf_return=acf_return,
        acf_abs_return=acf_abs,
        correlations=correlations,
        validation_nn=np.array([1.0, 2.0, 3.0]),
        synthetic_nn=np.array([1.5, 2.5, 3.5]),
        output_directory=tmp_path / "figures",
    )
    assert len(paths) == 6
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths.values())

    manifest = {field: "test" for field in DIAGNOSTIC_MANIFEST_FIELDS}
    manifest_path = tmp_path / "manifest.json"
    write_diagnostic_manifest(manifest, manifest_path)
    assert manifest_path.is_file()
    del manifest["sample_sha256"]
    with pytest.raises(ValueError, match="missing"):
        write_diagnostic_manifest(manifest, manifest_path)


def test_sampling_entrypoint_has_no_target_or_downstream_artifact_access() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/sample_diagnostics.py"
    source = path.read_text(encoding="utf-8").lower()
    forbidden = (
        "windows/nvda",
        "test_index.parquet",
        "nvda_visible",
        "nvda_hidden",
        "nvda_full_history",
        "ridge",
    )
    assert all(value not in source for value in forbidden)
