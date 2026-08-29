"""Generate and diagnose 1,000 normalized DDPM samples from the best seed-42 model."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.data_adapter import load_canonical_donor_tensors  # noqa: E402
from generadores.daniel.src.diagnostic_plots import generate_diagnostic_figures  # noqa: E402
from generadores.daniel.src.diagnostics import (  # noqa: E402
    acf_comparison,
    channel_statistics,
    correlation_table,
    cross_channel_correlations,
    diversity_diagnostics,
    load_sample_pool,
    memorization_diagnostics,
    save_sample_pool,
    sha256_file,
    wasserstein_by_channel,
    write_diagnostic_manifest,
)
from generadores.daniel.src.diffusion import GaussianDiffusion  # noqa: E402
from generadores.daniel.src.network import TemporalDenoiser  # noqa: E402
from generadores.daniel.src.run_artifacts import read_manifest, validate_frozen_baseline  # noqa: E402
from generadores.daniel.src.sampler import DDPMSampler  # noqa: E402
from generadores.daniel.src.temporary_normalizer import (  # noqa: E402
    TemporaryTickerChannelNormalizer,
)
from generadores.daniel.src.validation import CHANNEL_ORDER  # noqa: E402

TRAINING_RUN_ID = "diffusion_seed42_baseline"
DIAGNOSTIC_RUN_ID = "diffusion_seed42_normalized_diagnostic"
TRAINING_COMMIT = "d611f0efe2238197c08d24dc97cdfb60e02812d3"
BASE_MASTER_COMMIT = "eb66f836d4eaafa6e32cc204f7c720d1f0400e18"
BEST_CHECKPOINT_SHA256 = "33631ad41807c58bf555bd1f7f1d0bb13d590e3b6f8f3780aa29109a9f76d99e"
DONOR_TRAIN_SHA256 = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"
DONOR_VALIDATION_SHA256 = "134f51a2ac9e546bf1a2f21f4efbf56a62bf019a08de14209058563b0a88ae23"
N_SAMPLES = 1000
SAMPLING_SEED = 42


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _acf_summary(table: pd.DataFrame) -> dict:
    maximum_index = table["absolute_difference"].idxmax()
    return {
        "mean_absolute_difference": float(table["absolute_difference"].mean()),
        "max_absolute_difference": float(table.loc[maximum_index, "absolute_difference"]),
        "max_difference_lag": int(table.loc[maximum_index, "lag"]),
    }


def main() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("Diagnostic sampling requires a clean versioned worktree")
    diagnostic_commit = _git("rev-parse", "HEAD")
    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    config_path = REPOSITORY_ROOT / "generadores/daniel/config/diffusion.yaml"
    checkpoint_path = artifact_root / f"checkpoints/{TRAINING_RUN_ID}/best_model.pt"
    training_manifest_path = artifact_root / f"manifests/{TRAINING_RUN_ID}.json"
    normalizer_path = artifact_root / "manifests/diffusion_seed42_normalizer.json"
    sample_path = artifact_root / "samples/diffusion_seed42_normalized_diagnostic_1000.npz"
    manifest_path = artifact_root / f"manifests/{DIAGNOSTIC_RUN_ID}.json"
    table_directory = artifact_root / "manifests"
    figure_directory = artifact_root / "figures"

    checkpoint_hash_before = sha256_file(checkpoint_path)
    if checkpoint_hash_before != BEST_CHECKPOINT_SHA256:
        raise RuntimeError("Best checkpoint hash does not match the certified run")
    training_manifest = read_manifest(training_manifest_path)
    if training_manifest["git_commit"] != TRAINING_COMMIT:
        raise RuntimeError("Training manifest identifies an unexpected code commit")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_frozen_baseline(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model_config = config["model"]
    model = TemporalDenoiser(
        input_length=model_config["input_length"],
        input_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
        time_embedding_dim=model_config["time_embedding_dim"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    diffusion_config = config["diffusion"]
    diffusion = GaussianDiffusion(
        model,
        steps=diffusion_config["steps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    ).to(device)
    sampler = DDPMSampler(diffusion)

    sampling_start = perf_counter()
    samples = sampler.sample(N_SAMPLES, seed=SAMPLING_SEED)
    sampling_runtime = perf_counter() - sampling_start
    repeated = sampler.sample(N_SAMPLES, seed=SAMPLING_SEED)
    different_seed = sampler.sample(N_SAMPLES, seed=SAMPLING_SEED + 1)
    same_seed_equal = bool(torch.equal(samples, repeated))
    different_seed_different = bool(not torch.equal(samples, different_seed))
    if tuple(samples.shape) != (N_SAMPLES, 65, 3):
        raise RuntimeError(f"Unexpected sample shape: {tuple(samples.shape)}")
    if samples.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(f"Unexpected sample dtype: {samples.dtype}")
    if not torch.isfinite(samples).all().item():
        raise RuntimeError("Sample pool contains NaN or infinite values")
    if not same_seed_equal or not different_seed_different:
        raise RuntimeError("Sampling reproducibility contract failed")
    sample_sha256 = save_sample_pool(sample_path, samples, seed=SAMPLING_SEED)
    stored_pool = load_sample_pool(sample_path)
    if stored_pool["seed"] != SAMPLING_SEED or not np.array_equal(
        stored_pool["samples"], samples.numpy()
    ):
        raise RuntimeError("Stored sample pool failed round-trip verification")

    train, validation = load_canonical_donor_tensors(REPOSITORY_ROOT)
    if train.input_sha256 != DONOR_TRAIN_SHA256:
        raise RuntimeError("donor_train hash mismatch")
    if validation.input_sha256 != DONOR_VALIDATION_SHA256:
        raise RuntimeError("donor_validation hash mismatch")
    normalizer = TemporaryTickerChannelNormalizer.load_json(normalizer_path)
    train_normalized = normalizer.transform(train.tensor, train.tickers)
    validation_normalized = normalizer.transform(validation.tensor, validation.tickers)
    synthetic = samples.numpy()

    stats = channel_statistics(validation_normalized, synthetic)
    wasserstein = wasserstein_by_channel(validation_normalized, synthetic)
    acf_return = acf_comparison(validation_normalized, synthetic, max_lag=20)
    acf_abs_return = acf_comparison(
        validation_normalized, synthetic, max_lag=20, absolute=True
    )
    correlations = cross_channel_correlations(validation_normalized, synthetic)
    correlation_rows = correlation_table(correlations)
    memorization = memorization_diagnostics(
        synthetic, train_normalized, validation_normalized
    )
    diversity = diversity_diagnostics(synthetic, seed=SAMPLING_SEED)

    table_paths = {
        "channel_statistics": table_directory / "diffusion_seed42_normalized_channel_stats.csv",
        "wasserstein": table_directory / "diffusion_seed42_normalized_wasserstein.csv",
        "return_acf": table_directory / "diffusion_seed42_normalized_return_acf.csv",
        "abs_return_acf": table_directory / "diffusion_seed42_normalized_abs_return_acf.csv",
        "correlations": table_directory / "diffusion_seed42_normalized_correlations.csv",
        "nearest_neighbor": table_directory / "diffusion_seed42_normalized_nearest_neighbor.csv",
    }
    table_directory.mkdir(parents=True, exist_ok=True)
    stats.to_csv(table_paths["channel_statistics"], index=False)
    wasserstein.to_csv(table_paths["wasserstein"], index=False)
    acf_return.to_csv(table_paths["return_acf"], index=False)
    acf_abs_return.to_csv(table_paths["abs_return_acf"], index=False)
    correlation_rows.to_csv(table_paths["correlations"], index=False)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "source": "validation_to_train",
                    "query_index": np.arange(len(memorization["validation_to_train"])),
                    "nearest_distance": memorization["validation_to_train"],
                }
            ),
            pd.DataFrame(
                {
                    "source": "synthetic_to_train",
                    "query_index": np.arange(len(memorization["synthetic_to_train"])),
                    "nearest_distance": memorization["synthetic_to_train"],
                }
            ),
        ],
        ignore_index=True,
    ).to_csv(table_paths["nearest_neighbor"], index=False)

    absolute_figure_paths = generate_diagnostic_figures(
        validation_normalized,
        synthetic,
        acf_return=acf_return,
        acf_abs_return=acf_abs_return,
        correlations=correlations,
        validation_nn=memorization["validation_to_train"],
        synthetic_nn=memorization["synthetic_to_train"],
        output_directory=figure_directory,
    )
    figure_paths = {
        name: _relative(Path(path)) for name, path in absolute_figure_paths.items()
    }

    checkpoint_hash_after = sha256_file(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("Checkpoint changed during diagnostics")
    manifest = {
        "run_id": DIAGNOSTIC_RUN_ID,
        "model": "diffusion",
        "training_seed": 42,
        "sampling_seed": SAMPLING_SEED,
        "comparison_seed": SAMPLING_SEED + 1,
        "n_samples": N_SAMPLES,
        "checkpoint_path": _relative(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_after,
        "training_commit": TRAINING_COMMIT,
        "diagnostic_commit": diagnostic_commit,
        "base_master_commit": BASE_MASTER_COMMIT,
        "donor_train_sha256": train.input_sha256,
        "donor_validation_sha256": validation.input_sha256,
        "normalizer_path": _relative(normalizer_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "window_shape": [65, 3],
        "channels": list(CHANNEL_ORDER),
        "space": "normalized",
        "sample_dtype": str(samples.numpy().dtype),
        "sampling_runtime_seconds": sampling_runtime,
        "device": device,
        "sample_file": _relative(sample_path),
        "sample_sha256": sample_sha256,
        "finite": True,
        "same_seed_exactly_equal": same_seed_equal,
        "different_seed_different": different_seed_different,
        "n_unique": diversity["n_unique"],
        "duplicate_count": diversity["duplicate_count"],
        "channel_statistics": stats.to_dict(orient="records"),
        "wasserstein_1": {
            str(row.channel): float(row.wasserstein_1)
            for row in wasserstein.itertuples(index=False)
        },
        "acf_return_summary": _acf_summary(acf_return),
        "acf_abs_return_summary": _acf_summary(acf_abs_return),
        "cross_channel": {
            "real_matrix": correlations["real"].tolist(),
            "synthetic_matrix": correlations["synthetic"].tolist(),
            "absolute_difference_matrix": correlations["absolute_difference"].tolist(),
            "mean_absolute_off_diagonal_difference": correlations[
                "mean_absolute_off_diagonal_difference"
            ],
            "max_absolute_off_diagonal_difference": correlations[
                "max_absolute_off_diagonal_difference"
            ],
        },
        "memorization": {
            "validation_to_train": memorization["validation_to_train_summary"],
            "synthetic_to_train": memorization["synthetic_to_train_summary"],
            "exact_duplicate_count": memorization["exact_duplicate_count"],
            "near_exact_count_including_exact": memorization[
                "near_exact_count_including_exact"
            ],
            "near_exact_tolerance": memorization["near_exact_tolerance"],
        },
        "diversity": diversity,
        "methods": {
            "acf": "biased ACF computed per window, then equally averaged; no cross-window lag pairs",
            "correlation": "pooled time rows within windows; identical procedure for validation and synthetic",
            "nearest_neighbor": "Euclidean distance on flattened 195-vector, computed in query chunks",
            "wasserstein": "exact empirical one-dimensional CDF integral per channel",
            "kurtosis": "population excess kurtosis (Fisher definition)",
        },
        "tables": {name: _relative(path) for name, path in table_paths.items()},
        "figures": figure_paths,
    }
    write_diagnostic_manifest(manifest, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
