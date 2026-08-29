"""Generate one frozen 5,000-window normalized/NVDA-like paired pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.diffusion import GaussianDiffusion  # noqa: E402
from generadores.daniel.src.final_pools import (  # noqa: E402
    channel_sanity_summary,
    pairing_max_abs_error,
    save_final_pool,
    write_final_pool_manifest,
    write_json_artifact,
)
from generadores.daniel.src.frozen_runs import (  # noqa: E402
    sha256_file,
    verify_best_checkpoint_from_manifest,
)
from generadores.daniel.src.network import TemporalDenoiser  # noqa: E402
from generadores.daniel.src.run_artifacts import (  # noqa: E402
    FROZEN_RUN_IDS,
    FROZEN_TRAINING_SEEDS,
    read_manifest,
    validate_frozen_effective_config,
)
from generadores.daniel.src.sampler import DDPMSampler  # noqa: E402
from generadores.daniel.src.temporary_nvda_calibration import (  # noqa: E402
    NVDA_VISIBLE_END,
    NVDA_VISIBLE_SOURCE,
    NVDA_VISIBLE_SOURCE_SHA256,
    NVDA_VISIBLE_START,
    TemporaryNVDACalibrator,
    calculate_calibration_stats,
    generate_accepted_pool,
    load_nvda_visible_daily,
    physical_window_mask,
)
from generadores.daniel.src.validation import CHANNEL_ORDER  # noqa: E402


FROZEN_TRAINING_SHA = "b4db7a9c894598012c1574a64cccccabef14b89d"
BASE_MASTER_COMMIT = "eb66f836d4eaafa6e32cc204f7c720d1f0400e18"
DONOR_TRAIN_SHA256 = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"
DONOR_VALIDATION_SHA256 = "134f51a2ac9e546bf1a2f21f4efbf56a62bf019a08de14209058563b0a88ae23"
NORMALIZER_SHA256 = "fc81f1d25d68680e5f29af97d1d6b37d877e2524e41103154e51695f64a59b8f"
N_REQUESTED = 5000
BATCH_SIZE = 256


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _load_sampler(seed: int, device: str) -> tuple[DDPMSampler, Path, dict, str]:
    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    run_id = FROZEN_RUN_IDS[seed]
    training_manifest = read_manifest(artifact_root / "manifests" / f"{run_id}.json")
    if training_manifest["git_commit"] != FROZEN_TRAINING_SHA:
        raise RuntimeError("Training manifest does not identify frozen training code")
    if training_manifest["normalizer_sha256"] != NORMALIZER_SHA256:
        raise RuntimeError("Training manifest normalizer hash mismatch")
    normalizer_path = REPOSITORY_ROOT / training_manifest["normalizer_path"]
    if sha256_file(normalizer_path) != NORMALIZER_SHA256:
        raise RuntimeError("Frozen normalizer artifact hash mismatch")
    if training_manifest["train_count"] != 4910:
        raise RuntimeError("Frozen training manifest has an unexpected train count")
    if training_manifest["validation_count"] != 380:
        raise RuntimeError("Frozen training manifest has an unexpected validation count")
    validate_frozen_effective_config(training_manifest["effective_config"], seed)
    checkpoint_path, checkpoint_sha256 = verify_best_checkpoint_from_manifest(
        REPOSITORY_ROOT, training_manifest
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = training_manifest["effective_config"]
    model_config = config["model"]
    model = TemporalDenoiser(
        input_length=model_config["input_length"],
        input_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
        time_embedding_dim=model_config["time_embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    diffusion_config = config["diffusion"]
    diffusion = GaussianDiffusion(
        model,
        steps=diffusion_config["steps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    ).to(device)
    return DDPMSampler(diffusion), checkpoint_path, training_manifest, checkpoint_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=FROZEN_TRAINING_SEEDS, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    seed = int(args.seed)
    if _git("status", "--porcelain"):
        raise RuntimeError("Final generation requires a clean versioned worktree")
    generation_sha = _git("rev-parse", "HEAD")
    if not _git("merge-base", "--is-ancestor", FROZEN_TRAINING_SHA, generation_sha) == "":
        raise RuntimeError("Frozen training code is not an ancestor of generation code")
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_path = REPOSITORY_ROOT / NVDA_VISIBLE_SOURCE
    if sha256_file(source_path) != NVDA_VISIBLE_SOURCE_SHA256:
        raise RuntimeError("Canonical daily feature source hash mismatch")
    donor_train_path = REPOSITORY_ROOT / "data/features/windows/donor_train.parquet"
    donor_validation_path = (
        REPOSITORY_ROOT / "data/features/windows/donor_validation.parquet"
    )
    if sha256_file(donor_train_path) != DONOR_TRAIN_SHA256:
        raise RuntimeError("Canonical donor_train hash mismatch")
    if sha256_file(donor_validation_path) != DONOR_VALIDATION_SHA256:
        raise RuntimeError("Canonical donor_validation hash mismatch")
    visible = load_nvda_visible_daily(REPOSITORY_ROOT)
    stats = calculate_calibration_stats(visible)
    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    calibration_path = artifact_root / "manifests/nvda_visible_calibration_stats.json"
    calibration_payload = {
        "source_path": NVDA_VISIBLE_SOURCE.as_posix(),
        "source_sha256": NVDA_VISIBLE_SOURCE_SHA256,
        "ticker": "NVDA",
        "visible_protocol_start": NVDA_VISIBLE_START.strftime("%Y-%m-%d"),
        "visible_protocol_end": NVDA_VISIBLE_END.strftime("%Y-%m-%d"),
        "start_date": stats.observation_start,
        "end_date": stats.observation_end,
        **stats.as_dict(),
        "created_by": "generadores/daniel/scripts/generate_final_pools.py",
        "git_commit": generation_sha,
    }
    write_json_artifact(calibration_payload, calibration_path)
    calibration_sha256 = sha256_file(calibration_path)
    calibrator = TemporaryNVDACalibrator(stats)
    sampler, checkpoint_path, training_manifest, checkpoint_sha256 = _load_sampler(
        seed, device
    )

    start = perf_counter()
    result = generate_accepted_pool(
        sampler.sample,
        calibrator,
        n_requested=N_REQUESTED,
        base_seed=seed,
        batch_size=args.batch_size,
    )
    runtime = perf_counter() - start
    repeated = generate_accepted_pool(
        sampler.sample,
        calibrator,
        n_requested=N_REQUESTED,
        base_seed=seed,
        batch_size=args.batch_size,
    )
    reproducible = bool(
        torch.equal(result["normalized"], repeated["normalized"])
        and torch.equal(result["calibrated"], repeated["calibrated"])
        and result["subseeds"] == repeated["subseeds"]
        and result["rejection_reasons"] == repeated["rejection_reasons"]
    )
    if not reproducible:
        raise RuntimeError("Final accepted pool is not exactly reproducible")

    normalized = result["normalized"]
    calibrated = result["calibrated"]
    valid, remaining_reasons = physical_window_mask(calibrated)
    if not valid.all().item() or any(remaining_reasons.values()):
        raise RuntimeError("Accepted NVDA-like pool retains physically invalid windows")
    pairing_error = pairing_max_abs_error(normalized, calibrated, stats.mean, stats.std)
    volume_proxy = torch.expm1(calibrated[:, :, 2])
    physical_validity = {
        "finite": bool(torch.isfinite(calibrated).all().item()),
        "minimum_range": float(calibrated[:, :, 1].min().item()),
        "minimum_log1p_volume": float(calibrated[:, :, 2].min().item()),
        "minimum_volume_proxy": float(volume_proxy.min().item()),
        "invalid_remaining": int((~valid).sum().item()),
    }

    sample_directory = artifact_root / "samples"
    normalized_path = sample_directory / f"diffusion_seed{seed}_normalized_5000.npz"
    calibrated_path = sample_directory / f"diffusion_seed{seed}_nvda_like_5000.npz"
    normalized_hash = save_final_pool(
        normalized_path,
        normalized,
        training_seed=seed,
        sampling_seed=seed,
        space="normalized",
        checkpoint_sha256=checkpoint_sha256,
        calibration_stats_sha256=calibration_sha256,
        generation_commit=generation_sha,
    )
    calibrated_hash = save_final_pool(
        calibrated_path,
        calibrated,
        training_seed=seed,
        sampling_seed=seed,
        space="nvda_like",
        checkpoint_sha256=checkpoint_sha256,
        calibration_stats_sha256=calibration_sha256,
        generation_commit=generation_sha,
    )
    manifest = {
        "run_id": f"diffusion_seed{seed}_final_pool",
        "model": "diffusion",
        "training_seed": seed,
        "sampling_seed": seed,
        "frozen_training_sha": FROZEN_TRAINING_SHA,
        "generation_code_sha": generation_sha,
        "base_master_commit": BASE_MASTER_COMMIT,
        "checkpoint_path": _relative(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_manifest_sha256": training_manifest["best_checkpoint_sha256"],
        "donor_train_sha256": DONOR_TRAIN_SHA256,
        "donor_validation_sha256": DONOR_VALIDATION_SHA256,
        "normalizer_sha256": training_manifest["normalizer_sha256"],
        "nvda_calibration_source_path": NVDA_VISIBLE_SOURCE.as_posix(),
        "nvda_calibration_source_sha256": NVDA_VISIBLE_SOURCE_SHA256,
        "nvda_visible_start": NVDA_VISIBLE_START.strftime("%Y-%m-%d"),
        "nvda_visible_end": NVDA_VISIBLE_END.strftime("%Y-%m-%d"),
        "nvda_feature_observation_start": stats.observation_start,
        "nvda_feature_observation_end": stats.observation_end,
        "nvda_daily_observation_count": stats.n_daily_observations,
        "nvda_calibration_mean": calibration_payload["mean"],
        "nvda_calibration_std": calibration_payload["std"],
        "nvda_calibration_ddof": stats.ddof,
        "calibration_stats_path": _relative(calibration_path),
        "calibration_stats_sha256": calibration_sha256,
        "window_shape": [65, 3],
        "channels": list(CHANNEL_ORDER),
        "n_requested": N_REQUESTED,
        "n_candidates_generated": result["n_candidates"],
        "n_accepted": result["n_accepted"],
        "n_rejected": result["n_rejected"],
        "rejection_rate": result["n_rejected"] / result["n_candidates"],
        "rejection_reason_counts": result["rejection_reasons"],
        "sampling_subseeds": result["subseeds"],
        "sampling_subseed_mechanism": result["subseed_mechanism"],
        "sampling_batch_size": args.batch_size,
        "normalized_pool_path": _relative(normalized_path),
        "normalized_pool_sha256": normalized_hash,
        "nvda_like_pool_path": _relative(calibrated_path),
        "nvda_like_pool_sha256": calibrated_hash,
        "generation_runtime_seconds": runtime,
        "device": device,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "reproducibility_pass": reproducible,
        "physical_validity": physical_validity,
        "pairing_max_abs_error": pairing_error,
        "post_calibration_summary": channel_sanity_summary(calibrated),
        "NVDA_visible_used": True,
        "NVDA_hidden_used": False,
        "NVDA_full_history_used": False,
        "NVDA_test_used": False,
        "downstream_used": False,
    }
    manifest_path = artifact_root / "manifests" / f"diffusion_seed{seed}_final_pool.json"
    write_final_pool_manifest(manifest, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
