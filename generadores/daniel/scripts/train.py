"""Train one approved seed of the frozen DDPM baseline on donor data only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.data_adapter import load_canonical_donor_tensors  # noqa: E402
from generadores.daniel.src.diffusion import GaussianDiffusion  # noqa: E402
from generadores.daniel.src.network import TemporalDenoiser  # noqa: E402
from generadores.daniel.src.reproducibility import set_seed  # noqa: E402
from generadores.daniel.src.run_artifacts import (  # noqa: E402
    FROZEN_TRAINING_SEEDS,
    frozen_config_for_seed,
    global_channel_run_id,
    validate_frozen_baseline,
    write_history,
    write_manifest,
)
from generadores.daniel.src.temporary_normalizer import (  # noqa: E402
    GlobalChannelNormalizer,
)
from generadores.daniel.src.trainer import DiffusionTrainer, TrainerConfig  # noqa: E402
from generadores.daniel.src.validation import CHANNEL_ORDER  # noqa: E402

CANONICAL_RAW_SHA256 = "6ecd4c929ecd3bdca32c646aec8210a7757b566843a90102f21bd86d2da036d6"
DONOR_TRAIN_SHA256 = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"
DONOR_VALIDATION_SHA256 = "134f51a2ac9e546bf1a2f21f4efbf56a62bf019a08de14209058563b0a88ae23"


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_frozen_baseline(config)
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_clean_versioned_code() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError(
            "Training requires a clean working tree so git_commit identifies exact code"
        )


def _date_range(metadata) -> dict:
    return {
        "window_start_min": metadata["window_start_date"].min().strftime("%Y-%m-%d"),
        "window_end_max": metadata["window_end_date"].max().strftime("%Y-%m-%d"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "generadores/daniel/config/diffusion.yaml",
    )
    parser.add_argument("--seed", type=int, choices=FROZEN_TRAINING_SEEDS, default=42)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    expected_run_id = global_channel_run_id(args.seed)
    run_id = args.run_id or expected_run_id
    if run_id != expected_run_id:
        raise ValueError(f"Seed {args.seed} requires run_id {expected_run_id!r}")

    _assert_clean_versioned_code()
    config = frozen_config_for_seed(_load_config(args.config.resolve()), args.seed)
    start_time = datetime.now(timezone.utc)
    seed = int(config["reproducibility"]["seed"])
    validation_seed = int(config["reproducibility"]["validation_seed"])
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    environment = set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Scaler fit must see the original parquet precision. The normalized DDPM
    # tensors are converted to float32 by GlobalChannelNormalizer.transform.
    train, validation = load_canonical_donor_tensors(
        REPOSITORY_ROOT, dtype=torch.float64
    )
    if train.input_sha256 != DONOR_TRAIN_SHA256:
        raise RuntimeError("donor_train hash differs from the frozen baseline")
    if validation.input_sha256 != DONOR_VALIDATION_SHA256:
        raise RuntimeError("donor_validation hash differs from the frozen baseline")
    raw_manifest = json.loads(
        (REPOSITORY_ROOT / "data/raw/download_manifest.json").read_text(encoding="utf-8")
    )
    manifest_raw_hash = raw_manifest["checksums_sha256"]["data/raw/ohlcv_raw.parquet"]
    if manifest_raw_hash != CANONICAL_RAW_SHA256:
        raise RuntimeError("Raw manifest does not identify the canonical snapshot")

    normalizer = GlobalChannelNormalizer(
        std_threshold=float(config["normalization"]["std_threshold"])
    ).fit(train.tensor)
    normalized_train = normalizer.transform(train.tensor)
    normalized_validation = normalizer.transform(validation.tensor)
    reconstructed = normalizer.inverse_transform(normalized_train)
    roundtrip_error = float(torch.max(torch.abs(reconstructed - train.tensor)))
    scaler_state = normalizer.state_dict()
    minimum_std = min(scaler_state["std"])

    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    checkpoint_directory = artifact_root / "checkpoints" / run_id
    history_path = artifact_root / "histories" / f"{run_id}.csv"
    normalizer_path = artifact_root / "manifests" / f"{run_id}_normalizer.json"
    manifest_path = artifact_root / "manifests" / f"{run_id}.json"
    normalizer.save_json(normalizer_path)
    normalizer_sha256 = _sha256_file(normalizer_path)

    batch_size = int(config["training"]["batch_size"])
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(normalized_train),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        TensorDataset(normalized_validation),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model_config = config["model"]
    model = TemporalDenoiser(
        input_length=model_config["input_length"],
        input_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
        time_embedding_dim=model_config["time_embedding_dim"],
    )
    diffusion_config = config["diffusion"]
    diffusion = GaussianDiffusion(
        model,
        steps=diffusion_config["steps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )
    trainer_config = TrainerConfig(
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        gradient_clip_norm=config["training"]["gradient_clip_norm"],
        max_epochs=config["training"]["max_epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        seed=seed,
        validation_seed=validation_seed,
        device=device,
    )
    git_commit = _git("rev-parse", "HEAD")
    base_master_commit = _git("merge-base", "HEAD", "master")
    checkpoint_metadata = {
        "run_id": run_id,
        "effective_config": config,
        "seed": seed,
        "validation_seed": validation_seed,
        "channel_order": list(CHANNEL_ORDER),
        "window_shape": [65, 3],
        "git_commit": git_commit,
    }
    trainer = DiffusionTrainer(
        diffusion,
        trainer_config,
        checkpoint_directory=checkpoint_directory,
        checkpoint_metadata=checkpoint_metadata,
    )

    def on_epoch(row: dict, history: list[dict]) -> None:
        write_history(history, history_path)
        print(json.dumps(row), flush=True)

    result = trainer.fit(
        train_loader,
        validation_loader,
        epoch_callback=on_epoch,
    )
    write_history(result.history, history_path)
    end_time = datetime.now(timezone.utc)

    best_path = checkpoint_directory / "best_model.pt"
    last_path = checkpoint_directory / "last_model.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Required checkpoints were not created")
    manifest = {
        "run_id": run_id,
        "model": "diffusion",
        "model_config": model_config,
        "seed": seed,
        "training_seed": seed,
        "validation_seed": validation_seed,
        "git_commit": git_commit,
        "base_master_commit": base_master_commit,
        "canonical_raw_sha256": CANONICAL_RAW_SHA256,
        "donor_train_sha256": train.input_sha256,
        "donor_validation_sha256": validation.input_sha256,
        "train_count": int(train.tensor.shape[0]),
        "validation_count": int(validation.tensor.shape[0]),
        "window_shape": [65, 3],
        "channels": list(CHANNEL_ORDER),
        "train_dates": _date_range(train.metadata),
        "validation_dates": _date_range(validation.metadata),
        "normalization_type": "global_channel_zscore_train_only_float64_fit",
        "normalization_contract": scaler_state,
        "normalizer_path": normalizer_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "normalizer_sha256": normalizer_sha256,
        "normalizer_minimum_std": minimum_std,
        "normalizer_roundtrip_max_abs_error": roundtrip_error,
        "effective_config": config,
        "device": device,
        "python_version": environment["python_version"],
        "torch_version": environment["torch_version"],
        "cuda_version": environment["cuda_version"],
        "torch_num_threads": torch.get_num_threads(),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "epochs_completed": len(result.history),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "final_train_loss": result.history[-1]["train_loss"],
        "final_validation_loss": result.history[-1]["validation_loss"],
        "stopping_reason": result.stopping_reason,
        "total_seconds": result.total_seconds,
        "runtime_seconds": result.total_seconds,
        "mean_seconds_per_epoch": result.total_seconds / len(result.history),
        "checkpoint_best": best_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "checkpoint_last": last_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "best_checkpoint_path": best_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "last_checkpoint_path": last_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "best_checkpoint_sha256": _sha256_file(best_path),
        "last_checkpoint_sha256": _sha256_file(last_path),
        "history_path": history_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "validation_protocol": {
            "shuffle": False,
            "backward": False,
            "optimizer_step": False,
            "fixed_generator_reinitialized_each_epoch": True,
            "seed": validation_seed,
        },
    }
    write_manifest(manifest, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
