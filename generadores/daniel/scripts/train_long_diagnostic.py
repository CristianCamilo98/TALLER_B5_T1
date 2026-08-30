"""Run the isolated seed-42 long-training diagnostic without replacing frozen runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.data_adapter import load_canonical_donor_tensors  # noqa: E402
from generadores.daniel.src.diffusion import GaussianDiffusion  # noqa: E402
from generadores.daniel.src.frozen_runs import sha256_file  # noqa: E402
from generadores.daniel.src.network import TemporalDenoiser  # noqa: E402
from generadores.daniel.src.reproducibility import set_seed  # noqa: E402
from generadores.daniel.src.run_artifacts import (  # noqa: E402
    LONG_TRAINING_RUN_ID,
    long_training_diagnostic_config,
    validate_long_training_config,
    write_history,
    write_manifest,
)
from generadores.daniel.src.temporary_normalizer import (  # noqa: E402
    GlobalChannelNormalizer,
)
from generadores.daniel.src.trainer import DiffusionTrainer, TrainerConfig  # noqa: E402
from generadores.daniel.src.training_diagnostics import (  # noqa: E402
    best_history_point,
    plot_frozen_vs_long,
    plot_training_views,
)
from generadores.daniel.src.validation import CHANNEL_ORDER  # noqa: E402


CANONICAL_RAW_SHA256 = "6ecd4c929ecd3bdca32c646aec8210a7757b566843a90102f21bd86d2da036d6"
DONOR_TRAIN_SHA256 = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"
DONOR_VALIDATION_SHA256 = "134f51a2ac9e546bf1a2f21f4efbf56a62bf019a08de14209058563b0a88ae23"


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _assert_clean_versioned_code() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("Long training requires a clean versioned worktree")


def _date_range(metadata) -> dict:
    return {
        "window_start_min": metadata["window_start_date"].min().strftime("%Y-%m-%d"),
        "window_end_max": metadata["window_end_date"].max().strftime("%Y-%m-%d"),
    }


def _new_diffusion(config: dict) -> GaussianDiffusion:
    model_config = config["model"]
    model = TemporalDenoiser(
        input_length=model_config["input_length"],
        input_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
        time_embedding_dim=model_config["time_embedding_dim"],
    )
    diffusion_config = config["diffusion"]
    return GaussianDiffusion(
        model,
        steps=diffusion_config["steps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )


def _verify_checkpoint(path: Path, config: dict, expected_epoch: int | None = None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    fresh = _new_diffusion(config)
    fresh.model.load_state_dict(payload["model_state_dict"], strict=True)
    if expected_epoch is not None and int(payload["epoch"]) != int(expected_epoch):
        raise RuntimeError("Best checkpoint epoch does not match history argmin")
    if not all(torch.isfinite(parameter).all() for parameter in fresh.model.parameters()):
        raise RuntimeError("Loaded checkpoint contains non-finite parameters")
    return payload


def main() -> None:
    _assert_clean_versioned_code()
    config_path = REPOSITORY_ROOT / "generadores/daniel/config/diffusion.yaml"
    frozen_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = long_training_diagnostic_config(frozen_config)
    validate_long_training_config(config)

    seed = int(config["reproducibility"]["seed"])
    validation_seed = int(config["reproducibility"]["validation_seed"])
    if seed != 42 or validation_seed != 424242:
        raise RuntimeError("Long diagnostic is restricted to seed 42 and validation seed 424242")
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    environment = set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = datetime.now(timezone.utc)

    train, validation = load_canonical_donor_tensors(
        REPOSITORY_ROOT, dtype=torch.float64
    )
    if train.input_sha256 != DONOR_TRAIN_SHA256:
        raise RuntimeError("donor_train hash differs from the certified input")
    if validation.input_sha256 != DONOR_VALIDATION_SHA256:
        raise RuntimeError("donor_validation hash differs from the certified input")

    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    frozen_manifest_path = (
        artifact_root / "manifests/diffusion_seed42_global_channel.json"
    )
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if frozen_manifest["canonical_raw_sha256"] != CANONICAL_RAW_SHA256:
        raise RuntimeError("Frozen manifest does not identify the canonical raw snapshot")
    normalizer_path = REPOSITORY_ROOT / frozen_manifest["normalizer_path"]
    normalizer_sha256 = frozen_manifest["normalizer_sha256"]
    if sha256_file(normalizer_path) != normalizer_sha256:
        raise RuntimeError("Frozen train-only normalizer hash mismatch")
    normalizer = GlobalChannelNormalizer.load_json(normalizer_path)
    normalized_train = normalizer.transform(train.tensor)
    normalized_validation = normalizer.transform(validation.tensor)

    run_id = LONG_TRAINING_RUN_ID
    checkpoint_directory = artifact_root / "checkpoints" / run_id
    history_path = artifact_root / "histories" / f"{run_id}.csv"
    manifest_path = artifact_root / "manifests" / f"{run_id}.json"
    full_plot = artifact_root / "figures/diffusion_seed42_global_channel_long_training_full.png"
    zoom_plot = artifact_root / "figures/diffusion_seed42_global_channel_long_training_zoom.png"
    comparison_plot = artifact_root / "figures/diffusion_seed42_global_channel_frozen_vs_long_training.png"

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

    diffusion = _new_diffusion(config)
    training = config["training"]
    trainer_config = TrainerConfig(
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        gradient_clip_norm=training["gradient_clip_norm"],
        max_epochs=training["max_epochs"],
        early_stopping_patience=training["early_stopping_patience"],
        seed=seed,
        validation_seed=validation_seed,
        device=device,
    )
    git_commit = _git("rev-parse", "HEAD")
    base_master_commit = _git("merge-base", "HEAD", "master")
    trainer = DiffusionTrainer(
        diffusion,
        trainer_config,
        checkpoint_directory=checkpoint_directory,
        checkpoint_metadata={
            "run_id": run_id,
            "effective_config": config,
            "seed": seed,
            "validation_seed": validation_seed,
            "channel_order": list(CHANNEL_ORDER),
            "window_shape": [65, 3],
            "git_commit": git_commit,
            "experiment_type": "long_training_diagnostic_not_frozen_replacement",
        },
    )

    def on_epoch(row: dict, history: list[dict]) -> None:
        write_history(history, history_path)
        print(json.dumps(row), flush=True)

    result = trainer.fit(train_loader, validation_loader, epoch_callback=on_epoch)
    write_history(result.history, history_path)
    finished = datetime.now(timezone.utc)

    best_epoch, best_loss = best_history_point(pd.DataFrame(result.history))
    if best_epoch != result.best_epoch or best_loss != result.best_validation_loss:
        raise RuntimeError("Trainer best checkpoint is not the validation-loss argmin")
    best_path = checkpoint_directory / "best_model.pt"
    last_path = checkpoint_directory / "last_model.pt"
    best_payload = _verify_checkpoint(best_path, config, expected_epoch=best_epoch)
    _verify_checkpoint(last_path, config, expected_epoch=len(result.history))
    if float(best_payload["validation_loss"]) != best_loss:
        raise RuntimeError("Best checkpoint loss does not match history argmin")

    training_plots = plot_training_views(
        history_path, full_plot, zoom_plot, run_id=run_id
    )
    comparison = plot_frozen_vs_long(
        artifact_root / "histories/diffusion_seed42_global_channel.csv",
        history_path,
        comparison_plot,
    )

    manifest = {
        "run_id": run_id,
        "model": "diffusion",
        "model_config": config["model"],
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
        "normalizer_path": normalizer_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "normalizer_sha256": normalizer_sha256,
        "effective_config": config,
        "frozen_reference_config": frozen_config,
        "approved_config_differences": {
            "training.max_epochs": [200, 300],
            "training.early_stopping_patience": [20, 30],
        },
        "device": device,
        "python_version": environment["python_version"],
        "torch_version": environment["torch_version"],
        "cuda_version": environment["cuda_version"],
        "start_time": started.isoformat(),
        "end_time": finished.isoformat(),
        "epochs_completed": len(result.history),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "final_train_loss": result.history[-1]["train_loss"],
        "final_validation_loss": result.history[-1]["validation_loss"],
        "stopping_reason": result.stopping_reason,
        "runtime_seconds": result.total_seconds,
        "checkpoint_best": best_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "checkpoint_last": last_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history_path": history_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "training_plots": {
            "full": full_plot.relative_to(REPOSITORY_ROOT).as_posix(),
            "zoom": zoom_plot.relative_to(REPOSITORY_ROOT).as_posix(),
            "zoom_bounds": [training_plots["zoom_start"], training_plots["zoom_end"]],
            "frozen_vs_long": comparison_plot.relative_to(REPOSITORY_ROOT).as_posix(),
        },
        "frozen_comparison": {
            "frozen_best_epoch": frozen_manifest["best_epoch"],
            "frozen_best_validation_loss": frozen_manifest["best_validation_loss"],
            **comparison,
            "improved": comparison["delta_best_validation_loss"] < 0,
            "automatic_replacement_permitted": False,
        },
        "validation_protocol": frozen_manifest["validation_protocol"],
    }
    write_manifest(manifest, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
