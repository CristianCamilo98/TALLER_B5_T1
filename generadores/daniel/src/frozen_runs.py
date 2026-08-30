"""Validation and summary helpers for the three frozen Diffusion runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from .run_artifacts import (
    FROZEN_RUN_IDS,
    FROZEN_TRAINING_SEEDS,
    read_history,
    read_manifest,
    validate_frozen_effective_config,
)


SUMMARY_COLUMNS = (
    "seed",
    "run_id",
    "epochs_completed",
    "best_epoch",
    "best_validation_loss",
    "final_train_loss",
    "final_validation_loss",
    "stopping_reason",
    "runtime_seconds",
    "best_checkpoint_sha256",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256_hex(value: str, *, field: str = "sha256") -> str:
    """Require an exact lowercase, 64-character SHA256 hexadecimal digest."""

    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must contain exactly 64 lowercase hexadecimal characters")
    return value


def verify_best_checkpoint_from_manifest(
    repository_root: Path | str, manifest: dict[str, Any]
) -> tuple[Path, str]:
    """Resolve and hash a best checkpoint using its frozen manifest as authority."""

    expected = validate_sha256_hex(
        manifest.get("best_checkpoint_sha256"), field="best_checkpoint_sha256"
    )
    relative_path = manifest.get("best_checkpoint_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Frozen manifest lacks best_checkpoint_path")
    root = Path(repository_root).resolve()
    checkpoint = (root / relative_path).resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError as error:
        raise ValueError("Checkpoint path escapes the repository root") from error
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen best checkpoint does not exist: {checkpoint}")
    calculated = sha256_file(checkpoint)
    if calculated != expected:
        raise RuntimeError(
            "Frozen best checkpoint bytes do not match best_checkpoint_sha256 in manifest"
        )
    return checkpoint, calculated


def validate_frozen_manifests(manifests: list[dict[str, Any]]) -> None:
    """Require the three manifests to differ experimentally only by seed."""

    if len(manifests) != len(FROZEN_TRAINING_SEEDS):
        raise ValueError("Exactly three frozen run manifests are required")
    by_seed = {int(manifest["training_seed"]): manifest for manifest in manifests}
    if set(by_seed) != set(FROZEN_TRAINING_SEEDS):
        raise ValueError("Frozen manifests must cover seeds 42, 123, and 2026")

    reference = by_seed[FROZEN_TRAINING_SEEDS[0]]
    invariant_fields = (
        "git_commit",
        "base_master_commit",
        "canonical_raw_sha256",
        "donor_train_sha256",
        "donor_validation_sha256",
        "train_count",
        "validation_count",
        "window_shape",
        "channels",
        "normalizer_sha256",
        "validation_seed",
    )
    for seed, manifest in by_seed.items():
        if manifest["run_id"] != FROZEN_RUN_IDS[seed]:
            raise ValueError(f"Unexpected run_id for seed {seed}")
        validate_frozen_effective_config(manifest["effective_config"], seed)
        for field in invariant_fields:
            if manifest[field] != reference[field]:
                raise ValueError(f"Frozen runs differ in invariant field {field!r}")


def build_frozen_summary(manifests: list[dict[str, Any]]) -> pd.DataFrame:
    validate_frozen_manifests(manifests)
    rows = []
    for manifest in sorted(manifests, key=lambda item: int(item["training_seed"])):
        rows.append(
            {
                "seed": int(manifest["training_seed"]),
                "run_id": manifest["run_id"],
                "epochs_completed": int(manifest["epochs_completed"]),
                "best_epoch": int(manifest["best_epoch"]),
                "best_validation_loss": float(manifest["best_validation_loss"]),
                "final_train_loss": float(manifest["final_train_loss"]),
                "final_validation_loss": float(manifest["final_validation_loss"]),
                "stopping_reason": manifest["stopping_reason"],
                "runtime_seconds": float(manifest["runtime_seconds"]),
                "best_checkpoint_sha256": manifest["best_checkpoint_sha256"],
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def write_frozen_summary(frame: pd.DataFrame, path: Path | str) -> None:
    if tuple(frame.columns) != SUMMARY_COLUMNS or len(frame) != 3:
        raise ValueError("Frozen summary has an incompatible schema")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(frame.to_csv(index=False), encoding="utf-8")


def model_states_equal(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> tuple[bool, float]:
    if tuple(first) != tuple(second):
        return False, float("inf")
    maximum = 0.0
    for key in first:
        left = first[key].detach().cpu()
        right = second[key].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, float("inf")
        if left.numel():
            maximum = max(maximum, float(torch.max(torch.abs(left - right))))
        if not torch.equal(left, right):
            return False, maximum
    return True, maximum


def histories_numerically_equal(
    first_path: Path | str, second_path: Path | str
) -> tuple[bool, float]:
    """Compare learning metrics while excluding nondeterministic wall-clock time."""

    first = read_history(first_path)
    second = read_history(second_path)
    columns = ["epoch", "train_loss", "validation_loss", "learning_rate"]
    if first.shape != second.shape:
        return False, float("inf")
    difference = np.abs(
        first.loc[:, columns].to_numpy(dtype=np.float64)
        - second.loc[:, columns].to_numpy(dtype=np.float64)
    )
    maximum = float(difference.max(initial=0.0))
    return bool(np.array_equal(difference, np.zeros_like(difference))), maximum


def load_frozen_manifests(artifact_root: Path | str) -> list[dict[str, Any]]:
    root = Path(artifact_root)
    return [
        read_manifest(root / "manifests" / f"{FROZEN_RUN_IDS[seed]}.json")
        for seed in FROZEN_TRAINING_SEEDS
    ]
