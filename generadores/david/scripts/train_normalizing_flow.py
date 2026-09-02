#!/usr/bin/env python3
"""Train David's official RealNVP Normalizing Flow."""

from __future__ import annotations

import argparse
import csv
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_MODULE = REPO_ROOT / "common_pipeline" / "01_contract"
SRC_DIR = REPO_ROOT / "generadores" / "david" / "src"
for module_path in (REPO_ROOT, CONTRACT_MODULE, SRC_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from constants import (  # noqa: E402
    CANONICAL_MEAN,
    CANONICAL_STD,
    CHANNEL_ORDER,
    DONOR_TRAIN_PATH,
    DONOR_TRAIN_SHA256,
    EXPECTED_TRAINING_SEED,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from io_utils import reconstruct_tensor, sha256_file, write_json  # noqa: E402
from normalizer import fit_donor_train_normalizer  # noqa: E402
from normalizing_flow import (  # noqa: E402
    FlowConfig,
    TrainingConfig,
    flatten_windows,
    save_checkpoint,
    train_real_nvp,
)

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "generadores"
    / "david"
    / "artifacts"
    / "checkpoints"
    / "normalizing_flow_seed42.npz"
)
DEFAULT_HISTORY = (
    REPO_ROOT
    / "generadores"
    / "david"
    / "artifacts"
    / "loss_history.csv"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "generadores"
    / "david"
    / "artifacts"
    / "training_manifest_seed42.json"
)
DEFAULT_CONVERGENCE_FIGURE = (
    REPO_ROOT
    / "generadores"
    / "david"
    / "artifacts"
    / "normalizing_flow_convergence.png"
)
DONOR_VALIDATION_PATH = REPO_ROOT / "data" / "features" / "windows" / "donor_validation.parquet"
DONOR_VALIDATION_SHA256 = "134f51a2ac9e546bf1a2f21f4efbf56a62bf019a08de14209058563b0a88ae23"
SOURCE_MODEL = "normalizing_flow"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_raw_windows(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    windows = [reconstruct_tensor(values) for values in frame["features_flat"]]
    return np.stack(windows, axis=0)


def _parameter_count(model) -> int:
    return int(sum(param.size for param in model.parameters().values()))


def _write_convergence_figure(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_nll = [row["train_nll"] for row in history]
    validation_nll = [row["validation_nll"] for row in history]
    best_row = min(history, key=lambda row: row["validation_nll"])

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(epochs, train_nll, label="train NLL", linewidth=1.8)
    ax.plot(epochs, validation_nll, label="donor_validation NLL", linewidth=1.8)
    ax.axvline(best_row["epoch"], color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_title("David Normalizing Flow convergence")
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative log likelihood")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_CONVERGENCE_FIGURE)
    parser.add_argument("--seed", type=int, default=EXPECTED_TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--coupling-layers", type=int, default=8)
    parser.add_argument("--scale-bound", type=float, default=1.5)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = _resolve(args.checkpoint_path)
    history_path = _resolve(args.history_path)
    manifest_path = _resolve(args.manifest_path)
    figure_path = _resolve(args.figure_path)

    normalizer = fit_donor_train_normalizer(DONOR_TRAIN_PATH)
    donor = normalizer.normalize(_load_raw_windows(DONOR_TRAIN_PATH))
    donor_validation = normalizer.normalize(_load_raw_windows(DONOR_VALIDATION_PATH))
    donor_sha = sha256_file(DONOR_TRAIN_PATH)
    if donor_sha != DONOR_TRAIN_SHA256:
        raise ValueError(f"donor_train SHA mismatch: {donor_sha} != {DONOR_TRAIN_SHA256}")
    if not np.allclose(normalizer.mean, np.asarray(CANONICAL_MEAN), rtol=0.0, atol=1.0e-12):
        raise ValueError("donor_train mean does not match canonical stats")
    if not np.allclose(normalizer.std, np.asarray(CANONICAL_STD), rtol=0.0, atol=1.0e-12):
        raise ValueError("donor_train std does not match canonical stats")
    validation_sha = sha256_file(DONOR_VALIDATION_PATH)
    if validation_sha != DONOR_VALIDATION_SHA256:
        raise ValueError(f"donor_validation SHA mismatch: {validation_sha} != {DONOR_VALIDATION_SHA256}")
    flow_config = FlowConfig(
        hidden_dims=tuple([args.hidden_dim] * args.hidden_layers),
        n_coupling_layers=args.coupling_layers,
        scale_bound=args.scale_bound,
        seed=args.seed,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        seed=args.seed,
    )

    def _progress(row: dict[str, float], improved: bool, stale_epochs: int) -> None:
        epoch = int(row["epoch"])
        if args.log_every <= 0:
            return
        if epoch == 1 or epoch % args.log_every == 0 or stale_epochs >= args.patience:
            marker = " best" if improved else ""
            print(
                "epoch "
                f"{epoch:05d} "
                f"train_nll={row['train_nll']:.6f} "
                f"val_nll={row['validation_nll']:.6f} "
                f"grad_norm={row['grad_norm']:.3f} "
                f"stale={stale_epochs}/{args.patience}"
                f"{marker}",
                flush=True,
            )

    model, history = train_real_nvp(
        donor,
        flow_config=flow_config,
        training_config=training_config,
        validation_windows=donor_validation,
        progress_callback=_progress,
    )
    train_nll = model.negative_log_likelihood(flatten_windows(donor))
    validation_nll = model.negative_log_likelihood(flatten_windows(donor_validation))
    best_row = min(history, key=lambda row: row["validation_nll"])
    code_commit = _git_commit()
    parameter_count = _parameter_count(model)
    metadata = {
        "source_model": SOURCE_MODEL,
        "model_family": "normalizing_flow",
        "owner": "david",
        "architecture": {
            "name": "RealNVP",
            "transform": "actnorm_affine_coupling_fixed_permutation",
            "components": [
                "learned_actnorm",
                "affine_coupling",
                "fixed_random_permutation",
            ],
            "invertible": True,
            "uses_log_det_jacobian": True,
            "base_distribution": "standard_normal",
            "flow_config": flow_config.to_dict(),
            "parameter_count": parameter_count,
        },
        "training": {
            "objective": "negative_log_likelihood",
            "optimizer": "Adam",
            "training_config": training_config.to_dict(),
            "training_split": "donor_train",
            "validation_split": "donor_validation",
            "selection_metric": "donor_validation_negative_log_likelihood",
            "final_full_donor_train_nll": train_nll,
            "final_donor_validation_nll": validation_nll,
            "epochs_completed": len(history),
            "best_epoch": int(best_row["epoch"]),
            "best_validation_nll": best_row["validation_nll"],
            "stopping_reason": "early_stopping" if len(history) < args.epochs else "max_epochs",
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "training_seed": args.seed,
        "space": GLOBAL_NORMALIZED_SPACE,
        "window_length": WINDOW_LENGTH,
        "n_channels": N_CHANNELS,
        "channel_order": list(CHANNEL_ORDER),
        "mean": list(CANONICAL_MEAN),
        "std": list(CANONICAL_STD),
        "donor_train_path": "data/features/windows/donor_train.parquet",
        "donor_train_sha256": donor_sha,
        "donor_validation_path": "data/features/windows/donor_validation.parquet",
        "donor_validation_sha256": validation_sha,
        "git_commit": code_commit,
    }
    save_checkpoint(checkpoint_path, model, metadata=metadata, history=history)
    checkpoint_sha = sha256_file(checkpoint_path)
    metadata["checkpoint_path"] = _relative(checkpoint_path)
    metadata["checkpoint_sha256"] = checkpoint_sha

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "batch_nll", "train_nll", "validation_nll", "grad_norm"],
        )
        writer.writeheader()
        writer.writerows(history)

    _write_convergence_figure(history, figure_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        manifest_path,
        {
            **metadata,
            "history_path": _relative(history_path),
            "history_sha256": sha256_file(history_path),
            "convergence_figure_path": _relative(figure_path),
            "convergence_figure_sha256": sha256_file(figure_path),
        },
    )

    print(f"david normalizing-flow checkpoint: {checkpoint_path}")
    print(f"checkpoint_sha256: {checkpoint_sha}")
    print(f"epochs_completed: {len(history)}")
    print(f"best_epoch: {int(best_row['epoch'])}")
    print(f"final_full_donor_train_nll: {train_nll:.6f}")
    print(f"final_donor_validation_nll: {validation_nll:.6f}")
    print(f"best_validation_nll: {metadata['training']['best_validation_nll']:.6f}")
    print(f"parameter_count: {parameter_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
