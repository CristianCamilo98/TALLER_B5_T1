#!/usr/bin/env python3
"""Train David's official RealNVP Normalizing Flow."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    EXPECTED_TRAINING_SEED,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from io_utils import sha256_file, write_json  # noqa: E402
from normalizer import load_donor_train_normalized  # noqa: E402
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
    / "training_history_seed42.csv"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "generadores"
    / "david"
    / "artifacts"
    / "training_manifest_seed42.json"
)
SOURCE_MODEL = "normalizing_flow"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=EXPECTED_TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--coupling-layers", type=int, default=6)
    parser.add_argument("--scale-clip", type=float, default=1.5)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = _resolve(args.checkpoint_path)
    history_path = _resolve(args.history_path)
    manifest_path = _resolve(args.manifest_path)

    donor = load_donor_train_normalized(DONOR_TRAIN_PATH)
    donor_sha = sha256_file(DONOR_TRAIN_PATH)
    flow_config = FlowConfig(
        hidden_dims=tuple([args.hidden_dim] * args.hidden_layers),
        n_coupling_layers=args.coupling_layers,
        scale_clip=args.scale_clip,
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
    model, history = train_real_nvp(
        donor,
        flow_config=flow_config,
        training_config=training_config,
    )
    train_nll = model.negative_log_likelihood(flatten_windows(donor))
    metadata = {
        "source_model": SOURCE_MODEL,
        "model_family": "normalizing_flow",
        "architecture": {
            "name": "RealNVP",
            "transform": "affine_coupling",
            "invertible": True,
            "uses_log_det_jacobian": True,
            "base_distribution": "standard_normal",
            "flow_config": flow_config.to_dict(),
        },
        "training": {
            "objective": "negative_log_likelihood",
            "optimizer": "Adam",
            "training_config": training_config.to_dict(),
            "final_full_donor_train_nll": train_nll,
            "epochs_completed": len(history),
            "best_validation_nll": min(row["validation_nll"] for row in history),
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

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        manifest_path,
        {
            **metadata,
            "history_path": _relative(history_path),
            "history_sha256": sha256_file(history_path),
        },
    )

    print(f"david normalizing-flow checkpoint: {checkpoint_path}")
    print(f"checkpoint_sha256: {checkpoint_sha}")
    print(f"epochs_completed: {len(history)}")
    print(f"final_full_donor_train_nll: {train_nll:.6f}")
    print(f"best_validation_nll: {metadata['training']['best_validation_nll']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
