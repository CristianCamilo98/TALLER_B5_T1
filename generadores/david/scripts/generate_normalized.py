#!/usr/bin/env python3
"""Generate David's official normalized output from a trained Normalizing Flow."""

from __future__ import annotations

import argparse
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_MODULE = REPO_ROOT / "common_pipeline" / "01_contract"
SRC_DIR = REPO_ROOT / "generadores" / "david" / "src"
for module_path in (REPO_ROOT, CONTRACT_MODULE, SRC_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from constants import (  # noqa: E402
    BASELINE_OUTPUT_NAME,
    CANONICAL_MEAN,
    CANONICAL_STD,
    CHANNEL_ORDER,
    DONOR_TRAIN_PATH,
    EXPECTED_ROWS,
    EXPECTED_TRAINING_SEED,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from io_utils import flatten_window, sha256_file, write_json  # noqa: E402
from normalizing_flow import load_checkpoint, sample_windows  # noqa: E402

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "generadores" / "david" / "outputs" / BASELINE_OUTPUT_NAME
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "generadores"
    / "david"
    / "artifacts"
    / "checkpoints"
    / "normalizing_flow_seed42.npz"
)
OFFICIAL_SOURCE_MODEL = "normalizing_flow"
DEFAULT_TEMPERATURE = 1.0


def make_canonical_frame(
    windows: np.ndarray,
    *,
    source_model: str = OFFICIAL_SOURCE_MODEL,
    training_seed: int = EXPECTED_TRAINING_SEED,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=EXPECTED_TRAINING_SEED)
    parser.add_argument("--n-windows", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Prior sampling temperature selected for the official run.",
    )
    parser.add_argument(
        "--no-repro-check",
        action="store_true",
        help="Skip the deterministic sampling reproducibility check.",
    )
    return parser.parse_args()


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


def _sample_official_windows(
    model,
    *,
    n_windows: int,
    seed: int,
    temperature: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    windows = sample_windows(
        model,
        n_windows=n_windows,
        seed=seed,
        temperature=temperature,
    )
    return windows, {
        "mode": "direct_base_distribution_sampling",
        "drawn": int(n_windows),
        "accepted": int(n_windows),
        "rejected": 0,
    }


def main() -> int:
    args = parse_args()
    checkpoint_path = _resolve(args.checkpoint_path)
    output_path = _resolve(args.output_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Missing Normalizing Flow checkpoint. Run "
            "`python generadores/david/scripts/train_normalizing_flow.py` first."
        )

    model, checkpoint_metadata, history = load_checkpoint(checkpoint_path)
    windows, sampling_metadata = _sample_official_windows(
        model,
        n_windows=args.n_windows,
        seed=args.seed,
        temperature=args.temperature,
    )
    if not args.no_repro_check:
        reproduced, reproduced_sampling = _sample_official_windows(
            model,
            n_windows=args.n_windows,
            seed=args.seed,
            temperature=args.temperature,
        )
        if not np.array_equal(windows, reproduced) or sampling_metadata != reproduced_sampling:
            raise RuntimeError("Normalizing Flow sampling is not reproducible")

    frame = make_canonical_frame(
        windows,
        source_model=OFFICIAL_SOURCE_MODEL,
        training_seed=args.seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    output_sha = sha256_file(output_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    donor_sha = sha256_file(DONOR_TRAIN_PATH)
    write_json(
        output_path.with_suffix(".provenance.json"),
        {
            "generator": OFFICIAL_SOURCE_MODEL,
            "owner": "david",
            "source_model": OFFICIAL_SOURCE_MODEL,
            "model_family": "normalizing_flow",
            "architecture": checkpoint_metadata["architecture"],
            "training": checkpoint_metadata["training"],
            "algorithm": "RealNVP ActNorm affine-coupling normalizing flow",
            "seed": args.seed,
            "training_seed": args.seed,
            "sampling_temperature": args.temperature,
            "sampling": sampling_metadata,
            "space": GLOBAL_NORMALIZED_SPACE,
            "channel_order": list(CHANNEL_ORDER),
            "mean": list(CANONICAL_MEAN),
            "std": list(CANONICAL_STD),
            "donor_train_path": "data/features/windows/donor_train.parquet",
            "donor_train_sha256": donor_sha,
            "checkpoint_path": _relative(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_training_seed": checkpoint_metadata["training_seed"],
            "checkpoint_history_rows": len(history),
            "git_commit": _git_commit(),
            "n_windows": int(len(windows)),
            "logical_shape": [int(len(windows)), WINDOW_LENGTH, N_CHANNELS],
            "parquet_sha256": output_sha,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    print(f"david normalizing-flow output: {output_path}")
    print(f"shape: {(args.n_windows, WINDOW_LENGTH, N_CHANNELS)}")
    print(f"seed: {args.seed}")
    print(f"source_model: {OFFICIAL_SOURCE_MODEL}")
    print(f"checkpoint_sha256: {checkpoint_sha}")
    print(f"sha256: {output_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
