#!/usr/bin/env python3
"""Generate David's official normalized output from a trained Normalizing Flow."""

from __future__ import annotations

import argparse
import sys
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
from io_utils import flatten_window, sha256_file, stack_features, write_json  # noqa: E402
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
NVDA_VISIBLE_PATH = REPO_ROOT / "data" / "features" / "windows" / "nvda_visible.parquet"
OFFICIAL_SOURCE_MODEL = "normalizing_flow"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_OVERSAMPLE_FACTOR = 1.10
DEFAULT_MAX_DRAWS = 50_000
IDX_RANGE = CHANNEL_ORDER.index("log_high_low_range")
IDX_VOLUME = CHANNEL_ORDER.index("log1p_volume")


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
        "--physical-rejection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic rejection sampling against the official NVDA physical constraints.",
    )
    parser.add_argument("--oversample-factor", type=float, default=DEFAULT_OVERSAMPLE_FACTOR)
    parser.add_argument("--max-draws", type=int, default=DEFAULT_MAX_DRAWS)
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


def _nvda_normalized_physical_thresholds() -> tuple[np.ndarray, dict[str, object]]:
    if not NVDA_VISIBLE_PATH.is_file():
        raise FileNotFoundError(f"Missing NVDA visible windows: {NVDA_VISIBLE_PATH}")
    frame = pd.read_parquet(NVDA_VISIBLE_PATH).sort_values("window_start_date").reset_index(drop=True)
    windows = stack_features(frame)
    daily = np.concatenate([windows[0], windows[1:, -1, :]], axis=0)
    mu = daily.mean(axis=0)
    sigma = daily.std(axis=0, ddof=0)
    if np.any(sigma <= 0):
        raise ValueError("NVDA visible calibration std must be positive")
    thresholds = -mu / sigma
    return thresholds, {
        "calibration_path": "data/features/windows/nvda_visible.parquet",
        "n_unique_daily_observations": int(daily.shape[0]),
        "mean": mu.tolist(),
        "std": sigma.tolist(),
        "normalized_nonnegative_thresholds": {
            "log_high_low_range": float(thresholds[IDX_RANGE]),
            "log1p_volume": float(thresholds[IDX_VOLUME]),
        },
    }


def _nvda_physical_valid_mask(windows: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    finite = np.isfinite(windows).all(axis=(1, 2))
    nonnegative_range = (windows[:, :, IDX_RANGE] >= thresholds[IDX_RANGE]).all(axis=1)
    nonnegative_volume = (windows[:, :, IDX_VOLUME] >= thresholds[IDX_VOLUME]).all(axis=1)
    return finite & nonnegative_range & nonnegative_volume


def _sample_with_optional_rejection(
    model,
    *,
    n_windows: int,
    seed: int,
    temperature: float,
    physical_rejection: bool,
    oversample_factor: float,
    max_draws: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not physical_rejection:
        windows = sample_windows(
            model,
            n_windows=n_windows,
            seed=seed,
            temperature=temperature,
        )
        return windows, {
            "enabled": False,
            "drawn": int(n_windows),
            "accepted_before_trim": int(n_windows),
            "rejected": 0,
        }

    if oversample_factor < 1.0:
        raise ValueError("oversample_factor must be >= 1.0")
    if max_draws < n_windows:
        raise ValueError("max_draws must be >= n_windows")

    thresholds, calibration = _nvda_normalized_physical_thresholds()
    accepted: list[np.ndarray] = []
    accepted_count = 0
    total_drawn = 0
    attempt = 0
    first_draw = max(n_windows, int(np.ceil(n_windows * oversample_factor)))

    while accepted_count < n_windows and total_drawn < max_draws:
        remaining = n_windows - accepted_count
        draw_count = first_draw if attempt == 0 else max(remaining, 512)
        draw_count = min(draw_count, max_draws - total_drawn)
        candidates = sample_windows(
            model,
            n_windows=draw_count,
            seed=seed + attempt * 1_000_003,
            temperature=temperature,
        )
        valid_mask = _nvda_physical_valid_mask(candidates, thresholds)
        valid = candidates[valid_mask]
        if len(valid):
            accepted.append(valid)
            accepted_count += int(len(valid))
        total_drawn += int(draw_count)
        attempt += 1

    if accepted_count < n_windows:
        raise RuntimeError(
            f"Physical rejection accepted {accepted_count} windows after {total_drawn} draws; "
            f"need {n_windows}."
        )

    stacked = np.concatenate(accepted, axis=0)
    return stacked[:n_windows], {
        "enabled": True,
        "mode": "deterministic_rejection_sampling_no_clipping",
        "drawn": int(total_drawn),
        "accepted_before_trim": int(accepted_count),
        "rejected": int(total_drawn - accepted_count),
        "oversample_factor": float(oversample_factor),
        "max_draws": int(max_draws),
        "calibration": calibration,
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
    windows, sampling_metadata = _sample_with_optional_rejection(
        model,
        n_windows=args.n_windows,
        seed=args.seed,
        temperature=args.temperature,
        physical_rejection=args.physical_rejection,
        oversample_factor=args.oversample_factor,
        max_draws=args.max_draws,
    )
    if not args.no_repro_check:
        reproduced, reproduced_sampling = _sample_with_optional_rejection(
            model,
            n_windows=args.n_windows,
            seed=args.seed,
            temperature=args.temperature,
            physical_rejection=args.physical_rejection,
            oversample_factor=args.oversample_factor,
            max_draws=args.max_draws,
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
    print(f"physical_rejection: {sampling_metadata['enabled']}")
    print(f"checkpoint_sha256: {checkpoint_sha}")
    print(f"sha256: {output_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
