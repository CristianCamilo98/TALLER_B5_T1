#!/usr/bin/env python3
"""Generate David's official normalized temporal-jitter synthetic pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_MODULE = REPO_ROOT / "common_pipeline" / "01_contract"
SCRIPT_DIR = Path(__file__).resolve().parent
for module_path in (REPO_ROOT, CONTRACT_MODULE, SCRIPT_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from constants import (  # noqa: E402
    BASELINE_OUTPUT_NAME,
    DONOR_TRAIN_PATH,
    EXPECTED_TRAINING_SEED,
    EXPECTED_ROWS,
)
from normalizer import load_donor_train_normalized  # noqa: E402
from io_utils import sha256_file  # noqa: E402

from experiment_normalized import (  # noqa: E402
    CandidateSpec,
    generate_candidate_windows,
    write_official_output,
)

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "generadores" / "david" / "outputs" / BASELINE_OUTPUT_NAME
)
OFFICIAL_SOURCE_MODEL = "temporal_jitter_0p40_rho0p85"
OFFICIAL_NOISE_SCALE = 0.40
OFFICIAL_RHO = 0.85


def source_model_name(noise_scale: float, rho: float) -> str:
    if noise_scale == OFFICIAL_NOISE_SCALE and rho == OFFICIAL_RHO:
        return OFFICIAL_SOURCE_MODEL
    noise_tag = f"{noise_scale:.6g}".replace(".", "p")
    rho_tag = f"{rho:.6g}".replace(".", "p")
    return f"temporal_jitter_{noise_tag}_rho{rho_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Official David parquet path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=EXPECTED_TRAINING_SEED,
        help="Generation seed. The official protocol uses 42.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=OFFICIAL_NOISE_SCALE,
        help="Temporal Gaussian jitter standard deviation in normalized space.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=OFFICIAL_RHO,
        help="AR(1) persistence for temporal jitter.",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=EXPECTED_ROWS,
        help="Number of synthetic windows to generate. The official protocol uses 5000.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output_path
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    donor = load_donor_train_normalized(DONOR_TRAIN_PATH)
    spec = CandidateSpec(
        name=source_model_name(args.noise_scale, args.rho),
        family="bootstrap_temporal",
        noise_scale=args.noise_scale,
        rho=args.rho,
    )
    windows = generate_candidate_windows(
        donor,
        spec,
        seed=args.seed,
        n_windows=args.n_windows,
    )
    if output_path != DEFAULT_OUTPUT_PATH:
        from experiment_normalized import make_canonical_frame  # noqa: PLC0415

        frame = make_canonical_frame(
            windows,
            source_model=spec.name,
            training_seed=args.seed,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        result_path = output_path
    else:
        result_path = write_official_output(
            windows,
            source_model=spec.name,
            selected_from="generadores/david/scripts/generate_normalized.py",
            training_seed=args.seed,
            family=spec.family,
            noise_scale=spec.noise_scale,
            rho=spec.rho,
        )

    digest = sha256_file(result_path)
    print(f"david normalized output: {result_path}")
    print(f"shape: {(args.n_windows, 65, 3)}")
    print(f"seed: {args.seed}")
    print(f"noise_scale: {args.noise_scale}")
    print(f"rho: {args.rho}")
    print(f"source_model: {spec.name}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
