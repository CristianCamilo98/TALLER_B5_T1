#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic_io import (
    generate_synthetic_all_seeds,
    generate_synthetic_for_seed,
    load_gan_config,
    official_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera ventanas sintéticas WGAN-GP para las seeds oficiales (42, 123, 2026)."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "wgan_gp.yaml")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Generar solo una seed (default: las 3 oficiales del yaml)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Ventanas por seed (default: n_synthetic_windows del yaml, 5000)",
    )
    parser.add_argument("--validate", action="store_true", help="Validar cada parquet vs donor_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_gan_config(args.config)

    if args.seed is not None:
        generate_synthetic_for_seed(
            args.seed,
            n_samples=args.n_samples,
            validate=args.validate,
            cfg=cfg,
        )
        return

    paths = generate_synthetic_all_seeds(
        seeds=official_seeds(cfg),
        n_samples=args.n_samples,
        validate=args.validate,
        cfg=cfg,
    )
    print(f"Completado: {len(paths)} parquets desnormalizados (+ normalized por seed).")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
