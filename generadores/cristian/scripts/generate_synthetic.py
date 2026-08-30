#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_normalizer, synthetic_windows_to_frame
from src.io import load_generator, save_synthetic_parquet
from src.metrics import compare_parquet_splits, save_validation_report
from src.paths import artifacts_dir, default_windows_dir, outputs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera ventanas sintéticas con WGAN-GP.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "wgan_gp.yaml")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directorio del entrenamiento (artifacts/seed_XX)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Ruta al generator .keras")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Número de ventanas sintéticas (default: n_synthetic_windows del yaml, 5000)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validate", action="store_true", help="Comparar contra donor_validation")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint
    candidates = sorted((run_dir / "checkpoints").glob("generator_epoch_*.keras"))
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints en {run_dir / 'checkpoints'}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    run_dir = args.run_dir if args.run_dir.is_absolute() else artifacts_dir() / args.run_dir
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    normalizer = load_normalizer(run_dir / "normalizer.json")

    n_samples = args.n_samples if args.n_samples is not None else int(cfg.get("n_synthetic_windows", 5000))
    seed = args.seed if args.seed is not None else int(run_dir.name.split("_")[-1])

    generator = load_generator(checkpoint)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    noise = tf.random.normal([n_samples, generator.input_shape[-1]])
    generated_norm = generator(noise, training=False).numpy()
    generated = normalizer.denormalize(generated_norm.astype(np.float64))

    output = args.output or outputs_dir() / f"synthetic_seed{seed}_n{n_samples}.parquet"
    frame = synthetic_windows_to_frame(
        generated,
        seed=seed,
        ratio=None,
        checkpoint=str(checkpoint),
    )
    save_synthetic_parquet(frame, output)
    print(f"Generadas {len(frame)} ventanas -> {output}")

    if args.validate:
        from src.data import load_donor_windows

        val_frame = load_donor_windows("donor_validation", windows_dir=default_windows_dir())
        report = compare_parquet_splits(val_frame, frame)
        report_path = output.with_suffix(".validation.json")
        save_validation_report(report, report_path)
        print(f"MMD (flat): {report.mmd_flat:.6f}")
        print(f"Informe de validación -> {report_path}")


if __name__ == "__main__":
    main()
