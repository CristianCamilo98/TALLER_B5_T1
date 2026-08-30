#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    fit_normalizer,
    load_donor_windows,
    load_experiment_config,
    save_normalizer,
    windows_to_array,
)
from src.paths import artifacts_dir, default_windows_dir
from src.wgan_gp import TrainConfig, WGAN_GP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena WGAN-GP sobre ventanas donor.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "wgan_gp.yaml")
    parser.add_argument("--windows-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    experiment = load_experiment_config()

    seed = args.seed if args.seed is not None else int(cfg.get("seed", experiment["synthetic_experiment"]["seeds"][0]))
    train_cfg = TrainConfig(
        latent_dim=int(cfg.get("latent_dim", 100)),
        batch_size=args.batch_size or int(cfg.get("batch_size", 64)),
        epochs=args.epochs or int(cfg.get("epochs", 5000)),
        n_critic=int(cfg.get("n_critic", 5)),
        lambda_gp=float(cfg.get("lambda_gp", 10.0)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        sample_interval=int(cfg.get("sample_interval", 200)),
        seed=seed,
    )

    windows_dir = args.windows_dir or default_windows_dir()
    train_frame = load_donor_windows("donor_train", windows_dir=windows_dir)
    train_windows = windows_to_array(train_frame)
    normalizer = fit_normalizer(train_windows)

    run_name = args.run_name or f"seed_{seed}"
    run_dir = artifacts_dir() / run_name
    save_normalizer(normalizer, run_dir / "normalizer.json")

    print(f"Train windows: {train_windows.shape}")
    print(f"Run dir: {run_dir}")
    gan = WGAN_GP(train_cfg)
    gan.train(train_windows, run_dir=run_dir, normalizer=normalizer)
    print(f"Entrenamiento completado. Artefactos en {run_dir}")


if __name__ == "__main__":
    main()
