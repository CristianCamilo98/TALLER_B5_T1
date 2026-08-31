from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from .data import load_normalizer, synthetic_seed_column, synthetic_windows_to_contract_frame, synthetic_windows_to_local_frame
from .io import load_generator, save_synthetic_outputs
from .metrics import compare_parquet_splits, save_validation_report
from .paths import artifacts_dir, cristian_root, default_windows_dir, outputs_dir

OFFICIAL_SEEDS = [42, 123, 2026]


def load_gan_config(config_path: Path | None = None) -> dict:
    path = config_path or (cristian_root() / "configs" / "wgan_gp.yaml")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def official_seeds(cfg: dict | None = None) -> list[int]:
    payload = cfg or load_gan_config()
    return [int(seed) for seed in payload.get("official_seeds", OFFICIAL_SEEDS)]


def n_synthetic_windows(cfg: dict | None = None) -> int:
    payload = cfg or load_gan_config()
    return int(payload.get("n_synthetic_windows", 5000))


def run_dir_for_seed(seed: int) -> Path:
    return artifacts_dir() / f"seed_{seed}"


def synthetic_parquet_path(seed: int, n_samples: int) -> Path:
    return outputs_dir() / f"synthetic_seed{seed}_n{n_samples}.parquet"


def synthetic_parquet_path_normalized(seed: int, n_samples: int) -> Path:
    return outputs_dir() / f"synthetic_seed{seed}_n{n_samples}_normalized.parquet"


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None = None) -> Path:
    if checkpoint is not None:
        return checkpoint
    candidates = sorted((run_dir / "checkpoints").glob("generator_epoch_*.keras"))
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints en {run_dir / 'checkpoints'}")
    return candidates[-1]


def generate_synthetic_for_seed(
    seed: int,
    *,
    n_samples: int | None = None,
    run_dir: Path | None = None,
    checkpoint: Path | None = None,
    output: Path | None = None,
    validate: bool = False,
    cfg: dict | None = None,
) -> Path:
    cfg = cfg or load_gan_config()
    n = n_samples if n_samples is not None else n_synthetic_windows(cfg)
    run_dir = run_dir or run_dir_for_seed(seed)
    checkpoint = resolve_checkpoint(run_dir, checkpoint)
    normalizer = load_normalizer(run_dir / "normalizer.json")

    generator = load_generator(checkpoint)
    tf.keras.utils.set_random_seed(seed)
    noise = tf.random.normal([n, generator.input_shape[-1]])
    generated_norm = generator(noise, training=False).numpy()
    generated = normalizer.denormalize(generated_norm.astype(np.float64))

    output_path = output or synthetic_parquet_path(seed, n)
    frame = synthetic_windows_to_local_frame(
        generated,
        seed=seed,
        ratio=None,
        checkpoint=str(checkpoint),
    )
    parquet_path, csv_path = save_synthetic_outputs(frame, output_path)

    normalized_path = synthetic_parquet_path_normalized(seed, n)
    frame_normalized = synthetic_windows_to_contract_frame(
        generated_norm,
        training_seed=seed,
    )
    norm_parquet_path, norm_csv_path = save_synthetic_outputs(frame_normalized, normalized_path)

    print(
        f"seed {seed}: generadas {len(frame)} ventanas -> "
        f"{parquet_path.name}, {csv_path.name} (desnormalizado); "
        f"{norm_parquet_path.name}, {norm_csv_path.name} (normalizado)"
    )

    if validate:
        from .data import load_donor_windows

        val_frame = load_donor_windows("donor_validation", windows_dir=default_windows_dir())
        report = compare_parquet_splits(val_frame, frame)
        report_path = output_path.with_suffix(".validation.json")
        save_validation_report(report, report_path)
        print(f"seed {seed}: MMD (flat) = {report.mmd_flat:.6f} -> {report_path}")

    return output_path


def generate_synthetic_all_seeds(
    *,
    seeds: list[int] | None = None,
    n_samples: int | None = None,
    validate: bool = False,
    cfg: dict | None = None,
) -> list[Path]:
    cfg = cfg or load_gan_config()
    seed_list = seeds or official_seeds(cfg)
    return [
        generate_synthetic_for_seed(
            seed,
            n_samples=n_samples,
            validate=validate,
            cfg=cfg,
        )
        for seed in seed_list
    ]


def discover_synthetic_parquets(seeds: list[int] | None = None) -> list[Path]:
    cfg = load_gan_config()
    seed_list = seeds or official_seeds(cfg)
    n = n_synthetic_windows(cfg)
    paths: list[Path] = []
    missing: list[str] = []
    for seed in seed_list:
        path = synthetic_parquet_path(seed, n)
        if path.exists():
            paths.append(path)
        else:
            missing.append(path.name)
    if missing:
        raise FileNotFoundError(
            "Faltan parquets sintéticos en outputs/: "
            + ", ".join(missing)
            + ". Ejecuta generate_synthetic.py o notebook 05."
        )
    return paths


def load_synthetic_parquets(seeds: list[int] | None = None) -> tuple[pd.DataFrame, list[Path]]:
    paths = discover_synthetic_parquets(seeds)
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["source_parquet"] = path.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined, paths


def _seed_column(frame: pd.DataFrame) -> str:
    return synthetic_seed_column(frame)


def split_synthetic_by_seed(synthetic: pd.DataFrame) -> dict[int, pd.DataFrame]:
    col = _seed_column(synthetic)
    return {
        int(seed): synthetic.loc[synthetic[col].eq(seed)].reset_index(drop=True)
        for seed in sorted(synthetic[col].unique())
    }
