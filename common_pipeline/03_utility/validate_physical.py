"""
Common physical validator (Part 2).

Calibra cada pool sintetico con la formula afin OFICIAL (Part 1) -- sin
re-estandarizar el pool generado, para que los errores del generador se
vean tal cual, no se enmascaren. Despues valida fisicamente cada ventana
calibrada. Sin reparacion: solo VALID/INVALID.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import importlib

io_synthetic = importlib.import_module("common_pipeline.03_utility.io_synthetic")
utility_run = importlib.import_module("common_pipeline.03_utility.utility_run")
CHANNEL_ORDER = io_synthetic.CHANNEL_ORDER

IDX_RETURN, IDX_RANGE, IDX_VOLUME = 0, 1, 2


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    calib = pd.read_csv(path).set_index("channel").loc[list(CHANNEL_ORDER)]
    return calib["mean"].to_numpy(), calib["std"].to_numpy()


def calibrate(pool_normalized: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """X_syn_NVDA = mu + sigma * Z_syn -- afin, exacta, SIN re-estandarizar
    el pool antes (esa re-estandarizacion ocultaria el colapso de varianza
    u otros errores del generador, por eso la spec la prohibe)."""
    return mu.reshape(1, 1, -1) + sigma.reshape(1, 1, -1) * pool_normalized


def validate_window(window: np.ndarray) -> bool:
    """Reglas fisicas, por canal. Una ventana es INVALID si CUALQUIER
    elemento incumple. Sin abs()/clip()/winsorize -- solo se reporta."""
    ret, rng, vol = window[:, IDX_RETURN], window[:, IDX_RANGE], window[:, IDX_VOLUME]

    if not np.all(np.isfinite(ret)):
        return False
    if not (np.all(np.isfinite(rng)) and np.all(rng >= 0)):
        return False
    if not (np.all(np.isfinite(vol)) and np.all(vol >= 0)):
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=utility_run.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--results-root", type=Path, default=utility_run.RESULTS_ROOT)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = utility_run.existing_run_dir(
        results_root=args.results_root,
        registry_path=args.registry_path,
        run_id=args.run_id,
    )
    calibration_path = run_dir / "tables" / "nvda_calibration.csv"
    mu, sigma = load_calibration(calibration_path)
    pools = io_synthetic.load_pools_for_run(run_dir)

    rows = []
    calibrated_artifacts = {}
    for method, pool_normalized in pools.items():
        pool_calibrated = calibrate(pool_normalized, mu, sigma)

        valid_mask = np.array([validate_window(w) for w in pool_calibrated])
        n_generated = len(pool_calibrated)
        n_valid = int(valid_mask.sum())
        n_invalid = n_generated - n_valid

        rows.append({
            "method": method,
            "generated": n_generated,
            "valid": n_valid,
            "invalid": n_invalid,
            "invalid_rate": n_invalid / n_generated,
        })

        # Guardamos el pool calibrado + la mascara -- Part 3 en adelante
        # consume esto, ordenado deterministamente por posicion (synthetic_id
        # implicito = indice de fila), sin necesidad de recalibrar cada vez.
        cache_dir = run_dir / "calibrated_pools"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = utility_run.calibrated_pool_path(run_dir, method)
        np.savez_compressed(
            cache_path,
            values=pool_calibrated,
            valid_mask=valid_mask,
        )
        calibrated_artifacts[method] = {
            "path": utility_run.run_relative_path(run_dir, cache_path),
            "sha256": utility_run.contract_registry.registry_sha256(cache_path),
            "valid": n_valid,
            "invalid": n_invalid,
        }

        if n_valid < 186:
            print(f"[FAIL] {method}: solo {n_valid} ventanas validas (< 186 requeridas para mix_75)")

    result = pd.DataFrame(rows)
    output_path = run_dir / "tables" / "physical_validity.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    utility_run.update_run_manifest(
        run_dir,
        {
            "calibrated_pools": calibrated_artifacts,
            "physical_validity_table": utility_run.run_relative_path(run_dir, output_path),
        },
    )

    print(result.to_string(index=False))
    print(f"\nGuardado: {output_path}")


if __name__ == "__main__":
    main()
