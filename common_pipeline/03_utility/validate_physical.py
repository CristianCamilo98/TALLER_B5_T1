"""
Common physical validator (Part 2).

Calibra cada pool sintetico con la formula afin OFICIAL (Part 1) -- sin
re-estandarizar el pool generado, para que los errores del generador se
vean tal cual, no se enmascaren. Despues valida fisicamente cada ventana
calibrada. Sin reparacion: solo VALID/INVALID.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import importlib

io_synthetic = importlib.import_module("common_pipeline.03_utility.io_synthetic")
discover_synthetic_pools = io_synthetic.discover_synthetic_pools
CHANNEL_ORDER = io_synthetic.CHANNEL_ORDER

CALIBRATION_PATH = "common_pipeline/03_utility/results/tables/nvda_calibration.csv"
OUTPUT_PATH = Path("common_pipeline/03_utility/results/tables/physical_validity.csv")

IDX_RETURN, IDX_RANGE, IDX_VOLUME = 0, 1, 2


def load_calibration() -> tuple[np.ndarray, np.ndarray]:
    calib = pd.read_csv(CALIBRATION_PATH).set_index("channel").loc[list(CHANNEL_ORDER)]
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


def main() -> None:
    mu, sigma = load_calibration()
    pools = discover_synthetic_pools()

    rows = []
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
        cache_dir = Path("common_pipeline/03_utility/results/calibrated_pools")
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_dir / f"{method}_calibrated.npz",
            values=pool_calibrated,
            valid_mask=valid_mask,
        )

        if n_valid < 186:
            print(f"[FAIL] {method}: solo {n_valid} ventanas validas (< 186 requeridas para mix_75)")

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(result.to_string(index=False))
    print(f"\nGuardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
