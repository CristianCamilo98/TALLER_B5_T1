"""
Common mixture design + subsampling + downstream scaler (Parts 6, 7, 8).
"""
from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

features_target = importlib.import_module("common_pipeline.03_utility.features_target")
io_synthetic = importlib.import_module("common_pipeline.03_utility.io_synthetic")

build_features_and_target = features_target.build_features_and_target
load_real_visible_windows = features_target.load_real_visible_windows
FEATURE_NAMES = features_target.FEATURE_NAMES
CHANNEL_ORDER = io_synthetic.CHANNEL_ORDER

N_REAL = 62
RATIOS = (0.0, 0.25, 0.50, 0.75)
SUBSAMPLING_SEEDS = (42, 123, 2026)
MIN_VALID_FOR_75PCT = 186

CALIBRATED_POOLS_DIR = Path("common_pipeline/03_utility/results/calibrated_pools")
MIXTURE_DESIGN_PATH = Path("common_pipeline/03_utility/results/tables/mixture_design.csv")


def n_synthetic_for_ratio(n_real: int, ratio: float) -> int:
    """synthetic_share = n_synthetic / (n_real + n_synthetic) -- despejando:
    n_synthetic = n_real * ratio / (1 - ratio)"""
    if ratio == 0.0:
        return 0
    return int(round(n_real * ratio / (1 - ratio)))


def load_valid_calibrated_pool(method: str) -> np.ndarray:
    """Carga el pool calibrado (Part 2) y devuelve SOLO las ventanas validas,
    ordenadas deterministamente por posicion (indice de fila == synthetic_id
    implicito), tal y como exige la spec."""
    data = np.load(CALIBRATED_POOLS_DIR / f"{method}_calibrated.npz")
    values, valid_mask = data["values"], data["valid_mask"]
    valid_values = values[valid_mask]  # el orden de np.where/mascara ya preserva la posicion original
    return valid_values


def build_mixture(real_windows: np.ndarray, synthetic_valid: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Real-only (ratio=0) es deterministico -- no se samplea. Para ratio>0,
    subsampling SIN reemplazo desde el pool valido, con la seed dada."""
    n_synth = n_synthetic_for_ratio(N_REAL, ratio)
    if n_synth == 0:
        return real_windows
    if len(synthetic_valid) < n_synth:
        raise ValueError(f"pool valido insuficiente: {len(synthetic_valid)} < {n_synth} requeridas")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(synthetic_valid), size=n_synth, replace=False)
    return np.concatenate([real_windows, synthetic_valid[idx]], axis=0)


def fit_downstream_scaler(real_windows: np.ndarray) -> StandardScaler:
    """Part 8: ajustado UNA UNICA VEZ, solo con las 62 reales visibles.
    Prohibido reajustar por mixture o con sintetico -- asi el generador no
    puede alterar indirectamente la representacion de las features."""
    X_real, _ = build_features_and_target(real_windows)
    return StandardScaler().fit(X_real)


def main() -> None:
    real_windows = load_real_visible_windows()

    design_rows = []
    for ratio in RATIOS:
        n_synth = n_synthetic_for_ratio(N_REAL, ratio)
        design_rows.append({"ratio": ratio, "n_real": N_REAL, "n_synthetic": n_synth, "n_total": N_REAL + n_synth})
    design_df = pd.DataFrame(design_rows)
    MIXTURE_DESIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
    design_df.to_csv(MIXTURE_DESIGN_PATH, index=False)
    print(design_df.to_string(index=False))
    print(f"\nGuardado: {MIXTURE_DESIGN_PATH}")

    scaler = fit_downstream_scaler(real_windows)
    print(f"\nScaler ajustado con {N_REAL} ventanas reales -- mean(features): {scaler.mean_.round(3)}")

    methods = [p.stem.replace("_calibrated", "") for p in CALIBRATED_POOLS_DIR.glob("*_calibrated.npz")]
    print(f"\nMetodos con pool calibrado disponible: {methods}")
    for method in methods:
        valid_pool = load_valid_calibrated_pool(method)
        status = "OK" if len(valid_pool) >= MIN_VALID_FOR_75PCT else "FAIL"
        print(f"  {method:10s}: {len(valid_pool)} ventanas validas -> {status} (minimo {MIN_VALID_FOR_75PCT} para mix_75)")


if __name__ == "__main__":
    main()
