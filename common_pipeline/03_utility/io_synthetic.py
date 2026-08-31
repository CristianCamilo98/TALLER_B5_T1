"""
Descubre y carga los pools sinteticos normalizados de cada generador.
Unico denominador comun real entre los 4 esquemas de columnas distintos:
la columna `features_flat` (195 floats, row-major t=0..64 x canal).
Identificamos el metodo por la carpeta (generadores/<nombre>/), no por
columnas de metadata inconsistentes entre generadores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_LENGTH = 65
N_CHANNELS = 3
CHANNEL_ORDER = ("log_return", "log_high_low_range", "log1p_volume")

# Fichero normalizado oficial de seed=42 por generador -- fijado explicito,
# no adivinado por heuristica de nombre (Cristian tiene raw Y normalized en
# la misma carpeta; un match de texto ambiguo podria coger el equivocado).
KNOWN_NORMALIZED_FILES = {
    "cristian": "generadores/cristian/outputs/synthetic_seed42_n5000_normalized.parquet",
    "david": "generadores/david/outputs/bootstrap_jitter_seed42_normalized.parquet",
    "daniel": "generadores/daniel/outputs/diffusion_seed42_normalized.parquet",
    "marco": "generadores/marco/outputs/donor_synthetic_normalized_seed42.parquet",
}


def discover_synthetic_pools(extra_paths: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    """extra_paths permite anadir el baseline de 01_contract via CLI/path
    opcional cuando exista, sin tocar el diccionario fijo de arriba."""
    paths = dict(KNOWN_NORMALIZED_FILES)
    if extra_paths:
        paths.update(extra_paths)

    pools = {}
    for method, path in paths.items():
        p = Path(path)
        if not p.exists():
            print(f"[aviso] {method}: no encontrado en {path}, se omite")
            continue
        df = pd.read_parquet(p)
        if "features_flat" not in df.columns:
            raise ValueError(f"{method}: {path} no tiene columna 'features_flat'")
        values = np.stack([
            np.asarray(row, dtype="float32").reshape(WINDOW_LENGTH, N_CHANNELS)
            for row in df["features_flat"]
        ])
        pools[method] = values
        print(f"{method:10s}: {values.shape} cargado de {path}")
    return pools


if __name__ == "__main__":
    pools = discover_synthetic_pools()
    print(f"\nMetodos encontrados: {list(pools.keys())}")
