"""
Common mixture design + subsampling + downstream scaler (Parts 6, 7, 8).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

features_target = importlib.import_module("common_pipeline.03_utility.features_target")
io_synthetic = importlib.import_module("common_pipeline.03_utility.io_synthetic")
utility_run = importlib.import_module("common_pipeline.03_utility.utility_run")

build_features_and_target = features_target.build_features_and_target
load_real_visible_windows = features_target.load_real_visible_windows
FEATURE_NAMES = features_target.FEATURE_NAMES
CHANNEL_ORDER = io_synthetic.CHANNEL_ORDER

N_REAL = 62
RATIOS = (0.0, 0.25, 0.50, 0.75)
SUBSAMPLING_SEEDS = (42, 123, 2026)
MIN_VALID_FOR_75PCT = 186

def n_synthetic_for_ratio(n_real: int, ratio: float) -> int:
    """synthetic_share = n_synthetic / (n_real + n_synthetic) -- despejando:
    n_synthetic = n_real * ratio / (1 - ratio)"""
    if ratio == 0.0:
        return 0
    return int(round(n_real * ratio / (1 - ratio)))


def load_valid_calibrated_pool(method: str, *, run_dir: Path) -> np.ndarray:
    """Carga el pool calibrado (Part 2) y devuelve SOLO las ventanas validas,
    ordenadas deterministamente por posicion (indice de fila == synthetic_id
    implicito), tal y como exige la spec."""
    manifest = utility_run.load_run(run_dir)
    artifacts = manifest.get("calibrated_pools", {})
    if method not in artifacts:
        raise KeyError(f"Method {method!r} is not registered in run {manifest['run_id']}")
    expected_path = utility_run.calibrated_pool_path(run_dir, method)
    registered_path = utility_run.resolve_run_artifact(run_dir, artifacts[method]["path"])
    if registered_path != expected_path.resolve():
        raise RuntimeError(f"Calibrated path mismatch for {method}")
    actual_hash = utility_run.contract_registry.registry_sha256(expected_path)
    if actual_hash != artifacts[method]["sha256"]:
        raise RuntimeError(f"Calibrated cache SHA mismatch for {method}")
    data = np.load(expected_path)
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
    real_windows = load_real_visible_windows()

    design_rows = []
    for ratio in RATIOS:
        n_synth = n_synthetic_for_ratio(N_REAL, ratio)
        design_rows.append({"ratio": ratio, "n_real": N_REAL, "n_synthetic": n_synth, "n_total": N_REAL + n_synth})
    design_df = pd.DataFrame(design_rows)
    mixture_design_path = run_dir / "tables" / "mixture_design.csv"
    mixture_design_path.parent.mkdir(parents=True, exist_ok=True)
    design_df.to_csv(mixture_design_path, index=False)
    print(design_df.to_string(index=False))
    print(f"\nGuardado: {mixture_design_path}")

    scaler = fit_downstream_scaler(real_windows)
    print(f"\nScaler ajustado con {N_REAL} ventanas reales -- mean(features): {scaler.mean_.round(3)}")

    manifest = utility_run.load_run(run_dir)
    methods = sorted(manifest.get("calibrated_pools", {}))
    print(f"\nMetodos con pool calibrado disponible: {methods}")
    for method in methods:
        valid_pool = load_valid_calibrated_pool(method, run_dir=run_dir)
        status = "OK" if len(valid_pool) >= MIN_VALID_FOR_75PCT else "FAIL"
        print(f"  {method:10s}: {len(valid_pool)} ventanas validas -> {status} (minimo {MIN_VALID_FOR_75PCT} para mix_75)")
    utility_run.update_run_manifest(
        run_dir,
        {"mixture_design_table": utility_run.run_relative_path(run_dir, mixture_design_path)},
    )


if __name__ == "__main__":
    main()
