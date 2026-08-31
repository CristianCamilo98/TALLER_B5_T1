"""
Common NVDA calibration (Part 1).

Reconstruye las 126 observaciones diarias UNICAS de NVDA visible (no las 62
ventanas solapadas con stride=1) y calcula mu/sigma por canal, ddof=0.
Esta es la UNICA calibracion oficial del proyecto -- ningun metodo debe usar
una calibracion individual propia a partir de aqui.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import utility_run

NVDA_VISIBLE_PATH = utility_run.NVDA_VISIBLE_PATH
CHANNEL_ORDER = ("log_return", "log_high_low_range", "log1p_volume")
WINDOW_LENGTH = 65


def reconstruct_unique_daily(path: str) -> tuple[np.ndarray, object, object]:
    """
    Las ventanas de nvda_visible tienen stride=1: cada ventana comparte 64
    de sus 65 dias con la anterior, aportando solo 1 dia nuevo. Reconstruimos
    la serie diaria real concatenando la primera ventana completa + el
    ultimo dia de cada ventana siguiente -- sin esto, promediar sobre las 62
    ventanas aplanadas pesaria los dias centrales del semestre hasta 62x mas
    que los de los bordes.
    """
    df = pd.read_parquet(path).sort_values("window_start_date").reset_index(drop=True)
    windows = np.stack([
        np.asarray(row, dtype="float64").reshape(WINDOW_LENGTH, len(CHANNEL_ORDER))
        for row in df["features_flat"]
    ])

    primera = windows[0]
    resto = windows[1:, -1, :]
    daily = np.concatenate([primera, resto], axis=0)

    return daily, df["window_start_date"].iloc[0], df["window_end_date"].iloc[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=utility_run.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--results-root", type=Path, default=utility_run.RESULTS_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir, manifest = utility_run.prepare_run(
        registry_path=args.registry_path,
        results_root=args.results_root,
        run_id=args.run_id,
        allow_partial=args.allow_partial,
    )
    daily, start_date, end_date = reconstruct_unique_daily(str(NVDA_VISIBLE_PATH))
    n_unique_days = daily.shape[0]
    if n_unique_days != 126 or not np.isfinite(daily).all():
        raise ValueError("NVDA visible calibration must contain 126 finite unique days")

    mean = daily.mean(axis=0)
    std = daily.std(axis=0, ddof=0)
    if np.any(std <= 0):
        raise ValueError("NVDA visible calibration std must be positive")

    rows = [
        {
            "channel": channel,
            "n_unique_days": n_unique_days,
            "mean": mean[i],
            "std": std[i],
            "ddof": 0,
            "start_date": start_date,
            "end_date": end_date,
        }
        for i, channel in enumerate(CHANNEL_ORDER)
    ]
    result = pd.DataFrame(rows)

    output_path = run_dir / "tables" / "nvda_calibration.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    calibration = dict(manifest["nvda_calibration"])
    calibration.update(
        {
            "n_unique_daily_observations": n_unique_days,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "table_path": utility_run.run_relative_path(run_dir, output_path),
        }
    )
    utility_run.update_run_manifest(run_dir, {"nvda_calibration": calibration})

    print(f"n_unique_days reconstruidos: {n_unique_days} (esperado: 126)")
    print(result.to_string(index=False))
    print(f"\nGuardado: {output_path}")


if __name__ == "__main__":
    main()
