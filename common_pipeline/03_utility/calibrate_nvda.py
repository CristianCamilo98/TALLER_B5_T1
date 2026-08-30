"""
Common NVDA calibration (Part 1).

Reconstruye las 126 observaciones diarias UNICAS de NVDA visible (no las 62
ventanas solapadas con stride=1) y calcula mu/sigma por canal, ddof=0.
Esta es la UNICA calibracion oficial del proyecto -- ningun metodo debe usar
una calibracion individual propia a partir de aqui.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NVDA_VISIBLE_PATH = "data/features/windows/nvda_visible.parquet"
CHANNEL_ORDER = ("log_return", "log_high_low_range", "log1p_volume")
WINDOW_LENGTH = 65
OUTPUT_PATH = Path("common_pipeline/03_utility/results/tables/nvda_calibration.csv")


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


def main() -> None:
    daily, start_date, end_date = reconstruct_unique_daily(NVDA_VISIBLE_PATH)
    n_unique_days = daily.shape[0]

    mean = daily.mean(axis=0)
    std = daily.std(axis=0, ddof=0)

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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"n_unique_days reconstruidos: {n_unique_days} (esperado: 126)")
    print(result.to_string(index=False))
    print(f"\nGuardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
