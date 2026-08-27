#!/usr/bin/env python3
"""Features diarias generativas + ventanas T=65 (solo lectura de data/clean/).

Uso:
  .venv/bin/python scripts/build_features_windows.py

Entrada (no se modifica):
  data/clean/ohlcv_clean.parquet

Salidas:
  data/features/daily_features.parquet
  data/features/daily_features.csv
  data/features/windows_65.parquet
  data/features/features_manifest.json
  data/features/checksums.sha256
  data/features/README.md  (mantener alineado con las reglas de este script)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_VERSION = "features-0.1.0"
EXPECTED_INPUT_SHA256 = (
    "fb1d9e60853743fff8cf6a2b17fb7588915d4213022c75b7311f33944a974f25"
)

INPUT_PARQUET = ROOT / "data" / "clean" / "ohlcv_clean.parquet"
FEATURES_DIR = ROOT / "data" / "features"
DAILY_PARQUET = FEATURES_DIR / "daily_features.parquet"
DAILY_CSV = FEATURES_DIR / "daily_features.csv"
WINDOWS_PARQUET = FEATURES_DIR / "windows_65.parquet"
MANIFEST_PATH = FEATURES_DIR / "features_manifest.json"
CHECKSUMS_PATH = FEATURES_DIR / "checksums.sha256"
README_PATH = FEATURES_DIR / "README.md"

T = 65
STRIDE = 1
CHANNEL_ORDER = ["log_return", "log_high_low_range", "log_volume"]
N_CHANNELS = len(CHANNEL_ORDER)
FLAT_LEN = T * N_CHANNELS  # 195

FEATURE_COLS = ["date", "ticker", *CHANNEL_ORDER]
OHLCV_COLS = ["date", "ticker", "Open", "High", "Low", "Close", "Volume"]

APPLIED_RULES = [
    "per_ticker_temporal_order",
    "drop_volume_eq_0_before_features_no_ln1",
    "log_return_ln_close_t_over_close_t_minus_1",
    "log_high_low_range_ln_high_over_low_or_0_if_equal",
    "log_volume_ln_volume",
    "drop_first_row_per_ticker_missing_log_return",
    "drop_nan_inf_in_three_features",
    "no_ffill_no_winsorize_no_standardize",
    "windows_T65_stride1_consecutive_feature_rows_single_ticker",
    "channel_order_log_return_log_high_low_range_log_volume",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_date(series: pd.Series) -> pd.Series:
    """Fecha calendario naive (sin timezone), normalizada a medianoche."""
    dt = pd.to_datetime(series)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt.dt.normalize()


def build_daily_features(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construye panel de features diario por ticker.

    Returns
    -------
    features : DataFrame con columnas FEATURE_COLS
    stats : dict con conteos de drops
    """
    df = ohlcv.copy()
    df["date"] = normalize_date(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    n_vol0 = int((df["Volume"] == 0).sum())
    vol0_rows = (
        df.loc[df["Volume"] == 0, ["date", "ticker", "Volume"]]
        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )
    df = df.loc[df["Volume"] != 0].copy()

    parts: list[pd.DataFrame] = []
    dropped_first_row = 0
    dropped_nan_inf = 0

    for ticker, g in df.groupby("ticker", sort=True):
        g = g.sort_values("date").copy()
        prev_close = g["Close"].shift(1)
        log_return = np.log(g["Close"] / prev_close)

        high = g["High"].to_numpy(dtype=np.float64)
        low = g["Low"].to_numpy(dtype=np.float64)
        equal_hl = high == low
        with np.errstate(divide="ignore", invalid="ignore"):
            log_hl = np.log(high / low)
        log_hl = np.where(equal_hl, 0.0, log_hl)

        log_volume = np.log(g["Volume"].to_numpy(dtype=np.float64))

        feat = pd.DataFrame(
            {
                "date": g["date"].to_numpy(),
                "ticker": ticker,
                "log_return": log_return.to_numpy(dtype=np.float64),
                "log_high_low_range": log_hl,
                "log_volume": log_volume,
            }
        )

        # Primera fila sin log_return (shift) → fuera
        missing_return = feat["log_return"].isna()
        dropped_first_row += int(missing_return.sum())
        feat = feat.loc[~missing_return].copy()

        # Drop NaN/inf en las 3 features
        vals = feat[CHANNEL_ORDER]
        bad = ~np.isfinite(vals.to_numpy(dtype=np.float64)).all(axis=1)
        dropped_nan_inf += int(bad.sum())
        feat = feat.loc[~bad].copy()

        parts.append(feat)

    features = pd.concat(parts, ignore_index=True)
    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    features = features[FEATURE_COLS]

    stats = {
        "dropped_volume_eq_0": n_vol0,
        "volume_eq_0_rows": vol0_rows,
        "dropped_first_row_or_missing_log_return": dropped_first_row,
        "dropped_nan_inf_features": dropped_nan_inf,
    }
    return features, stats


def build_windows(features: pd.DataFrame) -> pd.DataFrame:
    """Ventanas [T, 3] por ticker; stride=1 sobre filas consecutivas de features.

    Columna `features_flat`: lista de FLAT_LEN floats, row-major
    (t=0..T-1, canal en CHANNEL_ORDER). Reconstruir:
        np.asarray(row.features_flat, dtype=np.float64).reshape(T, 3)
    """
    records: list[dict] = []

    for ticker, g in features.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n < T:
            continue
        mat = g[CHANNEL_ORDER].to_numpy(dtype=np.float64)
        dates = g["date"].to_numpy()
        for start in range(0, n - T + 1, STRIDE):
            end = start + T  # exclusive
            flat = mat[start:end].reshape(-1).tolist()
            records.append(
                {
                    "ticker": ticker,
                    "window_start_date": dates[start],
                    "window_end_date": dates[end - 1],
                    "features_flat": flat,
                }
            )

    windows = pd.DataFrame.from_records(records)
    if windows.empty:
        windows = pd.DataFrame(
            columns=["ticker", "window_start_date", "window_end_date", "features_flat"]
        )
    else:
        windows = windows.sort_values(
            ["ticker", "window_start_date"]
        ).reset_index(drop=True)
    return windows


def write_readme() -> None:
    text = f"""# Features diarias + ventanas T={T} (`{DATA_VERSION}`)

Panel de features generativas diarias y ventanas de {T} días a partir de
`data/clean/ohlcv_clean.parquet`. **Solo features + ventanas.**

Sin splits train/val/test, sin calibración a NVDA, sin estandarización, sin
entrenamiento (VAE/GAN/Diffusion/Ridge). Sin re-limpieza ni escritura en
`data/clean/`.

> **Mantenimiento:** cualquier cambio en fórmulas o políticas debe actualizar
> **este README** y `scripts/build_features_windows.py` (y regenerar artefactos +
> `features_manifest.json`).

## Entrada (solo lectura)

| Campo | Valor |
|---|---|
| Ruta | `data/clean/ohlcv_clean.parquet` |
| SHA256 esperado | `{EXPECTED_INPUT_SHA256}` |
| `data/clean/` | **no se modifica** |

Si el SHA del parquet de entrada no coincide, el script **aborta**.

## Features (por ticker, orden temporal)

1. `log_return_t = ln(Close_t / Close_{{t-1}})`
2. `log_high_low_range_t = ln(High_t / Low_t)`; si `High == Low` → `0.0`
3. `log_volume_t = ln(Volume_t)`; filas con `Volume == 0` → **DROP** (no `ln(1)`)

Reglas:

- Independiente por ticker.
- Filas `Volume == 0` se eliminan **antes** de calcular features (el `log_return`
  siguiente usa el `Close` de la fila previa restante).
- Primera fila de cada ticker sin `log_return` → fuera.
- Drop NaN/inf en las 3 features.
- **No** ffill. **No** winsorize. **No** estandarizar (calibración NVDA = otro chat).
- Huecos vs unión de fechas: las ventanas usan **filas consecutivas del panel de
  features**, no días de calendario rellenados.

## Ventanas

| Parámetro | Valor |
|---|---|
| `T` | {T} |
| `stride` | {STRIDE} |
| Canales (orden fijo) | `{CHANNEL_ORDER}` → shape `[{T}, 3]` |
| Ámbito | un solo ticker por ventana |
| Metadatos | `ticker`, `window_start_date`, `window_end_date` |

## Ejecutar

```bash
cd taller_cristian
.venv/bin/python scripts/build_features_windows.py
```

## Artefactos

| Ruta | Contenido |
|---|---|
| `data/features/daily_features.parquet` | Features diarias |
| `data/features/daily_features.csv` | Misma tabla en CSV |
| `data/features/windows_65.parquet` | Ventanas + `features_flat` |
| `data/features/features_manifest.json` | Metadatos, conteos, SHA, políticas |
| `data/features/checksums.sha256` | SHA256 de parquet/csv de salida |
| `data/features/README.md` | Este documento |

`data_version`: `{DATA_VERSION}`

## Reconstruir tensor `[65, 3]`

En `windows_65.parquet`, la columna `features_flat` es una lista de
`{FLAT_LEN}` floats en orden row-major: para cada `t` en `0..{T - 1}`, los
canales `{CHANNEL_ORDER}`.

```python
import numpy as np
import pandas as pd

w = pd.read_parquet("data/features/windows_65.parquet")
row = w.iloc[0]
X = np.asarray(row["features_flat"], dtype=np.float64).reshape({T}, {N_CHANNELS})
# X.shape == ({T}, {N_CHANNELS}); X[:, 0] == log_return, etc.
assert list(X.shape) == [{T}, {N_CHANNELS}]
```

## Verificar checksums

```bash
sha256sum -c data/features/checksums.sha256
```

## Usar features diarias

```python
import pandas as pd
f = pd.read_parquet("data/features/daily_features.parquet")
# columnas: date, ticker, log_return, log_high_low_range, log_volume
```

Fuera de alcance: splits de experimento, calibración/estandarización a NVDA,
entrenamiento de modelos.
"""
    README_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    if not INPUT_PARQUET.is_file():
        print(f"FALLO: no existe entrada {INPUT_PARQUET}", file=sys.stderr)
        return 1

    input_sha = sha256_file(INPUT_PARQUET)
    if input_sha != EXPECTED_INPUT_SHA256:
        print(
            "BLOQUEADO: SHA256 del parquet de entrada no coincide.\n"
            f"  esperado: {EXPECTED_INPUT_SHA256}\n"
            f"  obtenido: {input_sha}\n"
            "No se re-limpia ni se modifica data/clean/. Abortando.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_parquet(INPUT_PARQUET)
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        print(f"FALLO: faltan columnas {missing}; got={list(df.columns)}", file=sys.stderr)
        return 1

    rows_in = int(len(df))
    features, drop_stats = build_daily_features(df)
    n_features = int(len(features))
    if n_features == 0:
        print("FALLO: panel de features vacío.", file=sys.stderr)
        return 1

    # Sanity checks (mínimos; EDA larga queda en 02c)
    n_inf = int((~np.isfinite(features[CHANNEL_ORDER].to_numpy())).sum())
    if n_inf != 0:
        print(f"FALLO: quedan no-finitos en features: {n_inf}", file=sys.stderr)
        return 1

    windows = build_windows(features)
    n_windows = int(len(windows))
    if n_windows == 0:
        print("FALLO: 0 ventanas generadas.", file=sys.stderr)
        return 1

    # Validar longitud flat
    lens = windows["features_flat"].map(len)
    if not (lens == FLAT_LEN).all():
        print("FALLO: features_flat con longitud distinta de 195.", file=sys.stderr)
        return 1

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(DAILY_PARQUET, index=False)
    features.to_csv(DAILY_CSV, index=False)
    windows.to_parquet(WINDOWS_PARQUET, index=False)
    write_readme()

    daily_parquet_sha = sha256_file(DAILY_PARQUET)
    daily_csv_sha = sha256_file(DAILY_CSV)
    windows_sha = sha256_file(WINDOWS_PARQUET)
    readme_sha = sha256_file(README_PATH)

    checksums = {
        str(DAILY_PARQUET.relative_to(ROOT)): daily_parquet_sha,
        str(DAILY_CSV.relative_to(ROOT)): daily_csv_sha,
        str(WINDOWS_PARQUET.relative_to(ROOT)): windows_sha,
        str(README_PATH.relative_to(ROOT)): readme_sha,
    }
    CHECKSUMS_PATH.write_text(
        "\n".join(f"{digest}  {rel}" for rel, digest in checksums.items()) + "\n",
        encoding="utf-8",
    )

    n_features_by_ticker = {
        t: int(n) for t, n in features.groupby("ticker").size().sort_index().items()
    }
    n_windows_by_ticker = {
        t: int(n) for t, n in windows.groupby("ticker").size().sort_index().items()
    }
    # Tickers sin ventanas (n < T) aparecen con 0
    for t in n_features_by_ticker:
        n_windows_by_ticker.setdefault(t, 0)

    date_min = str(features["date"].min().date())
    date_max = str(features["date"].max().date())

    manifest = {
        "data_version": DATA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT_PARQUET.relative_to(ROOT)),
        "input_sha256_verified": input_sha,
        "input_sha256_expected": EXPECTED_INPUT_SHA256,
        "formulas": {
            "log_return": "ln(Close_t / Close_{t-1})",
            "log_high_low_range": "ln(High_t / Low_t); 0.0 if High == Low",
            "log_volume": "ln(Volume_t); rows with Volume == 0 are dropped (no ln(1))",
        },
        "policies": {
            "volume_eq_0": "DROP row before feature computation (no ln(1))",
            "high_eq_low": "log_high_low_range = 0.0",
            "gaps_vs_union_calendar": (
                "no ffill; windows over consecutive rows of the features panel "
                "(not calendar-filled days)"
            ),
            "winsorize": "NO",
            "standardize_or_nvda_calibration": "NO (out of scope for this chat)",
            "heteroscedasticity_outliers": (
                "documented only; strong cross-ticker scale differences expected "
                "in log_volume / ranges; no winsorization in this version"
            ),
        },
        "rules_applied": APPLIED_RULES,
        "rows_in_clean": rows_in,
        "n_features_rows": n_features,
        "n_features_rows_by_ticker": n_features_by_ticker,
        "n_windows": n_windows,
        "n_windows_by_ticker": n_windows_by_ticker,
        "date_min_features": date_min,
        "date_max_features": date_max,
        "window_params": {
            "T": T,
            "stride": STRIDE,
            "channel_order": CHANNEL_ORDER,
            "tensor_shape": [T, N_CHANNELS],
            "features_flat_length": FLAT_LEN,
            "features_flat_layout": (
                "row-major: for t in 0..T-1, channels in channel_order; "
                f"reshape({T}, {N_CHANNELS})"
            ),
        },
        "dropped": drop_stats,
        "sanity_checks": {
            "n_nonfinite_in_features": n_inf,
            "n_windows_flat_len_ok": bool((lens == FLAT_LEN).all()),
        },
        "schema": {
            "daily_features_columns": FEATURE_COLS,
            "windows_columns": [
                "ticker",
                "window_start_date",
                "window_end_date",
                "features_flat",
            ],
        },
        "output_paths": {
            "daily_features_parquet": str(DAILY_PARQUET.relative_to(ROOT)),
            "daily_features_csv": str(DAILY_CSV.relative_to(ROOT)),
            "windows_parquet": str(WINDOWS_PARQUET.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "checksums": str(CHECKSUMS_PATH.relative_to(ROOT)),
            "readme": str(README_PATH.relative_to(ROOT)),
        },
        "checksums_sha256": checksums,
        "notes": [
            "Splits train/val/test y protocolo de experimento quedan fuera de este chat.",
            "Calibración/estandarización a NVDA queda fuera de este chat.",
            "No se modifica data/clean/.",
            (
                "1 fila Volume==0 dropeada (AMD 2015-01-02) si aplica; "
                f"contado dropped_volume_eq_0={drop_stats['dropped_volume_eq_0']}."
            ),
            "Heterocedasticidad/outliers entre tickers: solo documentados; sin winsorize.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # checksums no incluye el manifest a propósito (circular); documentado en README
    print("OK — features + ventanas")
    print(f"  input sha256:     {input_sha}")
    print(f"  rows_clean:       {rows_in}")
    print(f"  dropped Vol==0:   {drop_stats['dropped_volume_eq_0']}")
    print(f"  n_features_rows:  {n_features}")
    print(f"  n_windows:        {n_windows}")
    print(f"  date range:       {date_min} → {date_max}")
    print(f"  daily parquet:    {DAILY_PARQUET}")
    print(f"  windows parquet:  {WINDOWS_PARQUET}")
    print(f"  daily sha256:     {daily_parquet_sha}")
    print(f"  windows sha256:   {windows_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
