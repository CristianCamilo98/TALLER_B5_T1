#!/usr/bin/env python3
"""Limpieza/validación OHLCV (solo lectura de data/raw/).

Uso:
  .venv/bin/python scripts/clean_ohlcv.py

Entrada (no se modifica):
  data/raw/ohlcv_raw.parquet

Salidas:
  data/clean/ohlcv_clean.parquet
  data/clean/ohlcv_clean.csv
  data/clean/clean_manifest.json
  data/clean/checksums.sha256
  data/clean/README.md  (mantener alineado con las reglas de este script)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_VERSION = "clean-0.1.0"
EXPECTED_INPUT_SHA256 = (
    "6ecd4c929ecd3bdca32c646aec8210a7757b566843a90102f21bd86d2da036d6"
)

INPUT_PARQUET = ROOT / "data" / "raw" / "ohlcv_raw.parquet"
CLEAN_DIR = ROOT / "data" / "clean"
PANEL_PARQUET = CLEAN_DIR / "ohlcv_clean.parquet"
PANEL_CSV = CLEAN_DIR / "ohlcv_clean.csv"
MANIFEST_PATH = CLEAN_DIR / "clean_manifest.json"
CHECKSUMS_PATH = CLEAN_DIR / "checksums.sha256"

OHLC_COLS = ["Open", "High", "Low", "Close"]
OHLCV_COLS = [*OHLC_COLS, "Volume"]
OUT_COLS = ["date", "ticker", *OHLCV_COLS]

COVERAGE_THRESHOLD = 0.95

APPLIED_RULES = [
    "no_forward_fill_prices_or_volume",
    "drop_rows_with_nan_in_open_high_low_close",
    "drop_rows_breaking_ohlc_invariants",
    "normalize_date_naive_sort_by_date_ticker",
    "coverage_vs_union_calendar_after_nan_and_invariant_drops_threshold_0.95",
    "no_invented_rows_no_interpolation",
    "raw_dir_read_only",
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


def ohlc_invariant_mask(df: pd.DataFrame) -> pd.Series:
    """True = fila válida respecto a invariantes OHLC + Volume >= 0."""
    return (
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
        & (df["Volume"] >= 0)
    )


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
            "No se re-descarga ni se modifica data/raw/. Abortando.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_parquet(INPUT_PARQUET)
    missing = [c for c in OUT_COLS if c not in df.columns]
    if missing:
        print(f"FALLO: faltan columnas {missing}; got={list(df.columns)}", file=sys.stderr)
        return 1

    rows_in = int(len(df))

    # 4 (parcial): normalizar date antes de drops para contar/ordenar de forma estable.
    df = df.copy()
    df["date"] = normalize_date(df["date"])
    df["ticker"] = df["ticker"].astype(str)

    # 2. Drop NaN en OHLC (no Volume obligatorio aquí; Volume NaN cae vía invariante Volume>=0
    #    solo si se compara numéricamente — NaN en Volume no pasa Volume>=0 → se dropea).
    nan_mask = df[OHLC_COLS].isna().any(axis=1)
    dropped_nan_ohlc = int(nan_mask.sum())
    df = df.loc[~nan_mask].copy()

    # 3. Invariantes OHLC (+ Volume >= 0). Sin ffill / sin interpolar.
    valid = ohlc_invariant_mask(df)
    dropped_ohlc_invariant = int((~valid).sum())
    df = df.loc[valid].copy()

    # Calendario de cobertura = unión de fechas del panel tras drops NaN+invariantes
    # (antes de excluir tickers por cobertura). Documentado en manifest y README.
    calendar_dates = pd.Index(sorted(df["date"].unique()))
    n_calendar = int(len(calendar_dates))
    if n_calendar == 0:
        print("FALLO: panel vacío tras drops NaN/invariantes.", file=sys.stderr)
        return 1

    coverage_note = (
        "ticker_coverage = n_fechas_distintas_del_ticker / |unión de fechas del panel "
        "tras drops NaN en OHLC e invariantes OHLC|; umbral "
        f"{COVERAGE_THRESHOLD}. Sin inventar filas."
    )

    ticker_coverage: dict[str, float] = {}
    dropped_tickers_low_coverage: list[str] = []
    keep_tickers: list[str] = []

    for ticker in sorted(df["ticker"].unique()):
        n_dates = int(df.loc[df["ticker"] == ticker, "date"].nunique())
        cov = n_dates / n_calendar
        ticker_coverage[ticker] = round(cov, 6)
        if cov < COVERAGE_THRESHOLD:
            dropped_tickers_low_coverage.append(ticker)
        else:
            keep_tickers.append(ticker)

    df = df.loc[df["ticker"].isin(keep_tickers)].copy()

    # 4. Orden final (date, ticker); columnas canónicas.
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df = df[OUT_COLS]

    rows_out = int(len(df))
    if rows_out == 0:
        print("FALLO: panel limpio vacío tras filtros de cobertura.", file=sys.stderr)
        return 1

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_PARQUET, index=False)
    df.to_csv(PANEL_CSV, index=False)

    parquet_sha = sha256_file(PANEL_PARQUET)
    csv_sha = sha256_file(PANEL_CSV)

    checksums = {
        str(PANEL_PARQUET.relative_to(ROOT)): parquet_sha,
        str(PANEL_CSV.relative_to(ROOT)): csv_sha,
    }
    CHECKSUMS_PATH.write_text(
        "\n".join(f"{digest}  {rel}" for rel, digest in checksums.items()) + "\n",
        encoding="utf-8",
    )

    date_min = str(df["date"].min().date())
    date_max = str(df["date"].max().date())
    tickers_out = sorted(df["ticker"].unique().tolist())

    manifest = {
        "data_version": DATA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT_PARQUET.relative_to(ROOT)),
        "input_sha256_verified": input_sha,
        "input_sha256_expected": EXPECTED_INPUT_SHA256,
        "rules_applied": APPLIED_RULES,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "coverage_calendar_definition": coverage_note,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dropped_nan_ohlc": dropped_nan_ohlc,
        "dropped_ohlc_invariant": dropped_ohlc_invariant,
        "ticker_coverage": ticker_coverage,
        "dropped_tickers_low_coverage": dropped_tickers_low_coverage,
        "calendar_n_dates": n_calendar,
        "date_min": date_min,
        "date_max": date_max,
        "tickers_out": tickers_out,
        "schema": {
            "columns": OUT_COLS,
            "format": "long (una fila = un ticker × un día de sesión)",
        },
        "output_paths": {
            "panel_parquet": str(PANEL_PARQUET.relative_to(ROOT)),
            "panel_csv": str(PANEL_CSV.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "checksums": str(CHECKSUMS_PATH.relative_to(ROOT)),
            "readme": str((CLEAN_DIR / "README.md").relative_to(ROOT)),
        },
        "checksums_sha256": checksums,
        "notes": [
            "Solo limpieza/validación OHLCV. Features, log_return, ventanas y splits quedan fuera.",
            "No forward-fill, no interpolación, no filas inventadas.",
            "data/raw/ se lee solo; no se escribe.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("OK — limpieza OHLCV")
    print(f"  input sha256:  {input_sha}")
    print(f"  rows_in/out:   {rows_in} → {rows_out}")
    print(f"  dropped_nan:   {dropped_nan_ohlc}")
    print(f"  dropped_inv:   {dropped_ohlc_invariant}")
    print(f"  low_coverage:  {dropped_tickers_low_coverage}")
    print(f"  tickers_out:   {tickers_out}")
    print(f"  parquet:       {PANEL_PARQUET}")
    print(f"  sha256:        {parquet_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
