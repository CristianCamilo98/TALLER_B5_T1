#!/usr/bin/env python3
"""Descarga OHLCV diario crudo vía yfinance (sin limpieza ni features).

Uso:
  .venv/bin/python scripts/download_ohlcv_raw.py

Salidas:
  data/raw/ohlcv_raw.parquet
  data/raw/ohlcv_raw.csv
  data/raw/by_ticker/<TICKER>.parquet
  data/raw/download_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "data_contract.yaml"

DATA_VERSION = "raw-0.1.0"
RAW_DIR = ROOT / "data" / "raw"
BY_TICKER_DIR = RAW_DIR / "by_ticker"
PANEL_PARQUET = RAW_DIR / "ohlcv_raw.parquet"
PANEL_CSV = RAW_DIR / "ohlcv_raw.csv"
MANIFEST_PATH = RAW_DIR / "download_manifest.json"

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
OUT_COLS = ["date", "ticker", *OHLCV_COLS]


def load_contract() -> dict:
    with CONTRACT_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def null_counts(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    return {c: int(df[c].isna().sum()) for c in cols}


def download_ticker(ticker: str, start: str, end_inclusive: str, auto_adjust: bool) -> pd.DataFrame:
    """Descarga un ticker. yfinance trata `end` como exclusivo → pedimos end+1 día."""
    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"Sin datos (vacío) para ticker={ticker!r}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    missing = [c for c in OHLCV_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"{ticker}: faltan columnas {missing}; got={list(raw.columns)}")

    out = raw[OHLCV_COLS].copy()
    # Índice a fechas naive (sin timezone) — no se inventan ni rellenan filas.
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx
    out["ticker"] = ticker
    out = out.reset_index(names="date")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out[OUT_COLS]
    if out.empty:
        raise RuntimeError(f"Sin filas tras normalizar columnas para ticker={ticker!r}")
    return out


def main() -> int:
    try:
        contract = load_contract()
    except Exception as exc:  # noqa: BLE001
        print(f"Error leyendo contrato: {exc}", file=sys.stderr)
        return 1

    start = contract["calendar"]["start"]
    end_inclusive = contract["calendar"]["end"]
    auto_adjust = bool(contract.get("auto_adjust", True))
    source = contract.get("source", "yfinance")
    tickers = [contract["universe"]["target"], *contract["universe"]["donors"]]

    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    BY_TICKER_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    per_ticker: dict[str, dict] = {}
    by_ticker_paths: dict[str, str] = {}

    for t in tickers:
        print(f"descargando {t} ...")
        try:
            part = download_ticker(t, start, end_inclusive, auto_adjust)
        except Exception as exc:  # noqa: BLE001
            print(f"FALLO explícito en {t}: {exc}", file=sys.stderr)
            return 1

        ticker_path = BY_TICKER_DIR / f"{t}.parquet"
        part.to_parquet(ticker_path, index=False)
        by_ticker_paths[t] = str(ticker_path.relative_to(ROOT))

        per_ticker[t] = {
            "n_rows": int(len(part)),
            "date_min": str(part["date"].min().date()),
            "date_max": str(part["date"].max().date()),
            "n_nulls": null_counts(part, OHLCV_COLS),
            "path": by_ticker_paths[t],
        }
        frames.append(part)
        print(
            f"  {t}: rows={per_ticker[t]['n_rows']} "
            f"{per_ticker[t]['date_min']} → {per_ticker[t]['date_max']}"
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel.to_parquet(PANEL_PARQUET, index=False)
    panel.to_csv(PANEL_CSV, index=False)

    checksums = {
        str(PANEL_PARQUET.relative_to(ROOT)): sha256_file(PANEL_PARQUET),
        str(PANEL_CSV.relative_to(ROOT)): sha256_file(PANEL_CSV),
    }
    for rel in by_ticker_paths.values():
        checksums[rel] = sha256_file(ROOT / rel)

    manifest = {
        "data_version": DATA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "contract_path": str(CONTRACT_PATH.relative_to(ROOT)),
        "tickers_requested": tickers,
        "yfinance_params": {
            "auto_adjust": auto_adjust,
            "start": start,
            "end_inclusive_requested": end_inclusive,
            "end_passed_to_yfinance": end_exclusive,
            "end_semantics": (
                "yfinance trata `end` como exclusivo; para incluir end_inclusive_requested "
                f"se pasa end={end_exclusive} (end_inclusive + 1 día calendario)."
            ),
        },
        "per_ticker": per_ticker,
        "panel": {
            "n_rows": int(len(panel)),
            "date_min": str(panel["date"].min().date()),
            "date_max": str(panel["date"].max().date()),
            "n_nulls": null_counts(panel, OHLCV_COLS),
        },
        "schema": {
            "columns": OUT_COLS,
            "format": "long (una fila = un ticker × un día de sesión)",
        },
        "output_paths": {
            "panel_parquet": str(PANEL_PARQUET.relative_to(ROOT)),
            "panel_csv": str(PANEL_CSV.relative_to(ROOT)),
            "by_ticker_dir": str(BY_TICKER_DIR.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        },
        "checksums_sha256": checksums,
        "notes": [
            "Solo descarga cruda: sin drop de NaN, sin invariantes OHLC, sin ffill, sin features.",
            "Nombres de columnas normalizados a Open/High/Low/Close/Volume; fechas naive (UTC-unaware).",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nOK — descarga raw")
    print(f"  panel parquet: {PANEL_PARQUET}")
    print(f"  panel csv:     {PANEL_CSV}")
    print(f"  by_ticker:     {BY_TICKER_DIR}/")
    print(f"  manifest:      {MANIFEST_PATH}")
    print(f"  rows={len(panel)} tickers={len(tickers)}")
    print(f"  sha256 parquet: {checksums[str(PANEL_PARQUET.relative_to(ROOT))]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
