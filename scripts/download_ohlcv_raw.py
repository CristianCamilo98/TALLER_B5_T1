#!/usr/bin/env python3
"""Descarga el snapshot OHLCV raw definido en configs/experiment.yaml."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from common_protocol import ROOT, load_experiment_config, root_path, sha256_file, write_json

DATA_VERSION = "raw-0.2.0"
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
OUT_COLS = ["date", "ticker", *OHLCV_COLS]


def download_ticker(
    ticker: str,
    *,
    start: str,
    end_inclusive: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"Sin datos para ticker={ticker!r}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    missing = [column for column in OHLCV_COLS if column not in raw.columns]
    if missing:
        raise RuntimeError(f"{ticker}: faltan columnas {missing}")

    out = raw[OHLCV_COLS].copy()
    index = pd.to_datetime(out.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    out.index = index
    out["ticker"] = ticker
    out = out.reset_index(names="date")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out[OUT_COLS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--reuse-snapshot",
        action="store_true",
        help="Regenera solo el manifest a partir del snapshot raw local; no usa red.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config, config_path = load_experiment_config(args.config)
        raw_panel = root_path(config, "raw_panel")
        raw_csv = root_path(config, "raw_csv")
        manifest_path = root_path(config, "raw_manifest")
        raw_dir = raw_panel.parent
        by_ticker_dir = raw_dir / "by_ticker"

        source = config["source"]
        start = source["download_start"]
        end_inclusive = source["download_end_inclusive"]
        interval = source["interval"]
        auto_adjust = bool(source["auto_adjust"])
        tickers = [config["universe"]["target"], *config["universe"]["donors"]]
        end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        raw_dir.mkdir(parents=True, exist_ok=True)
        by_ticker_dir.mkdir(parents=True, exist_ok=True)
        per_ticker: dict[str, dict] = {}
        output_files = [by_ticker_dir / f"{ticker}.parquet" for ticker in tickers]
        if args.reuse_snapshot:
            if not raw_panel.is_file() or not raw_csv.is_file():
                raise FileNotFoundError("No existe el snapshot raw local que se pidió reutilizar")
            panel = pd.read_parquet(raw_panel)
            acquisition_mode = "reuse_existing_snapshot_no_network"
        else:
            frames: list[pd.DataFrame] = []
            for ticker in tickers:
                print(f"descargando {ticker} ...")
                part = download_ticker(
                    ticker,
                    start=start,
                    end_inclusive=end_inclusive,
                    interval=interval,
                    auto_adjust=auto_adjust,
                )
                ticker_path = by_ticker_dir / f"{ticker}.parquet"
                part.to_parquet(ticker_path, index=False)
                frames.append(part)
            panel = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
            panel = panel.reset_index(drop=True)
            panel.to_parquet(raw_panel, index=False)
            panel.to_csv(raw_csv, index=False)
            acquisition_mode = "downloaded_from_yfinance"

        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        panel["ticker"] = panel["ticker"].astype(str)
        for ticker in tickers:
            ticker_path = by_ticker_dir / f"{ticker}.parquet"
            if not ticker_path.is_file():
                raise FileNotFoundError(f"Falta snapshot por ticker: {ticker_path}")
            part = panel.loc[panel["ticker"].eq(ticker)]
            if part.empty:
                raise RuntimeError(f"Snapshot local sin ticker={ticker}")
            per_ticker[ticker] = {
                "n_rows": int(len(part)),
                "date_min": str(part["date"].min().date()),
                "date_max": str(part["date"].max().date()),
                "n_nulls": {column: int(part[column].isna().sum()) for column in OHLCV_COLS},
                "path": str(ticker_path.relative_to(ROOT)).replace("\\", "/"),
            }
        if panel.duplicated(["date", "ticker"]).any():
            raise RuntimeError("La descarga contiene duplicados (date, ticker)")
        output_files.extend([raw_panel, raw_csv])
        checksums = {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in output_files
        }
        manifest = {
            "data_version": DATA_VERSION,
            "protocol_version": config["experiment"]["protocol_version"],
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": sha256_file(config_path),
            "source": source["provider"],
            "acquisition_mode": acquisition_mode,
            "tickers_requested": tickers,
            "yfinance_params": {
                "start": start,
                "end_inclusive_requested": end_inclusive,
                "end_passed_to_yfinance": end_exclusive,
                "interval": interval,
                "auto_adjust": auto_adjust,
            },
            "per_ticker": per_ticker,
            "panel": {
                "n_rows": int(len(panel)),
                "date_min": str(panel["date"].min().date()),
                "date_max": str(panel["date"].max().date()),
                "n_duplicates_date_ticker": int(panel.duplicated(["date", "ticker"]).sum()),
                "n_nulls": {column: int(panel[column].isna().sum()) for column in OHLCV_COLS},
            },
            "schema": {"columns": OUT_COLS, "format": "long"},
            "checksums_sha256": checksums,
            "checksum_policy": (
                "Hashes describe this observed snapshot. Downstream stages verify against this "
                "manifest; no future yfinance download is required to reproduce the same hash."
            ),
        }
        write_json(manifest_path, manifest)
    except Exception as exc:  # noqa: BLE001
        print(f"FALLO descarga raw: {exc}", file=sys.stderr)
        return 1

    panel_rel = str(raw_panel.relative_to(ROOT)).replace("\\", "/")
    print(f"OK raw: rows={len(panel)} snapshot={checksums[panel_rel]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
