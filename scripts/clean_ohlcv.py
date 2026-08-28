#!/usr/bin/env python3
"""Limpia OHLCV usando tolerancia explícita y el snapshot raw presente."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from common_protocol import (
    ROOT,
    load_experiment_config,
    normalize_date,
    root_path,
    sha256_file,
    verify_snapshot,
    write_checksums,
    write_json,
)

DATA_VERSION = "clean-0.2.0"
OHLC_COLS = ["Open", "High", "Low", "Close"]
OHLCV_COLS = [*OHLC_COLS, "Volume"]
OUT_COLS = ["date", "ticker", *OHLCV_COLS]


def ohlc_tolerance(df: pd.DataFrame, absolute: float, relative: float) -> pd.Series:
    scale = df[OHLC_COLS].abs().max(axis=1)
    return absolute + relative * scale


def ohlc_invariant_mask(
    df: pd.DataFrame,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> pd.Series:
    tolerance = ohlc_tolerance(df, absolute_tolerance, relative_tolerance)
    return (
        (df["High"] + tolerance >= df["Low"])
        & (df["High"] + tolerance >= df["Open"])
        & (df["High"] + tolerance >= df["Close"])
        & (df["Low"] - tolerance <= df["Open"])
        & (df["Low"] - tolerance <= df["Close"])
        & (df["Volume"] >= 0)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config, config_path = load_experiment_config(args.config)
        input_path = root_path(config, "raw_panel")
        raw_manifest = root_path(config, "raw_manifest")
        input_sha = verify_snapshot(input_path, raw_manifest)
        output_parquet = root_path(config, "clean_panel")
        output_csv = root_path(config, "clean_csv")
        manifest_path = root_path(config, "clean_manifest")
        quality_report_path = root_path(config, "clean_quality_report")
        checksums_path = output_parquet.parent / "checksums.sha256"

        cleaning = config["cleaning"]
        absolute = float(cleaning["ohlc_tolerance"]["absolute"])
        relative = float(cleaning["ohlc_tolerance"]["relative"])
        coverage_threshold = float(cleaning["min_ticker_coverage"])

        df = pd.read_parquet(input_path)
        missing = [column for column in OUT_COLS if column not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas raw: {missing}")
        rows_in = int(len(df))
        df = df.copy()
        df["date"] = normalize_date(df["date"])
        df["ticker"] = df["ticker"].astype(str)
        duplicates = int(df.duplicated(["date", "ticker"]).sum())
        if duplicates:
            raise ValueError(f"Duplicados raw (date,ticker): {duplicates}")

        nan_mask = df[OHLC_COLS].isna().any(axis=1) | df["Volume"].isna()
        dropped_nan = int(nan_mask.sum())
        after_nan = df.loc[~nan_mask].copy()
        strict_valid = ohlc_invariant_mask(
            after_nan, absolute_tolerance=0.0, relative_tolerance=0.0
        )
        tolerant_valid = ohlc_invariant_mask(
            after_nan,
            absolute_tolerance=absolute,
            relative_tolerance=relative,
        )
        rescued_by_tolerance = int((~strict_valid & tolerant_valid).sum())
        dropped_invariant = int((~tolerant_valid).sum())
        filtered = after_nan.loc[tolerant_valid].copy()

        calendar = pd.Index(sorted(filtered["date"].unique()))
        if calendar.empty:
            raise ValueError("Panel vacío después de validación")
        coverage = {
            ticker: float(group["date"].nunique() / len(calendar))
            for ticker, group in filtered.groupby("ticker")
        }
        dropped_tickers = sorted(
            ticker for ticker, value in coverage.items() if value < coverage_threshold
        )
        clean = filtered.loc[~filtered["ticker"].isin(dropped_tickers), OUT_COLS]
        clean = clean.sort_values(["date", "ticker"]).reset_index(drop=True)
        if clean.empty:
            raise ValueError("Panel limpio vacío")

        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(output_parquet, index=False)
        clean.to_csv(output_csv, index=False)

        quality_rows = [
            {"check": "rows_in", "value": rows_in, "status": "INFO"},
            {"check": "dropped_nan", "value": dropped_nan, "status": "PASS"},
            {"check": "strict_float_edge_rows", "value": rescued_by_tolerance, "status": "PASS"},
            {"check": "dropped_material_invariant", "value": dropped_invariant, "status": "PASS"},
            {"check": "duplicate_date_ticker", "value": duplicates, "status": "PASS"},
            {"check": "rows_out", "value": int(len(clean)), "status": "INFO"},
            {"check": "tickers_out", "value": int(clean["ticker"].nunique()), "status": "PASS"},
        ]
        pd.DataFrame(quality_rows).to_csv(quality_report_path, index=False)
        checksums = write_checksums(
            checksums_path, [output_parquet, output_csv, quality_report_path]
        )
        manifest = {
            "data_version": DATA_VERSION,
            "protocol_version": config["experiment"]["protocol_version"],
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": sha256_file(config_path),
            "input_path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "input_snapshot_sha256_verified": input_sha,
            "checksum_policy": "Input verified against its current manifest, not a hardcoded hash.",
            "ohlc_tolerance": {"absolute": absolute, "relative": relative},
            "rows_in": rows_in,
            "rows_out": int(len(clean)),
            "dropped_nan": dropped_nan,
            "strict_float_edge_rows_rescued": rescued_by_tolerance,
            "dropped_material_ohlc_invariant": dropped_invariant,
            "ticker_coverage": {key: round(value, 9) for key, value in sorted(coverage.items())},
            "dropped_tickers_low_coverage": dropped_tickers,
            "date_min": str(clean["date"].min().date()),
            "date_max": str(clean["date"].max().date()),
            "tickers_out": sorted(clean["ticker"].unique().tolist()),
            "schema": {"columns": OUT_COLS, "format": "long"},
            "checksums_sha256": checksums,
        }
        write_json(manifest_path, manifest)
    except Exception as exc:  # noqa: BLE001
        print(f"FALLO limpieza: {exc}", file=sys.stderr)
        return 1

    print(
        f"OK clean: {rows_in}->{len(clean)}; "
        f"float_edges_rescued={rescued_by_tolerance}; material_drops={dropped_invariant}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
