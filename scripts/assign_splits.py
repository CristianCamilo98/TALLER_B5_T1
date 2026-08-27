#!/usr/bin/env python3
"""Asigna roles temporales a filas diarias antes de construir cualquier ventana."""

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

DATA_VERSION = "daily-splits-1.0.0"
SPLIT_LABELS = (
    "donor_train",
    "donor_validation",
    "nvda_hidden",
    "nvda_visible",
    "nvda_test",
    "unused",
)
OUT_COLS = ["date", "ticker", "daily_row", "split"]


def assign_daily_labels(df: pd.DataFrame, config: dict) -> pd.Series:
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    target = config["universe"]["target"]
    donors = set(config["universe"]["donors"])
    ticker = df["ticker"]
    date = df["date"]
    is_donor = ticker.isin(donors)
    is_target = ticker.eq(target)

    masks = {
        "donor_train": is_donor & date.between(
            dates["donor_train_start"], dates["donor_train_end"]
        ),
        "donor_validation": is_donor & date.between(
            dates["donor_validation_start"], dates["donor_validation_end"]
        ),
        "nvda_hidden": is_target & date.between(
            dates["target_hidden_start"], dates["target_hidden_end"]
        ),
        "nvda_visible": is_target & date.between(
            dates["target_visible_start"], dates["target_visible_end"]
        ),
        "nvda_test": is_target & date.between(
            dates["target_test_start"], dates["target_test_end"]
        ),
    }
    overlap_count = sum(mask.astype(int) for mask in masks.values())
    if bool((overlap_count > 1).any()):
        raise ValueError("Las reglas de split diario se solapan")

    labels = pd.Series("unused", index=df.index, dtype="object")
    for label, mask in masks.items():
        labels = labels.mask(mask, label)
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config, config_path = load_experiment_config(args.config)
        clean_path = root_path(config, "clean_panel")
        clean_manifest = root_path(config, "clean_manifest")
        input_sha = verify_snapshot(clean_path, clean_manifest)
        output_path = root_path(config, "daily_splits")
        manifest_path = root_path(config, "split_manifest")
        report_path = root_path(config, "split_report")
        checksums_path = output_path.parent / "checksums.sha256"

        clean = pd.read_parquet(clean_path)
        required = {"date", "ticker"}
        if not required.issubset(clean.columns):
            raise ValueError(f"Clean sin columnas {sorted(required - set(clean.columns))}")
        clean = clean.copy()
        clean["date"] = normalize_date(clean["date"])
        clean["ticker"] = clean["ticker"].astype(str)
        if clean.duplicated(["date", "ticker"]).any():
            raise ValueError("Clean contiene duplicados (date,ticker)")

        assignments = clean[["date", "ticker"]].copy()
        assignments["daily_row"] = range(len(assignments))
        assignments["split"] = assign_daily_labels(assignments, config)
        assignments = assignments[OUT_COLS]
        if not set(assignments["split"]).issubset(SPLIT_LABELS):
            raise ValueError("Etiquetas de split inesperadas")

        report = (
            assignments.groupby("split", sort=False)
            .agg(n_rows=("date", "size"), date_min=("date", "min"), date_max=("date", "max"), n_tickers=("ticker", "nunique"))
            .reindex(SPLIT_LABELS)
            .reset_index()
        )
        report["date_min"] = report["date_min"].dt.strftime("%Y-%m-%d")
        report["date_max"] = report["date_max"].dt.strftime("%Y-%m-%d")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_parquet(output_path, index=False)
        report.to_csv(report_path, index=False)
        checksums = write_checksums(checksums_path, [output_path, report_path])
        counts = {
            row.split: {
                "n_rows": int(row.n_rows),
                "date_min": row.date_min,
                "date_max": row.date_max,
                "n_tickers": int(row.n_tickers),
            }
            for row in report.itertuples()
        }
        manifest = {
            "data_version": DATA_VERSION,
            "protocol_version": config["experiment"]["protocol_version"],
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": sha256_file(config_path),
            "input_path": str(clean_path.relative_to(ROOT)).replace("\\", "/"),
            "input_snapshot_sha256_verified": input_sha,
            "assignment_order": "daily rows first; windows are not created by this stage",
            "date_rules": config["dates"],
            "target": config["universe"]["target"],
            "donors": config["universe"]["donors"],
            "counts": counts,
            "assertions": {
                "partition_exhaustive": int(len(assignments)) == int(len(clean)),
                "natural_key_unique": not bool(assignments.duplicated(["date", "ticker"]).any()),
                "mutually_exclusive": True,
                "no_windows_created_before_split": True,
            },
            "schema": {"columns": OUT_COLS, "split_labels": list(SPLIT_LABELS)},
            "checksums_sha256": checksums,
        }
        write_json(manifest_path, manifest)
    except Exception as exc:  # noqa: BLE001
        print(f"FALLO splits diarios: {exc}", file=sys.stderr)
        return 1

    print("OK splits diarios")
    for row in report.itertuples():
        print(f"  {row.split}: rows={int(row.n_rows)} {row.date_min}->{row.date_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
