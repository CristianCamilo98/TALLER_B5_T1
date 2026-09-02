#!/usr/bin/env python3
"""Copy a local canonical data snapshot into this repository's data directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common_protocol import ROOT, sha256_file

REQUIRED_RELATIVE_FILES = (
    "raw/ohlcv_raw.parquet",
    "raw/ohlcv_raw.csv",
    "raw/download_manifest.json",
    "clean/ohlcv_clean.parquet",
    "clean/ohlcv_clean.csv",
    "clean/clean_manifest.json",
    "clean/quality_report.csv",
    "splits/daily_split_assignments.parquet",
    "splits/split_manifest.json",
    "splits/daily_split_report.csv",
    "features/daily_features_by_split.parquet",
    "features/features_manifest.json",
    "features/test_index.parquet",
    "features/window_counts.csv",
    "features/windows/donor_train.parquet",
    "features/windows/donor_validation.parquet",
    "features/windows/nvda_visible.parquet",
    "features/windows/nvda_full_history.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data-root",
        type=Path,
        required=True,
        help="Folder containing raw/, clean/, splits/, and features/.",
    )
    parser.add_argument(
        "--destination-data-root",
        type=Path,
        default=ROOT / "data",
        help="Repository data folder to update.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _child_dir_names(path: Path) -> set[str]:
    return {child.name for child in path.iterdir() if child.is_dir()}


def resolve_source(path: Path) -> Path:
    candidate = path.resolve()
    required_dirs = {"raw", "clean", "splits", "features"}
    if candidate.is_dir() and required_dirs.issubset(_child_dir_names(candidate)):
        return candidate
    nested = candidate / "data"
    if nested.is_dir() and required_dirs.issubset(_child_dir_names(nested)):
        return nested
    raise FileNotFoundError(
        f"{path} does not look like a canonical data root with raw/ clean/ splits/ features/"
    )


def validate_source(source: Path) -> None:
    missing = [rel for rel in REQUIRED_RELATIVE_FILES if not (source / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Source data snapshot is missing required files: {missing}")


def copy_snapshot(source: Path, destination: Path, *, dry_run: bool) -> list[Path]:
    copied: list[Path] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        copied.append(target)
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return copied


def main() -> int:
    args = parse_args()
    try:
        source = resolve_source(args.source_data_root)
        destination = args.destination_data_root.resolve()
        validate_source(source)
        copied = copy_snapshot(source, destination, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"FALLO importacion de datos: {exc}", file=sys.stderr)
        return 1

    donor = destination / "features" / "windows" / "donor_train.parquet"
    print(f"source_data_root: {source}")
    print(f"destination_data_root: {destination}")
    print(f"files_considered: {len(copied)}")
    print(f"dry_run: {args.dry_run}")
    if not args.dry_run:
        print(f"donor_train_sha256: {sha256_file(donor)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
