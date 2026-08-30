#!/usr/bin/env python3
"""Construye features y ventanas únicamente después del split diario certificado."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

DATA_VERSION = "features-windows-1.0.0"
FEATURE_BLOCKS = (
    "donor_train",
    "donor_validation",
    "nvda_visible",
    "nvda_full_history",
    "nvda_test_source",
)
WINDOW_SPLITS = (
    "donor_train",
    "donor_validation",
    "nvda_visible",
    "nvda_full_history",
)


def build_daily_features(ohlcv: pd.DataFrame, block: str) -> pd.DataFrame:
    """Calcula las tres features dentro de cada bloque, sin mirar la fila anterior al corte."""
    frames: list[pd.DataFrame] = []
    for ticker, group in ohlcv.groupby("ticker", sort=True):
        group = group.sort_values("date").copy()
        if (group[["High", "Low", "Close"]] <= 0).any().any():
            raise ValueError(f"{block}/{ticker}: precios no positivos")
        if (group["Volume"] < 0).any():
            raise ValueError(f"{block}/{ticker}: volumen negativo")
        frame = pd.DataFrame(
            {
                "feature_block": block,
                "date": group["date"].to_numpy(),
                "ticker": ticker,
                "log_return": np.log(
                    group["Close"] / group["Close"].shift(1)
                ).to_numpy(dtype=np.float64),
                "log_high_low_range": np.log(
                    group["High"].to_numpy(dtype=np.float64)
                    / group["Low"].to_numpy(dtype=np.float64)
                ),
                "log1p_volume": np.log1p(group["Volume"].to_numpy(dtype=np.float64)),
            }
        )
        frame = frame.dropna(subset=["log_return"]).copy()
        values = frame[["log_return", "log_high_low_range", "log1p_volume"]].to_numpy()
        frame = frame.loc[np.isfinite(values).all(axis=1)].copy()
        frames.append(frame)
    if not frames:
        raise ValueError(f"Bloque sin features: {block}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_windows(
    features: pd.DataFrame,
    *,
    split: str,
    stride: int,
    length: int,
    channels: list[str],
) -> pd.DataFrame:
    records: list[dict] = []
    for ticker, group in features.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        matrix = group[channels].to_numpy(dtype=np.float64)
        dates = group["date"].to_numpy()
        for start in range(0, len(group) - length + 1, stride):
            stop = start + length
            records.append(
                {
                    "split": split,
                    "ticker": ticker,
                    "window_start_date": dates[start],
                    "window_end_date": dates[stop - 1],
                    "features_flat": matrix[start:stop].reshape(-1).tolist(),
                }
            )
    if not records:
        raise ValueError(f"No se generaron ventanas para {split}")
    return pd.DataFrame.from_records(records).sort_values(
        ["ticker", "window_start_date"]
    ).reset_index(drop=True)


def build_test_index(
    features: pd.DataFrame,
    *,
    target: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    context: int,
    horizon: int,
    stride: int,
    channels: list[str],
) -> pd.DataFrame:
    group = features.loc[features["ticker"].eq(target)].sort_values("date").reset_index(drop=True)
    length = context + horizon
    valid_starts: list[int] = []
    for start in range(0, len(group) - length + 1):
        target_dates = group["date"].iloc[start + context : start + length]
        if len(target_dates) == horizon and target_dates.min() >= test_start and target_dates.max() <= test_end:
            valid_starts.append(start)
    selected = valid_starts[::stride]
    records: list[dict] = []
    for test_row, start in enumerate(selected):
        stop = start + length
        block = group.iloc[start:stop]
        context_dates = block["date"].iloc[:context]
        target_dates = block["date"].iloc[context:]
        records.append(
            {
                "test_row": test_row,
                "ticker": target,
                "window_start_date": block["date"].iloc[0],
                "window_end_date": block["date"].iloc[-1],
                "context_start_date": context_dates.iloc[0],
                "context_end_date": context_dates.iloc[-1],
                "target_start_date": target_dates.iloc[0],
                "target_end_date": target_dates.iloc[-1],
                "context_dates": [value.strftime("%Y-%m-%d") for value in context_dates],
                "target_dates": [value.strftime("%Y-%m-%d") for value in target_dates],
                "features_flat": block[channels].to_numpy(dtype=np.float64).reshape(-1).tolist(),
            }
        )
    if not records:
        raise ValueError("No se generó test_index")
    return pd.DataFrame.from_records(records)


def window_path(windows_dir: Path, split: str) -> Path:
    return windows_dir / f"{split}.parquet"


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
        split_path = root_path(config, "daily_splits")
        split_manifest = root_path(config, "split_manifest")
        clean_sha = verify_snapshot(clean_path, clean_manifest)
        split_sha = verify_snapshot(split_path, split_manifest)

        daily_features_path = root_path(config, "daily_features")
        windows_dir = root_path(config, "windows_dir")
        test_index_path = root_path(config, "test_index")
        manifest_path = root_path(config, "features_manifest")
        report_path = root_path(config, "window_report")
        checksums_path = daily_features_path.parent / "checksums.sha256"

        clean = pd.read_parquet(clean_path)
        assignments = pd.read_parquet(split_path)
        clean["date"] = normalize_date(clean["date"])
        assignments["date"] = normalize_date(assignments["date"])
        joined = clean.merge(
            assignments[["date", "ticker", "split"]],
            on=["date", "ticker"],
            how="inner",
            validate="1:1",
        )
        if len(joined) != len(clean):
            raise ValueError("El split diario no cubre el panel clean 1:1")

        dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
        target = config["universe"]["target"]
        blocks = {
            "donor_train": joined.loc[joined["split"].eq("donor_train")],
            "donor_validation": joined.loc[joined["split"].eq("donor_validation")],
            "nvda_visible": joined.loc[joined["split"].eq("nvda_visible")],
            "nvda_full_history": joined.loc[
                joined["ticker"].eq(target)
                & joined["date"].between(dates["target_hidden_start"], dates["target_visible_end"])
            ],
            "nvda_test_source": joined.loc[
                joined["ticker"].eq(target)
                & joined["date"].between(dates["target_visible_start"], dates["target_test_end"])
            ],
        }
        daily_by_block = {
            block: build_daily_features(frame, block) for block, frame in blocks.items()
        }
        daily_features = pd.concat(daily_by_block.values(), ignore_index=True)
        daily_features["feature_row"] = range(len(daily_features))

        channels = list(config["features"]["channels"])
        window_cfg = config["windows"]
        length = int(window_cfg["length"])
        stride_map = {
            "donor_train": int(window_cfg["strides"]["donor_train"]),
            "donor_validation": int(window_cfg["strides"]["donor_validation"]),
            "nvda_visible": int(window_cfg["strides"]["target_visible"]),
            "nvda_full_history": int(window_cfg["strides"]["target_full_history"]),
        }
        windows = {
            split: build_windows(
                daily_by_block[split],
                split=split,
                stride=stride_map[split],
                length=length,
                channels=channels,
            )
            for split in WINDOW_SPLITS
        }
        test_index = build_test_index(
            daily_by_block["nvda_test_source"],
            target=target,
            test_start=dates["target_test_start"],
            test_end=dates["target_test_end"],
            context=int(window_cfg["context"]),
            horizon=int(window_cfg["horizon"]),
            stride=int(window_cfg["strides"]["target_test"]),
            channels=channels,
        )

        boundary_limits = {
            "donor_train": (dates["donor_train_start"], dates["donor_train_end"]),
            "donor_validation": (
                dates["donor_validation_start"],
                dates["donor_validation_end"],
            ),
            "nvda_visible": (dates["target_visible_start"], dates["target_visible_end"]),
            "nvda_full_history": (dates["target_hidden_start"], dates["target_visible_end"]),
        }
        for split, frame in windows.items():
            lower, upper = boundary_limits[split]
            if frame["window_start_date"].min() < lower or frame["window_end_date"].max() > upper:
                raise ValueError(f"{split}: una ventana cruza su frontera temporal")

        target_sets = [set(values) for values in test_index["target_dates"]]
        targets_non_overlapping = all(
            left.isdisjoint(right)
            for index, left in enumerate(target_sets)
            for right in target_sets[index + 1 :]
        )
        target_lengths_ok = test_index["target_dates"].map(len).eq(int(window_cfg["horizon"])).all()
        targets_inside_test = (
            test_index["target_start_date"].min() >= dates["target_test_start"]
            and test_index["target_end_date"].max() <= dates["target_test_end"]
        )
        if not (targets_non_overlapping and target_lengths_ok and targets_inside_test):
            raise ValueError("test_index incumple horizonte, fronteras o no-solapamiento")

        windows_dir.mkdir(parents=True, exist_ok=True)
        daily_features.to_parquet(daily_features_path, index=False)
        output_files: list[Path] = [daily_features_path]
        for split, frame in windows.items():
            path = window_path(windows_dir, split)
            frame.to_parquet(path, index=False)
            output_files.append(path)
        test_index.to_parquet(test_index_path, index=False)
        output_files.append(test_index_path)

        report_rows = []
        for split, frame in windows.items():
            report_rows.append(
                {
                    "split": split,
                    "stride": stride_map[split],
                    "n_windows": int(len(frame)),
                    "date_min": frame["window_start_date"].min().strftime("%Y-%m-%d"),
                    "date_max": frame["window_end_date"].max().strftime("%Y-%m-%d"),
                    "boundary_check": "PASS",
                }
            )
        report_rows.append(
            {
                "split": "nvda_test",
                "stride": int(window_cfg["strides"]["target_test"]),
                "n_windows": int(len(test_index)),
                "date_min": test_index["target_start_date"].min().strftime("%Y-%m-%d"),
                "date_max": test_index["target_end_date"].max().strftime("%Y-%m-%d"),
                "boundary_check": "PASS",
            }
        )
        report = pd.DataFrame(report_rows)
        report.to_csv(report_path, index=False)
        output_files.append(report_path)
        checksums = write_checksums(checksums_path, output_files)

        counts = {split: int(len(frame)) for split, frame in windows.items()}
        counts["nvda_test"] = int(len(test_index))
        manifest = {
            "data_version": DATA_VERSION,
            "protocol_version": config["experiment"]["protocol_version"],
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": sha256_file(config_path),
            "input_snapshots": {
                str(clean_path.relative_to(ROOT)).replace("\\", "/"): clean_sha,
                str(split_path.relative_to(ROOT)).replace("\\", "/"): split_sha,
            },
            "pipeline_order": [
                "clean_daily_ohlcv",
                "daily_split_assignments",
                "features_within_each_block",
                "windows_within_each_block",
            ],
            "channels": channels,
            "formulas": {
                "log_return": "ln(Close_t / Close_t_minus_1)",
                "log_high_low_range": "ln(High_t / Low_t)",
                "log1p_volume": "ln(1 + Volume_t)",
            },
            "window_length": length,
            "context": int(window_cfg["context"]),
            "horizon": int(window_cfg["horizon"]),
            "strides": stride_map | {"nvda_test": int(window_cfg["strides"]["target_test"])},
            "daily_feature_counts": {
                block: int(len(frame)) for block, frame in daily_by_block.items()
            },
            "window_counts": counts,
            "test_index": {
                "n_rows": int(len(test_index)),
                "target_date_min": test_index["target_start_date"].min().strftime("%Y-%m-%d"),
                "target_date_max": test_index["target_end_date"].max().strftime("%Y-%m-%d"),
                "targets_non_overlapping": targets_non_overlapping,
                "all_targets_have_horizon_rows": bool(target_lengths_ok),
                "all_targets_inside_test": bool(targets_inside_test),
            },
            "boundary_evidence": {
                "donor_train": [str(dates["donor_train_start"].date()), str(dates["donor_train_end"].date())],
                "donor_validation": [str(dates["donor_validation_start"].date()), str(dates["donor_validation_end"].date())],
                "nvda_visible": [str(dates["target_visible_start"].date()), str(dates["target_visible_end"].date())],
                "nvda_full_history": [str(dates["target_hidden_start"].date()), str(dates["target_visible_end"].date())],
                "nvda_test_targets": [str(dates["target_test_start"].date()), str(dates["target_test_end"].date())],
            },
            "checksums_sha256": checksums,
        }
        write_json(manifest_path, manifest)
    except Exception as exc:  # noqa: BLE001
        print(f"FALLO features/windows: {exc}", file=sys.stderr)
        return 1

    print("OK features y ventanas post-split")
    for row in report.itertuples():
        print(f"  {row.split}: n={row.n_windows} stride={row.stride} {row.date_min}->{row.date_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
