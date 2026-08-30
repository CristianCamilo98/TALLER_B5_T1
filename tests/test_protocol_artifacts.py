from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_protocol import load_experiment_config, root_path, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_experiment_config()[0]


def _windows(split: str) -> pd.DataFrame:
    config = _config()
    return pd.read_parquet(root_path(config, "windows_dir") / f"{split}.parquet")


def test_daily_split_boundaries_and_roles() -> None:
    config = _config()
    frame = pd.read_parquet(root_path(config, "daily_splits"))
    donors = set(config["universe"]["donors"])
    rules = {
        "donor_train": ("2012-01-03", "2021-12-31"),
        "donor_validation": ("2022-01-03", "2022-12-30"),
        "nvda_hidden": ("2012-01-03", "2022-06-30"),
        "nvda_visible": ("2022-07-01", "2022-12-30"),
        "nvda_test": ("2023-01-03", "2025-12-31"),
    }
    for split, (start, end) in rules.items():
        block = frame.loc[frame["split"].eq(split)]
        assert not block.empty
        assert block["date"].min() >= pd.Timestamp(start)
        assert block["date"].max() <= pd.Timestamp(end)
    donor_rows = frame.loc[frame["split"].isin(["donor_train", "donor_validation"])]
    assert set(donor_rows["ticker"]) == donors
    assert "NVDA" not in set(donor_rows["ticker"])
    assert not frame.duplicated(["date", "ticker"]).any()


def test_all_standard_windows_stay_inside_their_daily_blocks() -> None:
    config = _config()
    channels = config["features"]["channels"]
    features = pd.read_parquet(root_path(config, "daily_features"))
    expected = {
        "donor_train": ("2012-01-03", "2021-12-31", 5),
        "donor_validation": ("2022-01-03", "2022-12-30", 5),
        "nvda_visible": ("2022-07-01", "2022-12-30", 1),
        "nvda_full_history": ("2012-01-03", "2022-12-30", 1),
    }
    for split, (start, end, stride) in expected.items():
        windows = _windows(split)
        assert windows["window_start_date"].min() >= pd.Timestamp(start)
        assert windows["window_end_date"].max() <= pd.Timestamp(end)
        assert windows["features_flat"].map(len).eq(65 * len(channels)).all()
        assert np.isfinite(np.vstack(windows["features_flat"].map(np.asarray))).all()
        block = features.loc[features["feature_block"].eq(split)]
        for ticker, ticker_windows in windows.groupby("ticker"):
            dates = block.loc[block["ticker"].eq(ticker)].sort_values("date")["date"].reset_index(drop=True)
            positions = {date: index for index, date in enumerate(dates)}
            starts = [positions[date] for date in ticker_windows.sort_values("window_start_date")["window_start_date"]]
            assert all(right - left == stride for left, right in zip(starts, starts[1:]))


def test_test_index_has_non_overlapping_five_day_targets_entirely_in_test() -> None:
    config = _config()
    frame = pd.read_parquet(root_path(config, "test_index"))
    features = pd.read_parquet(root_path(config, "daily_features"))
    source = features.loc[features["feature_block"].eq("nvda_test_source")].sort_values("date")
    positions = {date: index for index, date in enumerate(source["date"].reset_index(drop=True))}
    starts = [positions[date] for date in frame["window_start_date"]]
    assert all(right - left == 5 for left, right in zip(starts, starts[1:]))
    seen: set[str] = set()
    for row in frame.itertuples():
        context_dates = list(row.context_dates)
        target_dates = list(row.target_dates)
        assert len(context_dates) == 60
        assert len(target_dates) == 5
        assert len(row.features_flat) == 195
        assert min(target_dates) >= "2023-01-03"
        assert max(target_dates) <= "2025-12-31"
        assert seen.isdisjoint(target_dates)
        seen.update(target_dates)
    assert frame["target_start_date"].min() == pd.Timestamp("2023-01-03")


def test_counts_and_checksums_match_manifests() -> None:
    config = _config()
    manifest_path = root_path(config, "features_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {
        split: len(_windows(split))
        for split in ["donor_train", "donor_validation", "nvda_visible", "nvda_full_history"]
    }
    counts["nvda_test"] = len(pd.read_parquet(root_path(config, "test_index")))
    assert counts == manifest["window_counts"]
    for rel, expected in manifest["checksums_sha256"].items():
        assert sha256_file(ROOT / rel) == expected


def test_no_obsolete_global_window_artifacts_remain() -> None:
    obsolete = [
        *ROOT.glob("data/features/windows_65_stride*.parquet"),
        *ROOT.glob("data/splits/window_splits_stride*.parquet"),
    ]
    assert obsolete == []
