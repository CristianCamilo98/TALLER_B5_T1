from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clean_ohlcv import ohlc_invariant_mask

ROOT = Path(__file__).resolve().parents[1]


def test_float_precision_edge_is_accepted_but_material_error_is_rejected() -> None:
    frame = pd.DataFrame(
        [
            {"Open": 9.5, "High": 10.0, "Low": 9.0, "Close": 10.0 + 1e-14, "Volume": 1},
            {"Open": 9.5, "High": 10.0, "Low": 9.0, "Close": 10.01, "Volume": 1},
        ]
    )
    mask = ohlc_invariant_mask(
        frame, absolute_tolerance=1e-10, relative_tolerance=1e-12
    )
    assert mask.tolist() == [True, False]


def test_clean_snapshot_quality_is_certified() -> None:
    manifest = json.loads((ROOT / "data/clean/clean_manifest.json").read_text(encoding="utf-8"))
    quality = pd.read_csv(ROOT / "data/clean/quality_report.csv").set_index("check")
    clean = pd.read_parquet(ROOT / "data/clean/ohlcv_clean.parquet")
    assert manifest["strict_float_edge_rows_rescued"] == int(quality.loc["strict_float_edge_rows", "value"])
    assert quality.loc["strict_float_edge_rows", "status"] == "PASS"
    assert manifest["dropped_material_ohlc_invariant"] == 0
    assert manifest["dropped_nan"] == 0
    assert len(clean) == 38_720
    assert clean.duplicated(["date", "ticker"]).sum() == 0
