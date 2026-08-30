from __future__ import annotations

import numpy as np
import pandas as pd

from constants import CHANNEL_ORDER, FEATURE_DIM, WINDOW_LENGTH
from discovery import DiscoveredOutput
from io_utils import reconstruct_tensor
from validation import validate_output


def _discovered(path) -> DiscoveredOutput:
    frame = pd.read_parquet(path)
    return DiscoveredOutput(
        generator_id="fixture",
        path=path,
        filename=path.name,
        rows=len(frame),
        sha256="0" * 64,
        columns=tuple(frame.columns),
    )


def test_rejects_incorrect_shape(valid_parquet, valid_frame):
    bad = valid_frame.copy()
    bad.at[0, "features_flat"] = bad.at[0, "features_flat"][:10]
    row = validate_output(_discovered(valid_parquet), bad)
    assert row.contract_status == "FAIL"
    assert any("195" in error for error in row.errors)


def test_rejects_incorrect_channel_order(valid_parquet, valid_frame):
    bad = valid_frame.copy()
    bad["channel_order"] = [list(reversed(CHANNEL_ORDER))] * len(bad)
    row = validate_output(_discovered(valid_parquet), bad)
    assert row.contract_status == "FAIL"
    assert any("channel_order" in error for error in row.errors)


def test_rejects_nan_and_inf(valid_parquet, valid_frame):
    bad = valid_frame.copy()
    flat = list(bad.at[0, "features_flat"])
    flat[0] = float("nan")
    bad.at[0, "features_flat"] = flat
    row = validate_output(_discovered(valid_parquet), bad)
    assert row.contract_status == "FAIL"
    assert row.finite == "NO"


def test_reconstructs_flat_to_65x3(valid_frame):
    values = valid_frame.iloc[0]["features_flat"]
    tensor = reconstruct_tensor(values)
    assert tensor.shape == (WINDOW_LENGTH, FEATURE_DIM // WINDOW_LENGTH)


def test_detects_exact_duplicates(valid_parquet, valid_frame):
    bad = pd.concat([valid_frame, valid_frame.iloc[[0]]], ignore_index=True)
    bad.loc[4, "synthetic_id"] = 4
    row = validate_output(_discovered(valid_parquet), bad)
    assert row.contract_status == "FAIL"
    assert row.exact_duplicates >= 1
