from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import baseline
from constants import BASELINE_SEED, DONOR_TRAIN_PATH, EXPECTED_ROWS, WINDOW_LENGTH
from io_utils import sha256_file, stack_features
from normalizer import load_donor_train_normalized
from validation import validate_output
from discovery import DiscoveredOutput


def test_baseline_reproducible_with_seed42(tmp_path: Path, monkeypatch):
    output_a = tmp_path / "a.parquet"
    output_b = tmp_path / "b.parquet"
    monkeypatch.setattr(baseline, "BASELINE_OUTPUT_PATH", output_a)
    first = baseline.build_baseline(output_path=output_a, seed=BASELINE_SEED)
    monkeypatch.setattr(baseline, "BASELINE_OUTPUT_PATH", output_b)
    second = baseline.build_baseline(output_path=output_b, seed=BASELINE_SEED)
    assert first.sha256 == second.sha256


def test_baseline_does_not_modify_donor_train():
    before = sha256_file(DONOR_TRAIN_PATH)
    _ = load_donor_train_normalized(DONOR_TRAIN_PATH)
    after = sha256_file(DONOR_TRAIN_PATH)
    assert before == after


def test_baseline_output_shape_and_contract(tmp_path: Path, monkeypatch):
    output_path = tmp_path / "baseline.parquet"
    monkeypatch.setattr(baseline, "BASELINE_OUTPUT_PATH", output_path)
    result = baseline.build_baseline(output_path=output_path)
    assert result.shape == (EXPECTED_ROWS, WINDOW_LENGTH, 3)

    frame = pd.read_parquet(output_path)
    discovered = DiscoveredOutput(
        generator_id="baseline",
        path=output_path,
        filename=output_path.name,
        rows=len(frame),
        sha256=sha256_file(output_path),
        columns=tuple(frame.columns),
    )
    row = validate_output(discovered, frame)
    assert row.contract_status == "PASS"
    tensor = stack_features(frame)
    assert np.isfinite(tensor).all()
    assert len(frame) == EXPECTED_ROWS
    assert frame["synthetic_id"].nunique() == EXPECTED_ROWS
