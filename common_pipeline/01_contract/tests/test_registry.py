from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import registry
from constants import NEURAL_METHOD_FAMILY
from discovery import DiscoveredOutput
from validation import validate_output


def _discovered(path: Path, generator_id: str = "alice") -> DiscoveredOutput:
    frame = pd.read_parquet(path)
    return DiscoveredOutput(
        generator_id=generator_id,
        path=path,
        filename=path.name,
        rows=len(frame),
        sha256=registry.sha256_file(path),
        columns=tuple(frame.columns),
    )


def test_only_contract_pass_outputs_enter_registry(
    tmp_path: Path,
    valid_frame: pd.DataFrame,
) -> None:
    good_path = tmp_path / "generadores/alice/outputs/good.parquet"
    bad_path = tmp_path / "generadores/bob/outputs/bad.parquet"
    good_path.parent.mkdir(parents=True)
    bad_path.parent.mkdir(parents=True)
    valid_frame.to_parquet(good_path, index=False)
    valid_frame.drop(columns=["space"]).to_parquet(bad_path, index=False)

    good_output = _discovered(good_path, "alice")
    bad_output = _discovered(bad_path, "bob")
    good = validate_output(good_output, valid_frame, expected_rows=4)
    bad_frame = pd.read_parquet(bad_path)
    bad = validate_output(bad_output, bad_frame, expected_rows=4)
    assert good.contract_status == "PASS"
    assert bad.contract_status == "FAIL"

    with pytest.raises(ValueError, match="Only PASS"):
        registry.certified_entry(
            bad_output,
            bad,
            method_family=NEURAL_METHOD_FAMILY,
            repository_root=tmp_path,
        )

    entry = registry.certified_entry(
        good_output,
        good,
        method_family=NEURAL_METHOD_FAMILY,
        repository_root=tmp_path,
    )
    # The registry is always the final 5,000-row contract. The four-row
    # fixture above isolates validation behavior without weakening that gate.
    entry["rows"] = 5000
    entry["logical_shape"] = [5000, 65, 3]
    registry_path = tmp_path / "certified_outputs.json"
    payload = registry.write_certified_registry([entry], path=registry_path)
    assert [method["method_id"] for method in payload["methods"]] == ["alice"]
    assert all(method["contract_status"] == "PASS" for method in payload["methods"])

    tampered = dict(payload)
    tampered["methods"] = [dict(payload["methods"][0], training_seed=123)]
    with pytest.raises(ValueError, match="training_seed"):
        registry.validate_registry_payload(tampered)


@pytest.mark.parametrize("column", ["training_seed", "space", "channel_order"])
def test_required_metadata_missing_is_contract_fail(
    tmp_path: Path,
    valid_frame: pd.DataFrame,
    column: str,
) -> None:
    frame = valid_frame.drop(columns=[column])
    path = tmp_path / f"missing_{column}.parquet"
    frame.to_parquet(path, index=False)
    report = validate_output(_discovered(path), frame, expected_rows=4)
    assert report.contract_status == "FAIL"


def test_wrong_seed_and_wrong_rows_fail(
    tmp_path: Path,
    valid_frame: pd.DataFrame,
) -> None:
    wrong_seed = valid_frame.copy()
    wrong_seed["training_seed"] = 123
    seed_path = tmp_path / "wrong_seed.parquet"
    wrong_seed.to_parquet(seed_path, index=False)
    seed_report = validate_output(_discovered(seed_path), wrong_seed, expected_rows=4)
    assert seed_report.contract_status == "FAIL"

    short = valid_frame.iloc[:3].copy()
    rows_path = tmp_path / "wrong_rows.parquet"
    short.to_parquet(rows_path, index=False)
    rows_report = validate_output(_discovered(rows_path), short, expected_rows=4)
    assert rows_report.contract_status == "FAIL"
