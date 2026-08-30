from __future__ import annotations

from pathlib import Path

import discovery
from conftest import make_generator_tree


def test_discovery_fails_when_generator_has_multiple_outputs(tmp_path: Path, valid_parquet: Path, monkeypatch):
    second = tmp_path / "second.parquet"
    second.write_bytes(valid_parquet.read_bytes())
    make_generator_tree(
        tmp_path,
        {
            "alpha": [valid_parquet],
            "beta": [valid_parquet, second],
            "gamma": [valid_parquet],
            "delta": [valid_parquet],
        },
    )
    monkeypatch.setattr(discovery, "GENERATORS_ROOT", tmp_path / "generadores")
    result = discovery.discover_outputs()
    assert not result.ok
    assert any("beta" in error and "exactly one" in error for error in result.errors)


def test_discovery_succeeds_with_one_output_per_generator(tmp_path: Path, valid_parquet: Path, monkeypatch):
    make_generator_tree(
        tmp_path,
        {
            "alpha": [valid_parquet],
            "beta": [valid_parquet],
            "gamma": [valid_parquet],
            "delta": [valid_parquet],
        },
    )
    monkeypatch.setattr(discovery, "GENERATORS_ROOT", tmp_path / "generadores")
    result = discovery.discover_outputs()
    assert result.ok
    assert len(result.outputs) == 4
