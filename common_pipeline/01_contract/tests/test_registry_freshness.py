from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

registry = importlib.import_module("common_pipeline.01_contract.registry")

CHANNELS = ["log_return", "log_high_low_range", "log1p_volume"]


def _write_output(
    repository_root: Path,
    *,
    owner: str = "alice",
    valid: bool = True,
    value_offset: float = 0.0,
) -> Path:
    path = repository_root / "generadores" / owner / "outputs" / "synthetic.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 5000
    values = np.zeros((count, 195), dtype=np.float32)
    values[:, 0] = np.arange(count, dtype=np.float32) + value_offset
    frame = pd.DataFrame(
        {
            "synthetic_id": np.arange(count),
            "source_model": [f"{owner}_model"] * count,
            "training_seed": [42] * count,
            "space": ["global_channel_normalized"] * count,
            "window_length": [65] * count,
            "n_channels": [3] * count,
            "channel_order": [CHANNELS] * count,
            "features_flat": [row for row in values],
        }
    )
    if not valid:
        frame = frame.drop(columns="space")
    frame.to_parquet(path, index=False)
    return path


def _regenerate_registry(repository_root: Path, registry_path: Path) -> dict:
    entries, _statuses = registry.current_certified_snapshot(
        repository_root=repository_root,
    )
    return registry.write_certified_registry(entries, path=registry_path)


def test_fail_then_pass_output_makes_registry_stale_until_regenerated(
    tmp_path: Path,
) -> None:
    output = _write_output(tmp_path, valid=False)
    registry_path = tmp_path / "certified_outputs.json"
    original = _regenerate_registry(tmp_path, registry_path)
    assert original["methods"] == []

    _write_output(tmp_path, valid=True)
    with pytest.raises(RuntimeError, match="CERTIFIED REGISTRY STALE"):
        registry.validate_registry_freshness(original, repository_root=tmp_path)

    refreshed = _regenerate_registry(tmp_path, registry_path)
    result = registry.validate_registry_freshness(
        refreshed,
        repository_root=tmp_path,
    )
    assert result["fresh"] is True
    assert result["methods"] == ["alice"]
    assert output.is_file()


def test_changed_certified_output_sha_makes_registry_stale(tmp_path: Path) -> None:
    _write_output(tmp_path)
    registry_path = tmp_path / "certified_outputs.json"
    payload = _regenerate_registry(tmp_path, registry_path)

    _write_output(tmp_path, value_offset=0.5)
    with pytest.raises(RuntimeError, match=r"changed_sha=\['alice'\]"):
        registry.validate_registry_freshness(payload, repository_root=tmp_path)


def test_missing_certified_output_makes_registry_stale(tmp_path: Path) -> None:
    output = _write_output(tmp_path)
    registry_path = tmp_path / "certified_outputs.json"
    payload = _regenerate_registry(tmp_path, registry_path)

    output.unlink()
    with pytest.raises(RuntimeError, match=r"removed=\['alice'\]"):
        registry.validate_registry_freshness(payload, repository_root=tmp_path)


def test_synchronized_registry_is_fresh(tmp_path: Path) -> None:
    _write_output(tmp_path)
    registry_path = tmp_path / "certified_outputs.json"
    payload = _regenerate_registry(tmp_path, registry_path)

    result = registry.validate_registry_freshness(
        payload,
        repository_root=tmp_path,
    )
    assert result["fresh"] is True
    assert result["methods"] == ["alice"]
