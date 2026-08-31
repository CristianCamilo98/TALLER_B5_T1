from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate_fidelity import load_registry_methods

registry = importlib.import_module("common_pipeline.01_contract.registry")
CHANNELS = ["log_return", "log_high_low_range", "log1p_volume"]


@pytest.fixture(autouse=True)
def _treat_fixture_registries_as_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """These unit fixtures test registry consumption, not phase-01 discovery."""

    monkeypatch.setattr(
        registry,
        "validate_registry_freshness",
        lambda payload, repository_root: {"fresh": True},
    )


def _method(root: Path, method_id: str, family: str) -> dict:
    path = root / "published" / f"{method_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    window = np.zeros((65, 3), dtype=np.float32)
    count = 5000
    frame = pd.DataFrame(
        {
            "synthetic_id": range(count),
            "source_model": [method_id] * count,
            "training_seed": [42] * count,
            "space": ["global_channel_normalized"] * count,
            "window_length": [65] * count,
            "n_channels": [3] * count,
            "channel_order": [CHANNELS] * count,
            "features_flat": [window.reshape(-1)] * count,
        }
    )
    frame.to_parquet(path, index=False)
    return {
        "method_id": method_id,
        "method_family": family,
        "source_model": method_id,
        "source_directory": "published",
        "path": path.relative_to(root).as_posix(),
        "sha256": registry.sha256_file(path),
        "rows": count,
        "logical_shape": [count, 65, 3],
        "training_seed": 42,
        "space": "global_channel_normalized",
        "window_length": 65,
        "n_channels": 3,
        "channel_order": CHANNELS,
        "features_flat_length": 195,
        "normalization_status": "NORMALIZATION_NUMERICALLY_MATCHES",
        "contract_status": "PASS",
    }


def test_arbitrary_certified_names_are_loaded_and_unregistered_file_is_ignored(
    tmp_path: Path,
) -> None:
    names = ["alice", "bob", "charlie", "david_like_name"]
    methods = [_method(tmp_path, name, "neural_generator") for name in names]
    methods.append(_method(tmp_path, "simple_reference", "simple_baseline"))
    # This conforming-looking file is deliberately absent from the registry.
    _method(tmp_path, "new_generator_xyz", "neural_generator")
    registry_path = tmp_path / "certified_outputs.json"
    registry.write_certified_registry(methods, path=registry_path)

    _payload, resolved = load_registry_methods(
        registry_path,
        repository_root=tmp_path,
        allow_partial=False,
    )
    assert set(resolved) == {*names, "simple_reference"}
    assert "new_generator_xyz" not in resolved


def test_strict_registry_rejects_three_neural_plus_baseline(tmp_path: Path) -> None:
    methods = [
        _method(tmp_path, name, "neural_generator")
        for name in ["alice", "bob", "charlie"]
    ]
    methods.append(_method(tmp_path, "simple_reference", "simple_baseline"))
    registry_path = tmp_path / "certified_outputs.json"
    registry.write_certified_registry(methods, path=registry_path)
    with pytest.raises(RuntimeError, match="FINAL RUN BLOCKED"):
        load_registry_methods(
            registry_path,
            repository_root=tmp_path,
            allow_partial=False,
        )


def test_fidelity_invokes_shared_registry_freshness_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = _method(tmp_path, "alice", "neural_generator")
    registry_path = tmp_path / "certified_outputs.json"
    registry.write_certified_registry([method], path=registry_path)
    calls: list[Path] = []

    def record_freshness(payload: dict, *, repository_root: Path) -> dict:
        calls.append(repository_root)
        return {"fresh": True}

    monkeypatch.setattr(registry, "validate_registry_freshness", record_freshness)
    load_registry_methods(
        registry_path,
        repository_root=tmp_path,
        allow_partial=True,
    )
    assert calls == [tmp_path]
