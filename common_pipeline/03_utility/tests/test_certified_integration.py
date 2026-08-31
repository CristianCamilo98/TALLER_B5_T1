from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

io_synthetic = importlib.import_module("common_pipeline.03_utility.io_synthetic")
utility_run = importlib.import_module("common_pipeline.03_utility.utility_run")
build_mixtures = importlib.import_module("common_pipeline.03_utility.build_mixtures")
downstream = importlib.import_module("common_pipeline.03_utility.downstream_ridge")
registry = importlib.import_module("common_pipeline.01_contract.registry")

CHANNELS = ["log_return", "log_high_low_range", "log1p_volume"]


def _entry(root: Path, method_id: str, family: str, value: float = 0.0) -> dict:
    path = root / "outputs" / f"{method_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    window = np.full((65, 3), value, dtype=np.float32)
    count = 5000
    pd.DataFrame(
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
    ).to_parquet(path, index=False)
    return {
        "method_id": method_id,
        "method_family": family,
        "source_model": method_id,
        "source_directory": "outputs",
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


def _registry(root: Path, methods: list[dict], name: str = "registry.json") -> Path:
    path = root / name
    registry.write_certified_registry(methods, path=path)
    return path


def test_dynamic_four_neural_plus_baseline_without_owner_names(tmp_path: Path) -> None:
    names = ["alice", "bob", "charlie", "david_like_name"]
    methods = [_entry(tmp_path, name, "neural_generator", index) for index, name in enumerate(names)]
    methods.append(_entry(tmp_path, "simple_reference", "simple_baseline", 9.0))
    _entry(tmp_path, "new_generator_xyz", "neural_generator", 10.0)
    registry_path = _registry(tmp_path, methods)
    run_dir, _manifest = utility_run.prepare_run(
        registry_path=registry_path,
        repository_root=tmp_path,
        results_root=tmp_path / "runs",
        run_id="complete",
    )
    pools = io_synthetic.load_pools_for_run(run_dir, repository_root=tmp_path)
    assert set(pools) == {*names, "simple_reference"}
    assert "new_generator_xyz" not in pools


def test_utility_strict_mode_rejects_incomplete_registry(tmp_path: Path) -> None:
    methods = [_entry(tmp_path, name, "neural_generator") for name in ["a", "b", "c"]]
    methods.append(_entry(tmp_path, "simple", "simple_baseline"))
    registry_path = _registry(tmp_path, methods)
    with pytest.raises(RuntimeError, match="FINAL RUN BLOCKED"):
        utility_run.prepare_run(
            registry_path=registry_path,
            repository_root=tmp_path,
            results_root=tmp_path / "runs",
            run_id="incomplete",
        )


def test_real_only_is_one_common_result(monkeypatch) -> None:
    real = np.random.default_rng(1).normal(size=(62, 65, 3))
    test = np.random.default_rng(2).normal(size=(10, 65, 3))
    synthetic = {
        "alice": np.random.default_rng(3).normal(size=(200, 65, 3)),
        "bob": np.random.default_rng(4).normal(size=(200, 65, 3)),
    }
    calls = []

    def fake_evaluate(X_train_raw, y_train, scaler, X_test_scaled, y_test):
        calls.append(len(X_train_raw))
        return float(len(X_train_raw)), float(len(X_train_raw) / 2)

    monkeypatch.setattr(downstream, "evaluate_one", fake_evaluate)
    raw, summary = downstream.build_result_tables(real, test, synthetic)
    real_raw = raw[(raw.method == "REAL_ONLY") & (raw.ratio == 0.0)]
    real_summary = summary[(summary.method == "REAL_ONLY") & (summary.ratio == 0.0)]
    assert len(real_raw) == len(real_summary) == 1
    assert calls.count(62) == 1
    assert (summary.loc[summary.method != "REAL_ONLY", "delta_rmse_vs_real_only"] ==
            summary.loc[summary.method != "REAL_ONLY", "mean_rmse"] - 62.0).all()
    assert (summary.loc[summary.method != "REAL_ONLY", "delta_mae_vs_real_only"] ==
            summary.loc[summary.method != "REAL_ONLY", "mean_mae"] - 31.0).all()


def test_run_b_never_reads_run_a_calibrated_cache(tmp_path: Path) -> None:
    method_a = _entry(tmp_path / "repo_a", "alice", "neural_generator", 1.0)
    registry_a = _registry(tmp_path / "repo_a", [method_a], "registry.json")
    run_a, _ = utility_run.prepare_run(
        registry_path=registry_a,
        repository_root=tmp_path / "repo_a",
        results_root=tmp_path / "runs",
        run_id="run_A",
        allow_partial=True,
    )
    method_b = _entry(tmp_path / "repo_b", "alice", "neural_generator", 2.0)
    registry_b = _registry(tmp_path / "repo_b", [method_b], "registry.json")
    run_b, _ = utility_run.prepare_run(
        registry_path=registry_b,
        repository_root=tmp_path / "repo_b",
        results_root=tmp_path / "runs",
        run_id="run_B",
        allow_partial=True,
    )

    for run_dir, value in [(run_a, 11.0), (run_b, 22.0)]:
        path = utility_run.calibrated_pool_path(run_dir, "alice")
        path.parent.mkdir(parents=True)
        np.savez_compressed(
            path,
            values=np.full((200, 65, 3), value),
            valid_mask=np.ones(200, dtype=bool),
        )
        utility_run.update_run_manifest(
            run_dir,
            {
                "calibrated_pools": {
                    "alice": {
                        "path": utility_run.run_relative_path(run_dir, path),
                        "sha256": registry.sha256_file(path),
                        "valid": 200,
                        "invalid": 0,
                    }
                }
            },
        )
    loaded = build_mixtures.load_valid_calibrated_pool("alice", run_dir=run_b)
    assert np.all(loaded == 22.0)
