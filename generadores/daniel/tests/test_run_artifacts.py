from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from generadores.daniel.scripts.plot_training import plot_training
from generadores.daniel.src.frozen_runs import (
    SUMMARY_COLUMNS,
    build_frozen_summary,
    histories_numerically_equal,
    model_states_equal,
    validate_frozen_manifests,
)
from generadores.daniel.src.run_artifacts import (
    FROZEN_BASELINE,
    FROZEN_TRAINING_SEEDS,
    REQUIRED_MANIFEST_FIELDS,
    frozen_config_for_seed,
    frozen_run_id,
    global_channel_run_id,
    read_history,
    read_manifest,
    validate_frozen_baseline,
    write_history,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[3]


def _history() -> list[dict]:
    return [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "validation_loss": 0.9,
            "learning_rate": 0.0002,
            "epoch_seconds": 1.2,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "validation_loss": 0.7,
            "learning_rate": 0.0002,
            "epoch_seconds": 1.1,
        },
    ]


def test_repository_config_is_exact_frozen_baseline() -> None:
    path = ROOT / "generadores/daniel/config/diffusion.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_frozen_baseline(config)
    assert config == FROZEN_BASELINE
    changed = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed["training"]["learning_rate"] = 0.123
    with pytest.raises(ValueError, match="frozen"):
        validate_frozen_baseline(changed)


def test_runtime_seed_override_changes_only_training_seed() -> None:
    path = ROOT / "generadores/daniel/config/diffusion.yaml"
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    for seed in FROZEN_TRAINING_SEEDS:
        effective = frozen_config_for_seed(source, seed)
        expected = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected["reproducibility"]["seed"] = seed
        assert effective == expected
        assert effective["reproducibility"]["validation_seed"] == 424242
        assert frozen_run_id(seed) == f"diffusion_seed{seed}_frozen"
        assert global_channel_run_id(seed) == f"diffusion_seed{seed}_global_channel"
    assert source == FROZEN_BASELINE
    with pytest.raises(ValueError, match="one of"):
        frozen_config_for_seed(source, 7)


def test_history_schema_and_plot_are_driven_by_csv(tmp_path) -> None:
    history_path = tmp_path / "history.csv"
    figure_path = tmp_path / "loss.png"
    write_history(_history(), history_path)
    frame = read_history(history_path)
    assert list(frame["epoch"]) == [1, 2]
    result = plot_training(history_path, figure_path)
    assert result == {"best_epoch": 2, "best_validation_loss": 0.7}
    assert figure_path.is_file() and figure_path.stat().st_size > 0


def test_manifest_schema_round_trip(tmp_path) -> None:
    manifest = {field: "test" for field in REQUIRED_MANIFEST_FIELDS}
    manifest["seed"] = 42
    manifest["validation_seed"] = 424242
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest
    del manifest["git_commit"]
    with pytest.raises(ValueError, match="missing"):
        write_manifest(manifest, path)


def _frozen_manifest(seed: int) -> dict:
    effective = frozen_config_for_seed(FROZEN_BASELINE, seed)
    return {
        "run_id": frozen_run_id(seed),
        "training_seed": seed,
        "validation_seed": 424242,
        "git_commit": "frozen-code",
        "base_master_commit": "master",
        "canonical_raw_sha256": "raw",
        "donor_train_sha256": "train",
        "donor_validation_sha256": "validation",
        "train_count": 4910,
        "validation_count": 380,
        "window_shape": [65, 3],
        "channels": ["log_return", "log_high_low_range", "log1p_volume"],
        "normalizer_sha256": "normalizer",
        "effective_config": effective,
        "epochs_completed": 2,
        "best_epoch": 1,
        "best_validation_loss": 0.7 + seed / 10000,
        "final_train_loss": 0.5,
        "final_validation_loss": 0.8,
        "stopping_reason": "early_stopping_patience",
        "runtime_seconds": 10.0,
        "best_checkpoint_sha256": f"best-{seed}",
    }


def test_frozen_run_manifest_invariants_and_summary() -> None:
    manifests = [_frozen_manifest(seed) for seed in FROZEN_TRAINING_SEEDS]
    validate_frozen_manifests(manifests)
    summary = build_frozen_summary(manifests)
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert list(summary["seed"]) == [42, 123, 2026]
    changed = [_frozen_manifest(seed) for seed in FROZEN_TRAINING_SEEDS]
    changed[1]["validation_seed"] = 1
    with pytest.raises(ValueError, match="invariant"):
        validate_frozen_manifests(changed)


def test_reproduction_comparators_ignore_only_runtime(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_history(_history(), first)
    changed_runtime = pd.DataFrame(_history())
    changed_runtime["epoch_seconds"] += 100
    second.write_text(changed_runtime.to_csv(index=False), encoding="utf-8")
    equal, difference = histories_numerically_equal(first, second)
    assert equal and difference == 0.0
    state = {"weight": torch.tensor([1.0, 2.0])}
    equal, difference = model_states_equal(state, {"weight": state["weight"].clone()})
    assert equal and difference == 0.0
