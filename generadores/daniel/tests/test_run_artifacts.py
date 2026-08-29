from pathlib import Path

import pytest
import yaml

from generadores.daniel.scripts.plot_training import plot_training
from generadores.daniel.src.run_artifacts import (
    FROZEN_BASELINE,
    REQUIRED_MANIFEST_FIELDS,
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
