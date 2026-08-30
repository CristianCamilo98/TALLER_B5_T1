from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def cristian_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_windows_dir() -> Path:
    return repo_root() / "data" / "features" / "windows"


def default_experiment_config() -> Path:
    return repo_root() / "configs" / "experiment.yaml"


def artifacts_dir() -> Path:
    return cristian_root() / "artifacts"


def outputs_dir() -> Path:
    return cristian_root() / "outputs"
