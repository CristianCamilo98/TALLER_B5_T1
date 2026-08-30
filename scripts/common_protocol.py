"""Utilidades compartidas del common core del protocolo Synthetic NVDA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiment_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_experiment_config(config)
    return config, config_path


def validate_experiment_config(config: dict[str, Any]) -> None:
    required_sections = {
        "experiment",
        "source",
        "universe",
        "dates",
        "cleaning",
        "features",
        "windows",
        "synthetic_experiment",
        "downstream",
        "paths",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"Faltan secciones de configuración: {missing}")

    target = config["universe"]["target"]
    donors = list(config["universe"]["donors"])
    if target in donors or len(donors) != len(set(donors)):
        raise ValueError("Target y donors deben ser disjuntos y los donors únicos")

    channels = list(config["features"]["channels"])
    expected_channels = ["log_return", "log_high_low_range", "log1p_volume"]
    if channels != expected_channels:
        raise ValueError(f"Orden de canales inválido: {channels}")

    window = config["windows"]
    if int(window["length"]) != int(window["context"]) + int(window["horizon"]):
        raise ValueError("window.length debe ser context + horizon")
    required_stride_keys = {
        "donor_train",
        "donor_validation",
        "target_visible",
        "target_full_history",
        "target_test",
    }
    strides = {key: int(value) for key, value in window["strides"].items()}
    if set(strides) != required_stride_keys or any(value <= 0 for value in strides.values()):
        raise ValueError("Faltan strides requeridos o contienen valores no positivos")

    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    if dates["donor_train_end"] >= dates["donor_validation_start"]:
        raise ValueError("Donor train y validation se solapan")
    if dates["target_hidden_end"] >= dates["target_visible_start"]:
        raise ValueError("Target hidden y visible se solapan")
    if dates["target_visible_end"] >= dates["target_test_start"]:
        raise ValueError("Target visible y test se solapan")

    seeds = list(config["synthetic_experiment"]["seeds"])
    ratios = [float(value) for value in config["synthetic_experiment"]["ratios"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds vacías o duplicadas")
    if not ratios or any(value <= 0 or value >= 1 for value in ratios):
        raise ValueError("Los ratios sintéticos deben estar estrictamente entre 0 y 1")
    if float(config["downstream"]["alpha"]) <= 0:
        raise ValueError("downstream.alpha debe ser positivo")


def root_path(config: dict[str, Any], key: str) -> Path:
    return ROOT / config["paths"][key]


def normalize_date(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series)
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_convert("UTC").dt.tz_localize(None)
    return values.dt.normalize()


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_snapshot(path: Path, manifest_path: Path) -> str:
    """Verifica el fichero contra el hash observado en su manifest, nunca contra uno hardcodeado."""
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Falta snapshot o manifest: {path}, {manifest_path}")
    manifest = read_manifest(manifest_path)
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    expected = manifest.get("checksums_sha256", {}).get(rel)
    if expected is None:
        raise ValueError(f"El manifest {manifest_path} no certifica {rel}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Snapshot alterado desde su manifest: {rel}\n"
            f"  manifest={expected}\n  actual={actual}"
        )
    return actual


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_checksums(path: Path, files: list[Path]) -> dict[str, str]:
    checksums = {
        str(file.relative_to(ROOT)).replace("\\", "/"): sha256_file(file)
        for file in files
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{digest}  {rel}" for rel, digest in checksums.items()) + "\n",
        encoding="utf-8",
    )
    return checksums
