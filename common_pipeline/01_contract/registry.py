"""Certified synthetic-output registry shared by common phases 02 and 03.

Phase 01 is the only authority that decides contract conformance. Consumers
may verify registered hashes and tensor shape defensively, but they must never
infer missing metadata or add unregistered files.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:  # Package import from phases 02/03.
    from .constants import (
        CERTIFIED_OUTPUTS_JSON,
        CHANNEL_ORDER,
        EXPECTED_ROWS,
        EXPECTED_TRAINING_SEED,
        EXPECTED_NEURAL_METHODS,
        EXPECTED_SIMPLE_BASELINES,
        FEATURE_DIM,
        GLOBAL_NORMALIZED_SPACE,
        NEURAL_METHOD_FAMILY,
        N_CHANNELS,
        REPO_ROOT,
        SIMPLE_BASELINE_FAMILY,
        WINDOW_LENGTH,
    )
except ImportError:  # Direct script execution from 01_contract.
    from constants import (  # type: ignore
        CERTIFIED_OUTPUTS_JSON,
        CHANNEL_ORDER,
        EXPECTED_ROWS,
        EXPECTED_TRAINING_SEED,
        EXPECTED_NEURAL_METHODS,
        EXPECTED_SIMPLE_BASELINES,
        FEATURE_DIM,
        GLOBAL_NORMALIZED_SPACE,
        NEURAL_METHOD_FAMILY,
        N_CHANNELS,
        REPO_ROOT,
        SIMPLE_BASELINE_FAMILY,
        WINDOW_LENGTH,
    )

REGISTRY_VERSION = "certified_synthetic_outputs_v1"
REQUIRED_METHOD_FIELDS = {
    "method_id",
    "method_family",
    "source_model",
    "source_directory",
    "path",
    "sha256",
    "rows",
    "logical_shape",
    "training_seed",
    "space",
    "window_length",
    "n_channels",
    "channel_order",
    "features_flat_length",
    "normalization_status",
    "contract_status",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _repo_relative(path: Path, repository_root: Path = REPO_ROOT) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _source_model(path: Path) -> str:
    frame = pd.read_parquet(path, columns=["source_model"])
    values = frame["source_model"].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"{path}: source_model must contain exactly one value")
    return str(values[0])


def certified_entry(
    output: Any,
    report: Any,
    *,
    method_family: str,
    method_id: str | None = None,
    source_directory: str | None = None,
    repository_root: Path = REPO_ROOT,
) -> dict:
    if report.contract_status != "PASS":
        raise ValueError("Only PASS outputs may enter the certified registry")
    if method_family not in {NEURAL_METHOD_FAMILY, SIMPLE_BASELINE_FAMILY}:
        raise ValueError(f"Unsupported method family {method_family!r}")
    relative_path = _repo_relative(output.path, repository_root)
    return {
        "method_id": method_id or output.generator_id,
        "method_family": method_family,
        "source_model": _source_model(output.path),
        "source_directory": source_directory
        or Path(relative_path).parent.parent.as_posix(),
        "path": relative_path,
        "sha256": output.sha256,
        "rows": report.rows,
        "logical_shape": [report.rows, WINDOW_LENGTH, N_CHANNELS],
        "training_seed": report.training_seed,
        "space": report.space,
        "window_length": WINDOW_LENGTH,
        "n_channels": N_CHANNELS,
        "channel_order": list(CHANNEL_ORDER),
        "features_flat_length": FEATURE_DIM,
        "normalization_status": report.normalization_provenance,
        "contract_status": report.contract_status,
    }


def write_certified_registry(
    methods: Iterable[dict],
    *,
    path: Path = CERTIFIED_OUTPUTS_JSON,
) -> dict:
    entries = sorted((dict(item) for item in methods), key=lambda item: item["method_id"])
    method_ids = [item["method_id"] for item in entries]
    if len(method_ids) != len(set(method_ids)):
        raise ValueError("Certified method_id values must be unique")
    payload = {
        "registry_version": REGISTRY_VERSION,
        "contract_authority": "common_pipeline/01_contract",
        "expected_neural_methods": EXPECTED_NEURAL_METHODS,
        "expected_simple_baselines": EXPECTED_SIMPLE_BASELINES,
        "methods": entries,
    }
    validate_registry_payload(payload)
    write_json(path, payload)
    return payload


def validate_registry_payload(payload: dict) -> None:
    if payload.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("Unsupported certified registry version")
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("Certified registry methods must be a list")
    method_ids: list[str] = []
    for method in methods:
        missing = REQUIRED_METHOD_FIELDS - set(method)
        if missing:
            raise ValueError(f"Certified registry entry missing fields: {sorted(missing)}")
        if method["contract_status"] != "PASS":
            raise ValueError("Certified registry cannot contain FAIL entries")
        method_ids.append(str(method["method_id"]))
        if method["method_family"] not in {
            NEURAL_METHOD_FAMILY,
            SIMPLE_BASELINE_FAMILY,
        }:
            raise ValueError("Certified registry contains an unknown method family")
        if int(method["rows"]) != EXPECTED_ROWS:
            raise ValueError("Certified registry rows do not match the canonical contract")
        if list(method["logical_shape"]) != [EXPECTED_ROWS, WINDOW_LENGTH, N_CHANNELS]:
            raise ValueError("Certified registry logical_shape is not canonical")
        if int(method["training_seed"]) != EXPECTED_TRAINING_SEED:
            raise ValueError("Certified registry training_seed is not canonical")
        if method["space"] != GLOBAL_NORMALIZED_SPACE:
            raise ValueError("Certified registry space is not canonical")
        if int(method["window_length"]) != WINDOW_LENGTH:
            raise ValueError("Certified registry window_length is not canonical")
        if int(method["n_channels"]) != N_CHANNELS:
            raise ValueError("Certified registry n_channels is not canonical")
        if list(method["channel_order"]) != list(CHANNEL_ORDER):
            raise ValueError("Certified registry channel_order is not canonical")
        if int(method["features_flat_length"]) != FEATURE_DIM:
            raise ValueError("Certified registry features_flat_length is not canonical")
        if not re.fullmatch(r"[0-9a-f]{64}", str(method["sha256"])):
            raise ValueError("Certified registry SHA256 must be 64 lowercase hex characters")
        registered_path = Path(str(method["path"]))
        if registered_path.is_absolute() or ".." in registered_path.parts:
            raise ValueError("Certified registry paths must be repository-relative")
    if len(method_ids) != len(set(method_ids)):
        raise ValueError("Certified registry method_id values must be unique")


def load_certified_registry(path: Path = CERTIFIED_OUTPUTS_JSON) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_registry_payload(payload)
    return payload


def ensure_registry_completeness(
    payload: dict,
    *,
    allow_partial: bool = False,
    expected_neural: int | None = None,
    expected_baselines: int | None = None,
) -> None:
    neural_expected = int(
        payload["expected_neural_methods"] if expected_neural is None else expected_neural
    )
    baseline_expected = int(
        payload["expected_simple_baselines"]
        if expected_baselines is None
        else expected_baselines
    )
    neural = sum(m["method_family"] == NEURAL_METHOD_FAMILY for m in payload["methods"])
    baselines = sum(
        m["method_family"] == SIMPLE_BASELINE_FAMILY for m in payload["methods"]
    )
    if not allow_partial and (neural != neural_expected or baselines != baseline_expected):
        raise RuntimeError(
            "FINAL RUN BLOCKED — REQUIRED CERTIFIED METHODS MISSING: "
            f"neural={neural}/{neural_expected}, "
            f"simple_baseline={baselines}/{baseline_expected}"
        )


def resolve_certified_paths(
    payload: dict,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, tuple[Path, dict]]:
    resolved: dict[str, tuple[Path, dict]] = {}
    root = repository_root.resolve()
    for method in payload["methods"]:
        path = (root / method["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Certified path escapes repository root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Certified output missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != method["sha256"]:
            raise ValueError(
                f"Certified output SHA mismatch for {method['method_id']}: "
                f"{actual_hash} != {method['sha256']}"
            )
        resolved[str(method["method_id"])] = (path, dict(method))
    return resolved


def registry_sha256(path: Path = CERTIFIED_OUTPUTS_JSON) -> str:
    return sha256_file(path)
