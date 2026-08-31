"""Run-isolated state for the common utility pipeline."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "runs"
DEFAULT_REGISTRY_PATH = (
    REPOSITORY_ROOT / "common_pipeline/01_contract/results/certified_outputs.json"
)
NVDA_VISIBLE_PATH = REPOSITORY_ROOT / "data/features/windows/nvda_visible.parquet"

contract_registry = importlib.import_module("common_pipeline.01_contract.registry")


def _git_commit(repository_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNAVAILABLE_TEST_FIXTURE"
    return result.stdout.strip()


def default_run_id(registry_path: Path) -> str:
    return f"utility_{contract_registry.registry_sha256(registry_path)[:12]}"


def prepare_run(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    repository_root: Path = REPOSITORY_ROOT,
    results_root: Path = RESULTS_ROOT,
    run_id: str | None = None,
    allow_partial: bool = False,
) -> tuple[Path, dict]:
    registry_path = registry_path.resolve()
    repository_root = repository_root.resolve()
    payload = contract_registry.load_certified_registry(registry_path)
    contract_registry.ensure_registry_completeness(payload, allow_partial=allow_partial)
    resolved = contract_registry.resolve_certified_paths(
        payload,
        repository_root=repository_root,
    )
    selected_run_id = run_id or default_run_id(registry_path)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", selected_run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and dash")

    run_dir = results_root.resolve() / selected_run_id
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "run_id": selected_run_id,
        "git_commit": _git_commit(repository_root),
        "certified_registry_path": registry_path.relative_to(repository_root).as_posix(),
        "certified_registry_sha256": contract_registry.registry_sha256(registry_path),
        "allow_partial": bool(allow_partial),
        "input_methods": [entry for _path, entry in resolved.values()],
        "input_parquet_sha256": {
            method: entry["sha256"] for method, (_path, entry) in resolved.items()
        },
        "nvda_calibration": {
            "source_path": NVDA_VISIBLE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "source_sha256": contract_registry.registry_sha256(NVDA_VISIBLE_PATH),
            "expected_unique_daily_observations": 126,
            "ddof": 0,
            "formula": "mu_NVDA + sigma_NVDA * Z_synthetic",
        },
        "subsampling_seeds": [42, 123, 2026],
        "ratios": [0.0, 0.25, 0.5, 0.75],
        "downstream": {
            "n_real_visible": 62,
            "features": [
                "rv5",
                "rv20",
                "rv60",
                "mean_abs_return20",
                "momentum20",
                "mean_range20",
                "mean_log_volume20",
                "std_log_volume20",
            ],
            "feature_scaler_fit": "62 real visible windows only",
            "model": "Ridge(alpha=1.0, fit_intercept=True)",
            "metrics": ["rmse", "mae"],
        },
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "run_id",
            "git_commit",
            "certified_registry_path",
            "certified_registry_sha256",
            "allow_partial",
            "input_methods",
            "input_parquet_sha256",
            "subsampling_seeds",
            "ratios",
            "downstream",
        )
        if any(existing.get(key) != manifest.get(key) for key in immutable_keys):
            raise RuntimeError(
                f"Run {selected_run_id!r} already exists with different certified inputs"
            )
        return run_dir, existing

    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return run_dir, manifest


def load_run(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing utility run manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def existing_run_dir(
    *,
    results_root: Path = RESULTS_ROOT,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    run_id: str | None = None,
) -> Path:
    selected = run_id or default_run_id(registry_path.resolve())
    run_dir = results_root.resolve() / selected
    load_run(run_dir)
    return run_dir


def update_run_manifest(run_dir: Path, updates: dict) -> dict:
    manifest = load_run(run_dir)
    manifest.update(updates)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def resolve_manifest_methods(
    manifest: dict,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, tuple[Path, dict]]:
    payload = {
        "registry_version": contract_registry.REGISTRY_VERSION,
        "expected_neural_methods": 0,
        "expected_simple_baselines": 0,
        "methods": manifest["input_methods"],
    }
    contract_registry.validate_registry_payload(payload)
    return contract_registry.resolve_certified_paths(
        payload,
        repository_root=repository_root,
    )


def calibrated_pool_path(run_dir: Path, method_id: str) -> Path:
    return run_dir / "calibrated_pools" / f"{method_id}_calibrated.npz"


def run_relative_path(run_dir: Path, path: Path) -> str:
    """Store portable paths relative to the isolated run directory."""

    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def resolve_run_artifact(run_dir: Path, relative_path: str) -> Path:
    root = run_dir.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Run artifact escapes run directory: {relative_path}") from exc
    return path
