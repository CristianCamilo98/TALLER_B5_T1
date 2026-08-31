"""Load only phase-01-certified normalized synthetic pools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import utility_run

WINDOW_LENGTH = 65
N_CHANNELS = 3
CHANNEL_ORDER = ("log_return", "log_high_low_range", "log1p_volume")


def _load_registered_pool(path: Path, entry: dict) -> np.ndarray:
    frame = pd.read_parquet(path, columns=["features_flat"])
    if len(frame) != int(entry["rows"]):
        raise ValueError(f"{entry['method_id']}: row count changed after certification")
    lengths = frame["features_flat"].map(len)
    if not lengths.eq(int(entry["features_flat_length"])).all():
        raise ValueError(f"{entry['method_id']}: features_flat length changed")
    values = np.stack(
        [
            np.asarray(row, dtype=np.float32).reshape(WINDOW_LENGTH, N_CHANNELS)
            for row in frame["features_flat"]
        ]
    )
    expected_shape = tuple(entry["logical_shape"])
    if values.shape != expected_shape:
        raise ValueError(
            f"{entry['method_id']}: shape {values.shape} != certified {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{entry['method_id']}: non-finite values after certification")
    return values


def load_pools_for_run(
    run_dir: Path,
    *,
    repository_root: Path = utility_run.REPOSITORY_ROOT,
) -> dict[str, np.ndarray]:
    manifest = utility_run.load_run(run_dir)
    resolved = utility_run.resolve_manifest_methods(
        manifest,
        repository_root=repository_root,
    )
    return {
        method: _load_registered_pool(path, entry)
        for method, (path, entry) in sorted(resolved.items())
    }


def discover_synthetic_pools(
    *,
    registry_path: Path = utility_run.DEFAULT_REGISTRY_PATH,
    repository_root: Path = utility_run.REPOSITORY_ROOT,
    results_root: Path = utility_run.RESULTS_ROOT,
    run_id: str | None = None,
    allow_partial: bool = False,
) -> dict[str, np.ndarray]:
    """Prepare an isolated run and load every certified method dynamically."""

    run_dir, _manifest = utility_run.prepare_run(
        registry_path=registry_path,
        repository_root=repository_root,
        results_root=results_root,
        run_id=run_id,
        allow_partial=allow_partial,
    )
    return load_pools_for_run(run_dir, repository_root=repository_root)
