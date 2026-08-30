from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from constants import (  # noqa: E402
    CHANNEL_ORDER,
    EXPECTED_ROWS,
    EXPECTED_TRAINING_SEED,
    FEATURE_DIM,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from io_utils import flatten_window  # noqa: E402


@pytest.fixture
def valid_window() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(WINDOW_LENGTH, N_CHANNELS))


@pytest.fixture
def valid_frame(valid_window: np.ndarray) -> pd.DataFrame:
    records = []
    for synthetic_id in range(4):
        window = valid_window + synthetic_id * 0.01
        records.append(
            {
                "synthetic_id": synthetic_id,
                "source_model": "fixture_model",
                "training_seed": EXPECTED_TRAINING_SEED,
                "space": GLOBAL_NORMALIZED_SPACE,
                "window_length": WINDOW_LENGTH,
                "n_channels": N_CHANNELS,
                "channel_order": list(CHANNEL_ORDER),
                "features_flat": flatten_window(window.astype(np.float32)),
            }
        )
    return pd.DataFrame.from_records(records)


@pytest.fixture
def valid_parquet(tmp_path: Path, valid_frame: pd.DataFrame) -> Path:
    path = tmp_path / "valid.parquet"
    valid_frame.to_parquet(path, index=False)
    return path


def make_generator_tree(root: Path, generator_outputs: dict[str, list[Path]]) -> None:
    for generator_id, parquet_paths in generator_outputs.items():
        outputs_dir = root / "generadores" / generator_id / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        for src in parquet_paths:
            target = outputs_dir / src.name
            target.write_bytes(src.read_bytes())
