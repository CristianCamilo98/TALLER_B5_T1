from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_MODULE = REPO_ROOT / "common_pipeline" / "01_contract"
SCRIPT_PATH = REPO_ROOT / "generadores" / "david" / "scripts" / "generate_normalized.py"
EXPERIMENT_SCRIPT_PATH = REPO_ROOT / "generadores" / "david" / "scripts" / "experiment_normalized.py"
PLOT_SCRIPT_PATH = REPO_ROOT / "generadores" / "david" / "scripts" / "plot_experiment_diagnostics.py"
if str(CONTRACT_MODULE) not in sys.path:
    sys.path.insert(0, str(CONTRACT_MODULE))

import baseline  # noqa: E402
from constants import (  # noqa: E402
    BASELINE_NOISE_SCALE,
    BASELINE_SEED,
    BASELINE_SOURCE_MODEL,
    CHANNEL_ORDER,
    EXPECTED_ROWS,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from discovery import DiscoveredOutput  # noqa: E402
from io_utils import sha256_file, stack_features  # noqa: E402
from validation import validate_output  # noqa: E402


def load_generate_module():
    spec = importlib.util.spec_from_file_location("david_generate_normalized", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_experiment_module():
    spec = importlib.util.spec_from_file_location("david_experiment_normalized", EXPERIMENT_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_plot_module():
    spec = importlib.util.spec_from_file_location("david_plot_experiment_diagnostics", PLOT_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_output_path_is_david_official_location():
    module = load_generate_module()
    assert module.DEFAULT_OUTPUT_PATH == (
        REPO_ROOT
        / "generadores"
        / "david"
        / "outputs"
        / "bootstrap_jitter_seed42_normalized.parquet"
    )


def test_default_generator_is_temporal_jitter_improvement():
    module = load_generate_module()

    assert module.OFFICIAL_SOURCE_MODEL == "temporal_jitter_0p40_rho0p85"
    assert module.OFFICIAL_NOISE_SCALE == 0.40
    assert module.OFFICIAL_RHO == 0.85
    assert module.source_model_name(module.OFFICIAL_NOISE_SCALE, module.OFFICIAL_RHO) == (
        module.OFFICIAL_SOURCE_MODEL
    )


def test_david_baseline_is_reproducible_with_seed42(tmp_path: Path):
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"

    first = baseline.build_baseline(output_path=first_path, seed=BASELINE_SEED)
    second = baseline.build_baseline(output_path=second_path, seed=BASELINE_SEED)

    assert first.sha256 == second.sha256


def test_david_baseline_changes_when_seed_changes(tmp_path: Path):
    seed42_path = tmp_path / "seed42.parquet"
    seed123_path = tmp_path / "seed123.parquet"

    seed42 = baseline.build_baseline(output_path=seed42_path, seed=42)
    seed123 = baseline.build_baseline(output_path=seed123_path, seed=123)

    assert seed42.sha256 != seed123.sha256


def test_david_baseline_contract(tmp_path: Path):
    output_path = tmp_path / "david.parquet"
    result = baseline.build_baseline(
        output_path=output_path,
        seed=BASELINE_SEED,
        noise_scale=BASELINE_NOISE_SCALE,
        n_windows=EXPECTED_ROWS,
    )
    frame = pd.read_parquet(output_path)

    assert result.shape == (EXPECTED_ROWS, WINDOW_LENGTH, N_CHANNELS)
    assert list(frame.columns) == [
        "synthetic_id",
        "source_model",
        "training_seed",
        "space",
        "window_length",
        "n_channels",
        "channel_order",
        "features_flat",
    ]
    assert frame["source_model"].eq(BASELINE_SOURCE_MODEL).all()
    assert frame["training_seed"].eq(BASELINE_SEED).all()
    assert frame["space"].eq(GLOBAL_NORMALIZED_SPACE).all()
    assert frame["window_length"].eq(WINDOW_LENGTH).all()
    assert frame["n_channels"].eq(N_CHANNELS).all()
    assert frame["synthetic_id"].tolist() == list(range(EXPECTED_ROWS))
    assert all(tuple(value) == CHANNEL_ORDER for value in frame["channel_order"])

    tensor = stack_features(frame)
    assert tensor.shape == (EXPECTED_ROWS, WINDOW_LENGTH, N_CHANNELS)
    assert np.isfinite(tensor).all()

    discovered = DiscoveredOutput(
        generator_id="david",
        path=output_path,
        filename=output_path.name,
        rows=len(frame),
        sha256=sha256_file(output_path),
        columns=tuple(frame.columns),
    )
    report = validate_output(discovered, frame)
    assert report.contract_status == "PASS"


def test_experimental_candidates_stay_outside_official_outputs():
    module = load_experiment_module()
    official_outputs_dir = REPO_ROOT / "generadores" / "david" / "outputs"

    assert official_outputs_dir not in module.EXPERIMENT_OUTPUTS_DIR.parents
    assert module.OFFICIAL_OUTPUT.parent == official_outputs_dir


def test_experimental_candidate_frame_uses_canonical_contract():
    module = load_experiment_module()
    windows = np.zeros((2, WINDOW_LENGTH, N_CHANNELS), dtype=np.float32)

    frame = module.make_canonical_frame(windows, source_model="unit_test_candidate")

    assert list(frame.columns) == [
        "synthetic_id",
        "source_model",
        "training_seed",
        "space",
        "window_length",
        "n_channels",
        "channel_order",
        "features_flat",
    ]
    assert frame["source_model"].eq("unit_test_candidate").all()
    assert frame["training_seed"].eq(BASELINE_SEED).all()
    assert frame["space"].eq(GLOBAL_NORMALIZED_SPACE).all()
    assert all(tuple(value) == CHANNEL_ORDER for value in frame["channel_order"])


def test_diagnostic_plot_parser_extracts_noise_and_rho():
    module = load_plot_module()

    assert module.parse_noise_and_rho("temporal_jitter_0p40_rho0p85") == (0.40, 0.85)
    assert module.parse_noise_and_rho("bootstrap_jitter_0p15") == (None, None)
