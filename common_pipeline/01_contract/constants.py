"""Shared constants for the common synthetic output contract."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATORS_ROOT = REPO_ROOT / "generadores"
DONOR_TRAIN_PATH = REPO_ROOT / "data" / "features" / "windows" / "donor_train.parquet"
DONOR_TRAIN_SHA256 = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"

EXPECTED_GENERATOR_COUNT = 4
EXPECTED_ROWS = 5000
EXPECTED_TRAINING_SEED = 42
WINDOW_LENGTH = 65
N_CHANNELS = 3
FEATURE_DIM = WINDOW_LENGTH * N_CHANNELS
DONOR_TRAIN_ROWS = 4910
DONOR_TRAIN_SHAPE = (DONOR_TRAIN_ROWS, WINDOW_LENGTH, N_CHANNELS)

CHANNEL_ORDER = ("log_return", "log_high_low_range", "log1p_volume")
GLOBAL_NORMALIZED_SPACE = "global_channel_normalized"

CANONICAL_MEAN = (
    0.00081142897100880656,
    0.026025805148914841,
    16.06027218135258,
)
CANONICAL_STD = (
    0.023515504591060377,
    0.016724288791728319,
    1.0933253360280637,
)
CANONICAL_NORMALIZER_SHA256 = (
    "7e0fcce9c67d6a01581df4bed12e130555b164e7e1f846c39b25b4996eecef8e"
)

CANONICAL_COLUMNS = (
    "synthetic_id",
    "source_model",
    "training_seed",
    "space",
    "window_length",
    "n_channels",
    "channel_order",
    "features_flat",
)

FORBIDDEN_METADATA_COLUMNS = frozenset(
    {
        "ticker",
        "window_start_date",
        "window_end_date",
        "date",
        "dates",
        "session_dates",
    }
)

BASELINE_SOURCE_MODEL = "bootstrap_jitter"
BASELINE_SEED = 42
BASELINE_NOISE_SCALE = 0.05
BASELINE_OUTPUT_NAME = "bootstrap_jitter_seed42_normalized.parquet"

MODULE_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_ROOT / "results"
OUTPUTS_DIR = MODULE_ROOT / "outputs"
CONTRACT_REPORT_CSV = RESULTS_DIR / "output_contract_report.csv"
CONTRACT_REPORT_JSON = RESULTS_DIR / "output_contract_report.json"
BASELINE_OUTPUT_PATH = OUTPUTS_DIR / BASELINE_OUTPUT_NAME

STATS_ATOL = 1e-6
