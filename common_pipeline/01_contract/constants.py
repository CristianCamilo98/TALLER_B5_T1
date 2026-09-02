"""Shared constants for the common synthetic output contract."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATORS_ROOT = REPO_ROOT / "generadores"
DONOR_TRAIN_PATH = REPO_ROOT / "data" / "features" / "windows" / "donor_train.parquet"
_FALLBACK_DONOR_TRAIN_SHA256 = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"


def _sha256_existing_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


DONOR_TRAIN_SHA256 = (
    os.environ.get("SYNTHETIC_NVDA_DONOR_TRAIN_SHA256")
    or _sha256_existing_file(DONOR_TRAIN_PATH)
    or _FALLBACK_DONOR_TRAIN_SHA256
)

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
CERTIFIED_OUTPUTS_JSON = RESULTS_DIR / "certified_outputs.json"
BASELINE_OUTPUT_PATH = OUTPUTS_DIR / BASELINE_OUTPUT_NAME

EXPECTED_NEURAL_METHODS = 4
EXPECTED_SIMPLE_BASELINES = 1
NEURAL_METHOD_FAMILY = "neural_generator"
SIMPLE_BASELINE_FAMILY = "simple_baseline"

# Structural certification and membership in the official experiment are
# separate decisions. A well-formed Parquet may still be the wrong model for
# its owner's frozen experimental slot.
OFFICIAL_GENERATOR_ROLES = {
    "cristian": {"role": "wgan_gp", "aliases": ("wgan", "wgan_gp", "wgangp")},
    "daniel": {
        "role": "ddpm",
        "aliases": ("ddpm", "diffusion_ddpm", "ddpm_temporal_1d"),
    },
    "marco": {
        "role": "vae",
        "aliases": ("vae", "timevae", "time_vae", "marco_vae"),
    },
    "david": {
        "role": "normalizing_flow",
        "aliases": ("normalizing_flow", "normalized_flow", "normalizingflow"),
    },
}
OFFICIAL_BASELINE_ROLE = "bootstrap_jitter"

DONOR_LINEAGE_CANONICAL = "CANONICAL"
DONOR_LINEAGE_NON_CANONICAL = "NON_CANONICAL"
DONOR_LINEAGE_NOT_VERIFIABLE = "NOT_VERIFIABLE"

STATS_ATOL = 1e-6
