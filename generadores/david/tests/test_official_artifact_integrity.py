"""Read-only integrity check for David's frozen official Normalizing Flow artifacts.

This test does NOT train anything. It only verifies that the checkpoint and
the official Parquet already committed to the repository are the exact,
unmodified canonical artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_MODULE = REPO_ROOT / "common_pipeline" / "01_contract"
if str(CONTRACT_MODULE) not in sys.path:
    sys.path.insert(0, str(CONTRACT_MODULE))

from constants import CANONICAL_MEAN, CANONICAL_STD  # noqa: E402
from io_utils import sha256_file, stack_features  # noqa: E402

CHECKPOINT_PATH = (
    REPO_ROOT
    / "generadores"
    / "david"
    / "artifacts"
    / "checkpoints"
    / "normalizing_flow_seed42.npz"
)
OFFICIAL_OUTPUT_PATH = (
    REPO_ROOT
    / "generadores"
    / "david"
    / "outputs"
    / "normalizing_flow_seed42_normalized.parquet"
)
PROVENANCE_PATH = OFFICIAL_OUTPUT_PATH.with_suffix(".provenance.json")

EXPECTED_CHECKPOINT_SHA256 = "d2a9480f0017a4b8780124771f864f0f85f284e5f8d46566e4754821acea3b25"
EXPECTED_OUTPUT_SHA256 = "f36fe7ee5b79400a1567446e038fb21723e5a4b973c224c32f5a945dc66219ed"
EXPECTED_DONOR_TRAIN_SHA256 = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"
EXPECTED_DONOR_VALIDATION_SHA256 = "134f51a2ac9e546bf1a2f21f4efbf56a62bf019a08de14209058563b0a88ae23"
EXPECTED_ROWS = 5000
EXPECTED_LOGICAL_SHAPE = (5000, 65, 3)
EXPECTED_SEED = 42


def test_official_checkpoint_is_frozen_and_canonical():
    assert CHECKPOINT_PATH.is_file(), f"missing checkpoint: {CHECKPOINT_PATH}"
    assert sha256_file(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_SHA256


def test_official_output_exists_under_canonical_normalizing_flow_name():
    assert OFFICIAL_OUTPUT_PATH.is_file(), f"missing official output: {OFFICIAL_OUTPUT_PATH}"
    assert OFFICIAL_OUTPUT_PATH.name == "normalizing_flow_seed42_normalized.parquet"


def test_official_output_is_frozen_and_canonical():
    assert sha256_file(OFFICIAL_OUTPUT_PATH) == EXPECTED_OUTPUT_SHA256


def test_official_output_contract_shape_and_finiteness():
    frame = pd.read_parquet(OFFICIAL_OUTPUT_PATH)
    assert len(frame) == EXPECTED_ROWS

    tensor = stack_features(frame)
    assert tensor.shape == EXPECTED_LOGICAL_SHAPE
    assert np.isfinite(tensor).all()

    assert frame["source_model"].eq("normalizing_flow").all()
    assert frame["training_seed"].eq(EXPECTED_SEED).all()


def test_provenance_declares_canonical_lineage_and_seeds():
    assert PROVENANCE_PATH.is_file(), f"missing provenance: {PROVENANCE_PATH}"
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert provenance["training_seed"] == EXPECTED_SEED
    assert provenance["sampling_seed"] == EXPECTED_SEED

    assert provenance["donor_train_sha256"] == EXPECTED_DONOR_TRAIN_SHA256
    assert provenance["donor_validation_sha256"] == EXPECTED_DONOR_VALIDATION_SHA256

    assert provenance["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert sha256_file(CHECKPOINT_PATH) == provenance["checkpoint_sha256"]

    np.testing.assert_allclose(provenance["mean"], np.asarray(CANONICAL_MEAN), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(provenance["std"], np.asarray(CANONICAL_STD), rtol=0.0, atol=1.0e-12)

    assert provenance["parquet_sha256"] == EXPECTED_OUTPUT_SHA256
    assert provenance["logical_shape"] == list(EXPECTED_LOGICAL_SHAPE)
    assert provenance["training"]["best_epoch"] == 89
    assert provenance["training"]["epochs_completed"] == 289
    assert provenance["training"]["training_config"]["epochs"] == 10000
    assert provenance["training"]["best_validation_nll"] == 178.86906998378228
