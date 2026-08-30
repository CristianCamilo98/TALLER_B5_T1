from pathlib import Path

import pytest
import torch

from generadores.daniel.src.data_adapter import (
    inspect_donor_parquet,
    load_canonical_donor_tensors,
)
from generadores.daniel.src.validation import (
    CHANNEL_ORDER,
    validate_channel_order,
    validate_window_tensor,
)


ROOT = Path(__file__).resolve().parents[3]


def test_canonical_donor_counts_shapes_channels_and_hashes() -> None:
    train, validation = load_canonical_donor_tensors(ROOT)
    assert train.tensor.shape == (4910, 65, 3)
    assert validation.tensor.shape == (380, 65, 3)
    assert train.tensor.dtype == validation.tensor.dtype == torch.float32
    assert train.channels == validation.channels == CHANNEL_ORDER
    assert torch.isfinite(train.tensor).all()
    assert torch.isfinite(validation.tensor).all()
    assert len(set(train.tickers)) == len(set(validation.tickers)) == 10
    assert tuple(train.metadata["ticker"]) == train.tickers
    assert tuple(validation.metadata["ticker"]) == validation.tickers


def test_real_parquet_representation_is_certified_flat_195() -> None:
    train = inspect_donor_parquet("donor_train", ROOT)
    validation = inspect_donor_parquet("donor_validation", ROOT)
    assert train["rows"] == 4910
    assert validation["rows"] == 380
    assert train["features_flat_lengths"] == validation["features_flat_lengths"] == [195]
    assert train["sha256"] == train["expected_sha256"]
    assert validation["sha256"] == validation["expected_sha256"]
    assert tuple(train["channels"]) == CHANNEL_ORDER


def test_validators_reject_legacy_channel_name_and_nonfinite_tensor() -> None:
    with pytest.raises(ValueError, match="Channel order"):
        validate_channel_order(("log_return", "log_high_low_range", "log_volume"))
    tensor = torch.zeros((2, 65, 3))
    tensor[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_window_tensor(tensor)
