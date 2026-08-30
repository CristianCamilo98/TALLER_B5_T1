import copy
from pathlib import Path

import pytest
import torch

from generadores.daniel.src.data_adapter import load_donor_windows
from generadores.daniel.src.temporary_normalizer import GlobalChannelNormalizer


ROOT = Path(__file__).resolve().parents[3]
CANONICAL_MEAN = torch.tensor(
    [
        0.00081142897100880656,
        0.026025805148914841,
        16.06027218135258,
    ],
    dtype=torch.float64,
)
CANONICAL_STD = torch.tensor(
    [
        0.023515504591060377,
        0.016724288791728319,
        1.0933253360280637,
    ],
    dtype=torch.float64,
)


def _data() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn((6, 65, 3), generator=generator, dtype=torch.float64)


def test_fit_transform_round_trip_serialization_and_dtype(tmp_path) -> None:
    tensor = _data()
    original = tensor.clone()
    scaler = GlobalChannelNormalizer().fit(tensor)
    transformed = scaler.transform(tensor)
    reconstructed = scaler.inverse_transform(transformed)
    assert transformed.shape == tensor.shape
    assert transformed.dtype == torch.float32
    assert reconstructed.dtype == torch.float64
    assert torch.equal(tensor, original)
    assert torch.isfinite(transformed).all()
    assert torch.max(torch.abs(reconstructed - tensor)).item() < 2e-7
    state = scaler.state_dict()
    assert len(state["mean"]) == len(state["std"]) == 3
    assert state["fit_axes"] == [0, 1]
    assert state["ddof"] == 0

    path = tmp_path / "scaler.json"
    scaler.save_json(path)
    loaded = GlobalChannelNormalizer.load_json(path)
    assert loaded.state_dict() == scaler.state_dict()


def test_validation_values_cannot_change_train_fitted_parameters() -> None:
    train = _data()
    validation = train[:2].clone()
    scaler = GlobalChannelNormalizer().fit(train)
    before = copy.deepcopy(scaler.state_dict())
    scaler.transform(validation)
    validation.add_(1_000_000.0)
    scaler.transform(validation)
    assert scaler.state_dict() == before


def test_window_order_does_not_change_global_statistics() -> None:
    train = _data()
    first = GlobalChannelNormalizer().fit(train)
    second = GlobalChannelNormalizer().fit(train[torch.tensor([5, 2, 0, 4, 1, 3])])
    assert torch.allclose(first.mean, second.mean, rtol=0.0, atol=1e-15)
    assert torch.allclose(first.std, second.std, rtol=0.0, atol=1e-15)


def test_zero_policy_replaces_near_zero_std_with_one() -> None:
    tensor = torch.ones((2, 65, 3), dtype=torch.float64)
    scaler = GlobalChannelNormalizer().fit(tensor)
    assert torch.equal(scaler.std, torch.ones(3, dtype=torch.float64))
    assert torch.equal(scaler.transform(tensor), torch.zeros_like(tensor, dtype=torch.float32))


def test_fit_rejects_float32_to_protect_common_precision() -> None:
    with pytest.raises(TypeError, match="float64"):
        GlobalChannelNormalizer().fit(_data().float())


def test_canonical_statistics_match_common_float64_contract() -> None:
    train = load_donor_windows("donor_train", ROOT, dtype=torch.float64)
    scaler = GlobalChannelNormalizer().fit(train.tensor)
    direct_mean = torch.from_numpy(train.tensor.numpy().mean(axis=(0, 1), dtype="float64"))
    direct_std = torch.from_numpy(train.tensor.numpy().std(axis=(0, 1), ddof=0, dtype="float64"))
    assert torch.equal(scaler.mean, direct_mean)
    assert torch.equal(scaler.std, direct_std)
    assert torch.allclose(scaler.mean, CANONICAL_MEAN, rtol=0.0, atol=5e-15)
    assert torch.allclose(scaler.std, CANONICAL_STD, rtol=0.0, atol=5e-14)


def test_global_transform_has_no_ticker_dependency() -> None:
    tensor = _data()
    scaler = GlobalChannelNormalizer().fit(tensor)
    first = scaler.transform(tensor)
    second = scaler.transform(tensor.clone())
    assert torch.equal(first, second)
