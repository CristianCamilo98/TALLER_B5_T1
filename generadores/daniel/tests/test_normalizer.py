import copy

import pytest
import torch

from generadores.daniel.src.temporary_normalizer import (
    TemporaryTickerChannelNormalizer,
)


def _data() -> tuple[torch.Tensor, tuple[str, ...]]:
    generator = torch.Generator().manual_seed(7)
    tensor = torch.randn((6, 65, 3), generator=generator)
    tensor[3:] = tensor[3:] * 2.0 + 4.0
    return tensor, ("AMD", "AMD", "AMD", "INTC", "INTC", "INTC")


def test_fit_transform_round_trip_and_order_preservation(tmp_path) -> None:
    tensor, tickers = _data()
    original = tensor.clone()
    scaler = TemporaryTickerChannelNormalizer().fit(tensor, tickers)
    transformed = scaler.transform(tensor, tickers)
    reconstructed = scaler.inverse_transform(transformed, tickers)
    assert transformed.shape == tensor.shape
    assert torch.equal(tensor, original)
    assert torch.isfinite(transformed).all()
    assert torch.max(torch.abs(reconstructed - tensor)).item() < 1e-5
    assert scaler.tickers == ("AMD", "INTC")
    for parameters in scaler.state_dict()["parameters"].values():
        assert all(value > 0 for value in parameters["std"])

    path = tmp_path / "scaler.json"
    scaler.save_json(path)
    loaded = TemporaryTickerChannelNormalizer.load_json(path)
    assert loaded.state_dict() == scaler.state_dict()


def test_validation_values_cannot_change_train_fitted_parameters() -> None:
    train, train_tickers = _data()
    validation = train[:2].clone()
    validation_tickers = train_tickers[:2]
    scaler = TemporaryTickerChannelNormalizer().fit(train, train_tickers)
    before = copy.deepcopy(scaler.state_dict())
    scaler.transform(validation, validation_tickers)
    validation.add_(1_000_000.0)
    scaler.transform(validation, validation_tickers)
    assert scaler.state_dict() == before


def test_near_zero_standard_deviation_is_rejected() -> None:
    tensor = torch.ones((2, 65, 3))
    with pytest.raises(ValueError, match="Near-zero"):
        TemporaryTickerChannelNormalizer().fit(tensor, ("AMD", "AMD"))
