"""TEMPORARY / REPLACE WITH COMMON NORMALIZER WHEN AVAILABLE.

Statistics are fitted exclusively from donor-train windows. Each ticker and
channel receives one mean and population standard deviation. Validation data
is transform-only and can never update stored parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import torch

from .validation import CHANNEL_ORDER, validate_tensor_and_tickers


class TemporaryTickerChannelNormalizer:
    """Train-only, ticker-specific z-score normalization."""

    def __init__(self, *, std_tolerance: float = 1e-12) -> None:
        if std_tolerance <= 0:
            raise ValueError("std_tolerance must be positive")
        self.std_tolerance = float(std_tolerance)
        self._means: dict[str, torch.Tensor] = {}
        self._stds: dict[str, torch.Tensor] = {}

    @property
    def fitted(self) -> bool:
        return bool(self._means)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self._means))

    def fit(
        self,
        train_tensor: torch.Tensor,
        train_tickers: Sequence[str],
    ) -> "TemporaryTickerChannelNormalizer":
        """Fit one `(mean, std)` vector per ticker using train only."""

        validate_tensor_and_tickers(train_tensor, train_tickers, name="train_tensor")
        means: dict[str, torch.Tensor] = {}
        stds: dict[str, torch.Tensor] = {}
        labels = tuple(train_tickers)
        for ticker in sorted(set(labels)):
            indices = [index for index, label in enumerate(labels) if label == ticker]
            values = train_tensor[indices].detach().to(device="cpu", dtype=torch.float64)
            flattened = values.reshape(-1, values.shape[-1])
            mean = flattened.mean(dim=0)
            std = flattened.std(dim=0, unbiased=False)
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError(f"Non-finite scaler parameters for {ticker}")
            if torch.any(std <= self.std_tolerance):
                raise ValueError(
                    f"Near-zero standard deviation for {ticker}: {std.tolist()}"
                )
            means[ticker] = mean.clone()
            stds[ticker] = std.clone()
        self._means = means
        self._stds = stds
        return self

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Normalizer has not been fitted")

    def transform(self, tensor: torch.Tensor, tickers: Sequence[str]) -> torch.Tensor:
        """Apply frozen train statistics without mutating parameters or input."""

        return self._apply(tensor, tickers, inverse=False)

    def inverse_transform(self, tensor: torch.Tensor, tickers: Sequence[str]) -> torch.Tensor:
        """Map normalized values back to the original feature space."""

        return self._apply(tensor, tickers, inverse=True)

    def _apply(
        self,
        tensor: torch.Tensor,
        tickers: Sequence[str],
        *,
        inverse: bool,
    ) -> torch.Tensor:
        self._require_fitted()
        validate_tensor_and_tickers(tensor, tickers, name="tensor")
        labels = tuple(tickers)
        unknown = set(labels) - set(self._means)
        if unknown:
            raise ValueError(f"No fitted parameters for tickers: {sorted(unknown)}")

        result = torch.empty_like(tensor)
        for ticker in sorted(set(labels)):
            indices = [index for index, label in enumerate(labels) if label == ticker]
            mean = self._means[ticker].to(device=tensor.device, dtype=tensor.dtype)
            std = self._stds[ticker].to(device=tensor.device, dtype=tensor.dtype)
            if inverse:
                result[indices] = tensor[indices] * std + mean
            else:
                result[indices] = (tensor[indices] - mean) / std
        return result

    def state_dict(self) -> dict:
        """Return a JSON-serializable parameter representation."""

        self._require_fitted()
        return {
            "type": "temporary_ticker_channel_zscore",
            "warning": "TEMPORARY / REPLACE WITH COMMON NORMALIZER WHEN AVAILABLE",
            "fit_split": "donor_train",
            "channels": list(CHANNEL_ORDER),
            "std_tolerance": self.std_tolerance,
            "parameters": {
                ticker: {
                    "mean": self._means[ticker].tolist(),
                    "std": self._stds[ticker].tolist(),
                }
                for ticker in self.tickers
            },
        }

    def save_json(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.state_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path | str) -> "TemporaryTickerChannelNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(payload["channels"]) != CHANNEL_ORDER:
            raise ValueError("Serialized scaler uses an incompatible channel order")
        instance = cls(std_tolerance=float(payload["std_tolerance"]))
        for ticker, parameters in payload["parameters"].items():
            instance._means[ticker] = torch.tensor(parameters["mean"], dtype=torch.float64)
            instance._stds[ticker] = torch.tensor(parameters["std"], dtype=torch.float64)
        instance._require_fitted()
        return instance
