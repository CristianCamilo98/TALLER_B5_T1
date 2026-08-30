"""Global donor-train normalization aligned with the team contract.

The scaler is fitted in float64 on all donor-train windows and produces one
mean and one population standard deviation per channel. Validation is always
transform-only. Normalized tensors are converted to float32 at the DDPM
boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .validation import CHANNEL_ORDER, validate_window_tensor


class GlobalChannelNormalizer:
    """Train-only global z-score with one parameter pair per channel."""

    FIT_DTYPE = torch.float64
    OUTPUT_DTYPE = torch.float32

    def __init__(self, *, std_threshold: float = 1e-8) -> None:
        if std_threshold <= 0:
            raise ValueError("std_threshold must be positive")
        self.std_threshold = float(std_threshold)
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._raw_std: torch.Tensor | None = None

    @property
    def fitted(self) -> bool:
        return self._mean is not None and self._std is not None

    @property
    def mean(self) -> torch.Tensor:
        self._require_fitted()
        return self._mean.clone()  # type: ignore[union-attr]

    @property
    def std(self) -> torch.Tensor:
        self._require_fitted()
        return self._std.clone()  # type: ignore[union-attr]

    def fit(self, train_tensor: torch.Tensor) -> "GlobalChannelNormalizer":
        """Fit float64 statistics over `(window, session)` of donor train."""

        validate_window_tensor(train_tensor, name="train_tensor")
        if train_tensor.dtype != self.FIT_DTYPE:
            raise TypeError(
                "GlobalChannelNormalizer.fit requires donor_train loaded as float64 "
                "so scaler statistics preserve the common contract precision"
            )
        values = train_tensor.detach().to(device="cpu", dtype=self.FIT_DTYPE)
        # NumPy float64 reductions reproduce Cristian's canonical procedure
        # byte-for-byte at the scalar level; ddof=0 is the population std.
        array = values.numpy()
        mean = torch.from_numpy(np.mean(array, axis=(0, 1), dtype=np.float64))
        raw_std = torch.from_numpy(np.std(array, axis=(0, 1), ddof=0, dtype=np.float64))
        if not torch.isfinite(mean).all() or not torch.isfinite(raw_std).all():
            raise ValueError("Global scaler parameters contain NaN/Inf")
        std = torch.where(raw_std < self.std_threshold, torch.ones_like(raw_std), raw_std)
        self._mean = mean.clone()
        self._raw_std = raw_std.clone()
        self._std = std.clone()
        return self

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Normalizer has not been fitted")

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply frozen train statistics and return the DDPM float32 tensor."""

        self._require_fitted()
        validate_window_tensor(tensor, name="tensor")
        values = tensor.to(dtype=self.FIT_DTYPE)
        mean = self._mean.to(device=tensor.device)  # type: ignore[union-attr]
        std = self._std.to(device=tensor.device)  # type: ignore[union-attr]
        normalized = (values - mean) / std
        if not torch.isfinite(normalized).all():
            raise ValueError("Normalized tensor contains NaN/Inf")
        return normalized.to(dtype=self.OUTPUT_DTYPE)

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Map normalized values back to feature space as float64."""

        self._require_fitted()
        validate_window_tensor(tensor, name="tensor")
        values = tensor.to(dtype=self.FIT_DTYPE)
        mean = self._mean.to(device=tensor.device)  # type: ignore[union-attr]
        std = self._std.to(device=tensor.device)  # type: ignore[union-attr]
        reconstructed = values * std + mean
        if not torch.isfinite(reconstructed).all():
            raise ValueError("Inverse-transformed tensor contains NaN/Inf")
        return reconstructed

    def state_dict(self) -> dict:
        """Return the JSON-serializable common normalization contract."""

        self._require_fitted()
        return {
            "type": "global_channel_zscore",
            "scope": "all donor_train windows and sessions",
            "fit_split": "donor_train",
            "channels": list(CHANNEL_ORDER),
            "fit_axes": [0, 1],
            "ddof": 0,
            "fit_dtype": "float64",
            "output_dtype": "float32",
            "std_threshold": self.std_threshold,
            "zero_variance_policy": "replace sigma with 1.0 when sigma < threshold",
            "mean": self._mean.tolist(),  # type: ignore[union-attr]
            "raw_std": self._raw_std.tolist(),  # type: ignore[union-attr]
            "std": self._std.tolist(),  # type: ignore[union-attr]
        }

    def save_json(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load_json(cls, path: Path | str) -> "GlobalChannelNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "global_channel_zscore":
            raise ValueError("Serialized scaler is not a global channel normalizer")
        if tuple(payload["channels"]) != CHANNEL_ORDER:
            raise ValueError("Serialized scaler uses an incompatible channel order")
        if payload.get("fit_axes") != [0, 1] or payload.get("ddof") != 0:
            raise ValueError("Serialized scaler violates the common fit contract")
        if payload.get("fit_dtype") != "float64" or payload.get("output_dtype") != "float32":
            raise ValueError("Serialized scaler uses incompatible dtypes")
        instance = cls(std_threshold=float(payload["std_threshold"]))
        instance._mean = torch.tensor(payload["mean"], dtype=torch.float64)
        instance._raw_std = torch.tensor(
            payload.get("raw_std", payload["std"]), dtype=torch.float64
        )
        instance._std = torch.tensor(payload["std"], dtype=torch.float64)
        if instance._mean.shape != (3,) or instance._std.shape != (3,):
            raise ValueError("Serialized scaler must contain exactly three channels")
        if not torch.isfinite(instance._mean).all() or not torch.isfinite(instance._std).all():
            raise ValueError("Serialized scaler parameters contain NaN/Inf")
        instance._require_fitted()
        return instance
