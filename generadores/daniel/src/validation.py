"""Shape and metadata validation for Daniel's isolated generator."""

from __future__ import annotations

from collections.abc import Sequence

import torch

CHANNEL_ORDER = ("log_return", "log_high_low_range", "log1p_volume")
WINDOW_LENGTH = 65
INPUT_CHANNELS = 3
EXPECTED_COUNTS = {"donor_train": 4910, "donor_validation": 380}
DONOR_TICKERS = frozenset(
    {"AMD", "INTC", "QCOM", "AVGO", "MU", "TXN", "ADI", "MCHP", "MRVL", "NXPI"}
)
ALLOWED_FLOAT_DTYPES = (torch.float32, torch.float64)


def validate_channel_order(channels: Sequence[str]) -> None:
    """Require the exact certified feature order."""

    actual = tuple(channels)
    if actual != CHANNEL_ORDER:
        raise ValueError(f"Channel order must be {CHANNEL_ORDER}, got {actual}")


def validate_window_tensor(
    tensor: torch.Tensor,
    *,
    expected_count: int | None = None,
    name: str = "tensor",
) -> None:
    """Validate the public `(N, 65, 3)` tensor contract."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have 3 dimensions, got {tuple(tensor.shape)}")
    expected_tail = (WINDOW_LENGTH, INPUT_CHANNELS)
    if tuple(tensor.shape[1:]) != expected_tail:
        raise ValueError(f"{name} must have trailing shape {expected_tail}, got {tuple(tensor.shape)}")
    if expected_count is not None and tensor.shape[0] != expected_count:
        raise ValueError(f"{name} must contain {expected_count} windows, got {tensor.shape[0]}")
    if tensor.dtype not in ALLOWED_FLOAT_DTYPES:
        raise TypeError(f"{name} dtype must be float32 or float64, got {tensor.dtype}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")


def validate_tickers(
    tickers: Sequence[str],
    *,
    expected_count: int | None = None,
    require_all_donors: bool = False,
) -> None:
    """Validate donor labels without admitting target or unknown symbols."""

    if expected_count is not None and len(tickers) != expected_count:
        raise ValueError(f"Expected {expected_count} ticker labels, got {len(tickers)}")
    actual = set(tickers)
    unknown = actual - DONOR_TICKERS
    if unknown:
        raise ValueError(f"Unknown/non-donor tickers: {sorted(unknown)}")
    if require_all_donors and actual != DONOR_TICKERS:
        missing = DONOR_TICKERS - actual
        raise ValueError(f"Missing donor tickers: {sorted(missing)}")


def validate_tensor_and_tickers(
    tensor: torch.Tensor,
    tickers: Sequence[str],
    *,
    expected_count: int | None = None,
    name: str = "tensor",
) -> None:
    """Validate aligned tensor rows and donor labels."""

    validate_window_tensor(tensor, expected_count=expected_count, name=name)
    validate_tickers(tickers, expected_count=tensor.shape[0])
