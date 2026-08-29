"""TEMPORARY adapter from certified donor parquets to model tensors.

Replace this module with the common project loader when that contract exists.
It deliberately exposes only donor-train and donor-validation artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .validation import (
    CHANNEL_ORDER,
    DONOR_TICKERS,
    EXPECTED_COUNTS,
    INPUT_CHANNELS,
    WINDOW_LENGTH,
    validate_channel_order,
    validate_tensor_and_tickers,
)

_ALLOWED_SPLITS = frozenset(EXPECTED_COUNTS)
_REQUIRED_COLUMNS = (
    "split",
    "ticker",
    "window_start_date",
    "window_end_date",
    "features_flat",
)


@dataclass(frozen=True)
class DonorWindowBatch:
    """A tensor plus row-aligned donor provenance."""

    tensor: torch.Tensor
    tickers: tuple[str, ...]
    metadata: pd.DataFrame
    split: str
    input_sha256: str
    channels: tuple[str, ...] = CHANNEL_ORDER

    def validate(self) -> None:
        expected = EXPECTED_COUNTS[self.split]
        validate_channel_order(self.channels)
        validate_tensor_and_tickers(
            self.tensor,
            self.tickers,
            expected_count=expected,
            name=f"{self.split}_tensor",
        )
        if len(self.metadata) != expected:
            raise ValueError("Metadata and tensor row counts differ")
        if tuple(self.metadata["ticker"].astype(str)) != self.tickers:
            raise ValueError("Metadata ticker order does not match tensor order")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path) -> dict:
    path = root / "data/features/features_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_donor_parquet(split: str, repo_root: Path | str | None = None) -> dict:
    """Return the observed schema and certified lineage for one donor split."""

    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"Only donor splits are allowed, got {split!r}")
    root = Path(repo_root).resolve() if repo_root is not None else repository_root()
    relative = Path(f"data/features/windows/{split}.parquet")
    path = root / relative
    manifest = _manifest(root)
    expected_hash = manifest["checksums_sha256"][relative.as_posix()]
    frame = pd.read_parquet(path)
    lengths = sorted({int(np.asarray(value).size) for value in frame["features_flat"]})
    return {
        "path": relative.as_posix(),
        "sha256": _sha256(path),
        "expected_sha256": expected_hash,
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "features_flat_lengths": lengths,
        "channels": tuple(manifest["channels"]),
    }


def load_donor_windows(
    split: str,
    repo_root: Path | str | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    verify_hash: bool = True,
) -> DonorWindowBatch:
    """Load one certified donor split without accessing any target artifact."""

    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"Only donor splits are allowed, got {split!r}")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")

    root = Path(repo_root).resolve() if repo_root is not None else repository_root()
    relative = Path(f"data/features/windows/{split}.parquet")
    path = root / relative
    manifest = _manifest(root)
    validate_channel_order(manifest["channels"])

    expected_hash = manifest["checksums_sha256"][relative.as_posix()]
    actual_hash = _sha256(path)
    if verify_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Certified hash mismatch for {relative.as_posix()}: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    frame = pd.read_parquet(path)
    missing = set(_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required parquet columns: {sorted(missing)}")
    expected_count = EXPECTED_COUNTS[split]
    if len(frame) != expected_count:
        raise ValueError(f"{split} must contain {expected_count} rows, got {len(frame)}")
    if set(frame["split"].astype(str)) != {split}:
        raise ValueError(f"Unexpected split labels in {relative.as_posix()}")

    tickers = tuple(frame["ticker"].astype(str))
    if set(tickers) != DONOR_TICKERS:
        raise ValueError("The parquet does not contain exactly the certified donor universe")
    if frame[["split", "ticker", "window_start_date", "window_end_date"]].isna().any().any():
        raise ValueError("Window metadata contains missing values")
    if not (frame["window_start_date"] <= frame["window_end_date"]).all():
        raise ValueError("A window starts after its end date")

    rows: list[np.ndarray] = []
    expected_flat = WINDOW_LENGTH * INPUT_CHANNELS
    for row_number, value in enumerate(frame["features_flat"]):
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"features_flat row {row_number} is not numeric")
        if array.ndim != 1 or array.size != expected_flat:
            raise ValueError(
                f"features_flat row {row_number} must be a vector of length "
                f"{expected_flat}, got shape {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"features_flat row {row_number} contains NaN/Inf")
        rows.append(array.astype(np.float64, copy=False))

    matrix = np.stack(rows, axis=0).reshape(
        expected_count, WINDOW_LENGTH, INPUT_CHANNELS, order="C"
    )
    tensor = torch.from_numpy(matrix.copy()).to(dtype=dtype)
    metadata = frame.loc[:, _REQUIRED_COLUMNS[:-1]].copy()
    metadata.insert(0, "window_index", np.arange(expected_count, dtype=np.int64))

    batch = DonorWindowBatch(
        tensor=tensor,
        tickers=tickers,
        metadata=metadata,
        split=split,
        input_sha256=actual_hash,
    )
    batch.validate()
    return batch


def load_canonical_donor_tensors(
    repo_root: Path | str | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[DonorWindowBatch, DonorWindowBatch]:
    """Load donor train and validation in canonical row order."""

    train = load_donor_windows("donor_train", repo_root, dtype=dtype)
    validation = load_donor_windows("donor_validation", repo_root, dtype=dtype)
    if set(validation.tickers) - set(train.tickers):
        raise ValueError("Validation contains a donor unseen during training")
    return train, validation
