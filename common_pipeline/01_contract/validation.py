"""Structural validation for one discovered synthetic output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from constants import (
    CHANNEL_ORDER,
    EXPECTED_ROWS,
    EXPECTED_TRAINING_SEED,
    FEATURE_DIM,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from discovery import DiscoveredOutput
from io_utils import reconstruct_tensor, stack_features
from normalizer import assess_normalization_provenance
from schema import assess_schema


@dataclass(frozen=True)
class ContractReportRow:
    generator: str
    file: str
    sha256: str
    rows: int
    logical_shape: str
    training_seed: int | None
    space: str | None
    channel_order: str
    finite: str
    exact_duplicates: int
    normalization_provenance: str
    contract_status: str
    errors: tuple[str, ...]


def _normalize_channel_order(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return tuple(str(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip("[]")
        parts = [part.strip().strip("'\"") for part in stripped.split(",") if part.strip()]
        return tuple(parts)
    return None


def _count_exact_duplicates(tensor: np.ndarray) -> int:
    flat = tensor.reshape(len(tensor), -1)
    _, counts = np.unique(flat, axis=0, return_counts=True)
    return int((counts > 1).sum())


def _unique_channel_orders(frame: pd.DataFrame) -> list[tuple[str, ...] | None]:
    if "channel_order" not in frame.columns:
        return []
    return [_normalize_channel_order(value) for value in frame["channel_order"].dropna()]


def validate_output(output: DiscoveredOutput, frame: pd.DataFrame) -> ContractReportRow:
    errors: list[str] = []

    schema = assess_schema(tuple(frame.columns))
    if not schema.ok:
        errors.extend(schema.errors)

    if len(frame) != EXPECTED_ROWS:
        errors.append(f"rows={len(frame)}, expected {EXPECTED_ROWS}")

    training_seed = None
    if "training_seed" in frame.columns:
        seeds = frame["training_seed"].dropna().unique()
        if len(seeds) != 1:
            errors.append(f"training_seed not unique: {seeds.tolist()}")
        else:
            training_seed = int(seeds[0])
            if training_seed != EXPECTED_TRAINING_SEED:
                errors.append(f"training_seed={training_seed}, expected {EXPECTED_TRAINING_SEED}")

    space = None
    if "space" in frame.columns:
        spaces = frame["space"].dropna().unique()
        if len(spaces) != 1:
            errors.append(f"space not unique: {spaces.tolist()}")
        else:
            space = str(spaces[0])
            if space != GLOBAL_NORMALIZED_SPACE:
                errors.append(f"space={space!r}, expected {GLOBAL_NORMALIZED_SPACE!r}")

    if "window_length" in frame.columns:
        values = frame["window_length"].dropna().unique()
        if len(values) != 1 or int(values[0]) != WINDOW_LENGTH:
            errors.append(f"window_length invalid: {values.tolist()}")

    if "n_channels" in frame.columns:
        values = frame["n_channels"].dropna().unique()
        if len(values) != 1 or int(values[0]) != N_CHANNELS:
            errors.append(f"n_channels invalid: {values.tolist()}")

    channel_order_text = "unknown"
    if "channel_order" in frame.columns:
        parsed_orders = _unique_channel_orders(frame)
        unique_orders = {order for order in parsed_orders if order is not None}
        if len(unique_orders) != 1:
            errors.append(f"channel_order not unique: {[list(order) for order in unique_orders]}")
        else:
            parsed = next(iter(unique_orders))
            channel_order_text = str(list(parsed))
            if parsed != CHANNEL_ORDER:
                errors.append(f"channel_order={parsed}, expected {CHANNEL_ORDER}")

    if "synthetic_id" in frame.columns and frame["synthetic_id"].duplicated().any():
        errors.append("synthetic_id contains duplicates")

    feature_lengths = frame["features_flat"].map(lambda values: len(values))
    if not feature_lengths.eq(FEATURE_DIM).all():
        errors.append("features_flat length is not 195 for all rows")

    try:
        tensor = stack_features(frame)
    except ValueError as exc:
        errors.append(str(exc))
        tensor = np.empty((0, WINDOW_LENGTH, N_CHANNELS))

    finite = "NO"
    exact_duplicates = 0
    logical_shape = "(invalid)"
    if tensor.size:
        finite = "YES" if np.isfinite(tensor).all() else "NO"
        if finite == "NO":
            errors.append("tensor contains NaN or Inf")
        exact_duplicates = _count_exact_duplicates(tensor)
        if exact_duplicates:
            errors.append(f"found {exact_duplicates} exact duplicate windows")
        logical_shape = f"({tensor.shape[0]},{WINDOW_LENGTH},{N_CHANNELS})"
        if tensor.shape != (len(frame), WINDOW_LENGTH, N_CHANNELS):
            errors.append(f"tensor shape={tensor.shape}, expected ({len(frame)}, {WINDOW_LENGTH}, {N_CHANNELS})")

        for idx, values in enumerate(frame["features_flat"]):
            direct = np.asarray(values, dtype=np.float64).reshape(WINDOW_LENGTH, N_CHANNELS)
            if not np.array_equal(direct, tensor[idx]):
                errors.append(f"row {idx}: C-order/session-major reconstruction mismatch")
                break

    provenance = assess_normalization_provenance(output.generator_id)
    if provenance == "NORMALIZATION_MISMATCH":
        errors.append("normalization provenance indicates mismatch with canonical normalizer")

    status = "PASS" if not errors else "FAIL"
    return ContractReportRow(
        generator=output.generator_id,
        file=output.filename,
        sha256=output.sha256,
        rows=int(len(frame)),
        logical_shape=logical_shape,
        training_seed=training_seed,
        space=space,
        channel_order=channel_order_text,
        finite=finite,
        exact_duplicates=exact_duplicates,
        normalization_provenance=provenance,
        contract_status=status,
        errors=tuple(errors),
    )


def validate_outputs(outputs: tuple[DiscoveredOutput, ...]) -> tuple[ContractReportRow, ...]:
    rows: list[ContractReportRow] = []
    for output in outputs:
        frame = pd.read_parquet(output.path)
        rows.append(validate_output(output, frame))
    return tuple(rows)
