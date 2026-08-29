"""Inspect only the two certified donor window artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.data_adapter import (  # noqa: E402
    inspect_donor_parquet,
    load_canonical_donor_tensors,
)


def main() -> None:
    inspections = {
        split: inspect_donor_parquet(split, REPOSITORY_ROOT)
        for split in ("donor_train", "donor_validation")
    }
    train, validation = load_canonical_donor_tensors(REPOSITORY_ROOT)
    report = {
        "inspections": inspections,
        "tensor_shapes": {
            "donor_train": list(train.tensor.shape),
            "donor_validation": list(validation.tensor.shape),
        },
        "ticker_counts": {
            "donor_train": len(set(train.tickers)),
            "donor_validation": len(set(validation.tickers)),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
