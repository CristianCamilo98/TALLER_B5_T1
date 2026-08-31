"""Inspect physical-validity margins for one isolated utility run."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

io_synthetic = importlib.import_module("common_pipeline.03_utility.io_synthetic")
utility_run = importlib.import_module("common_pipeline.03_utility.utility_run")
validate_physical = importlib.import_module("common_pipeline.03_utility.validate_physical")
CHANNEL_ORDER = io_synthetic.CHANNEL_ORDER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=utility_run.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--results-root", type=Path, default=utility_run.RESULTS_ROOT)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_dir = utility_run.existing_run_dir(
        results_root=args.results_root,
        registry_path=args.registry_path,
        run_id=args.run_id,
    )
    mu, sigma = validate_physical.load_calibration(
        run_dir / "tables" / "nvda_calibration.csv"
    )
    threshold_z = -mu / sigma
    pools = io_synthetic.load_pools_for_run(run_dir)
    print(f"Umbral Z por canal: {dict(zip(CHANNEL_ORDER, threshold_z.round(2)))}")
    for method, pool in pools.items():
        print(f"--- {method} ---")
        for index, channel in enumerate(CHANNEL_ORDER):
            z_min = float(np.min(pool[:, :, index]))
            z_max = float(np.max(pool[:, :, index]))
            print(
                f"{channel:20s}: Z min={z_min:8.3f} Z max={z_max:8.3f} "
                f"margen={z_min - threshold_z[index]:6.3f}"
            )


if __name__ == "__main__":
    main()
