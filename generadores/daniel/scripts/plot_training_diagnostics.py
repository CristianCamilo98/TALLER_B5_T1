"""Generate full/zoom training plots for one persisted Daniel run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.training_diagnostics import plot_training_views  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    history = artifact_root / "histories" / f"{args.run_id}.csv"
    prefix = args.output_prefix or args.run_id
    full = artifact_root / "figures" / f"{prefix}_training_full.png"
    zoom = artifact_root / "figures" / f"{prefix}_training_zoom.png"
    result = plot_training_views(history, full, zoom, run_id=args.run_id)
    print(f"full={full}")
    print(f"zoom={zoom}")
    print(result)


if __name__ == "__main__":
    main()
