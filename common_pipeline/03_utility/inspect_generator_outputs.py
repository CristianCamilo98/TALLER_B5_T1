"""Read-only inspection of the phase-01 certified registry."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

registry = importlib.import_module("common_pipeline.01_contract.registry")
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "01_contract/results/certified_outputs.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    payload = registry.load_certified_registry(args.registry_path)
    selected_payload, _selection = registry.select_experiment_methods(
        payload, allow_partial=args.allow_partial
    )
    for method in selected_payload["methods"]:
        print(
            f"{method['method_id']}: {method['path']} "
            f"{tuple(method['logical_shape'])} {method['method_family']}"
        )


if __name__ == "__main__":
    main()
