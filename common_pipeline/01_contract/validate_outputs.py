#!/usr/bin/env python3
"""Audit official generator outputs against the common synthetic contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import baseline  # noqa: E402
import discovery  # noqa: E402
import report  # noqa: E402
import schema  # noqa: E402
import validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only audit generator outputs; do not build bootstrap baseline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    discovery_result = discovery.discover_outputs()

    rows: tuple[validation.ContractReportRow, ...] = ()
    if discovery_result.outputs:
        schemas = {item.generator_id: item.columns for item in discovery_result.outputs}
        equivalent, schema_errors = schema.assess_schema_equivalence(schemas)
        if not equivalent:
            discovery_result = discovery.DiscoveryResult(
                ok=False,
                outputs=discovery_result.outputs,
                errors=discovery_result.errors + schema_errors,
            )
        rows = validation.validate_outputs(discovery_result.outputs)

    report.write_contract_report(discovery_result, rows)

    exit_code = 0
    if not discovery_result.ok:
        exit_code = 1
    elif any(row.contract_status != "PASS" for row in rows):
        exit_code = 1

    if not args.skip_baseline:
        baseline.build_baseline()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
