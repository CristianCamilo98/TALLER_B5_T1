#!/usr/bin/env python3
"""Audit official generator outputs against the common synthetic contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import baseline  # noqa: E402
import discovery  # noqa: E402
import report  # noqa: E402
import registry  # noqa: E402
import schema  # noqa: E402
import validation  # noqa: E402
from constants import (  # noqa: E402
    BASELINE_OUTPUT_PATH,
    NEURAL_METHOD_FAMILY,
    SIMPLE_BASELINE_FAMILY,
)


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
    if not args.skip_baseline:
        baseline.build_baseline()

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

    baseline_output = None
    baseline_row = None
    if BASELINE_OUTPUT_PATH.is_file():
        baseline_output = discovery.inspect_output(
            BASELINE_OUTPUT_PATH,
            generator_id="bootstrap_jitter",
        )
        baseline_frame = pd.read_parquet(BASELINE_OUTPUT_PATH)
        baseline_row = validation.validate_output(baseline_output, baseline_frame)

    report_rows = rows + ((baseline_row,) if baseline_row is not None else ())
    report.write_contract_report(discovery_result, report_rows)

    certified = []
    for output, row in zip(discovery_result.outputs, rows):
        if row.contract_status == "PASS":
            certified.append(
                registry.certified_entry(
                    output,
                    row,
                    method_family=NEURAL_METHOD_FAMILY,
                )
            )
    if baseline_output is not None and baseline_row is not None:
        if baseline_row.contract_status == "PASS":
            certified.append(
                registry.certified_entry(
                    baseline_output,
                    baseline_row,
                    method_family=SIMPLE_BASELINE_FAMILY,
                    method_id="bootstrap_jitter",
                    source_directory="common_pipeline/01_contract",
                )
            )
    registry.write_certified_registry(certified)

    exit_code = 0
    if not discovery_result.ok:
        exit_code = 1
    elif any(row.contract_status != "PASS" for row in report_rows):
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
