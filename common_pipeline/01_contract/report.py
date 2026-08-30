"""Write contract audit reports."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from constants import CONTRACT_REPORT_CSV, CONTRACT_REPORT_JSON
from discovery import DiscoveryResult
from io_utils import write_json
from validation import ContractReportRow


def _row_dict(row: ContractReportRow) -> dict:
    payload = asdict(row)
    payload.pop("errors")
    return payload


def write_contract_report(
    discovery: DiscoveryResult,
    rows: tuple[ContractReportRow, ...],
) -> tuple[pd.DataFrame, dict]:
    report_rows = []
    for row in rows:
        report_rows.append(_row_dict(row))

    if not discovery.ok:
        for error in discovery.errors:
            report_rows.append(
                {
                    "generator": "discovery",
                    "file": "",
                    "sha256": "",
                    "rows": 0,
                    "logical_shape": "",
                    "training_seed": None,
                    "space": None,
                    "channel_order": "",
                    "finite": "",
                    "exact_duplicates": 0,
                    "normalization_provenance": "",
                    "contract_status": "FAIL",
                    "discovery_error": error,
                }
            )

    frame = pd.DataFrame(report_rows)
    CONTRACT_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CONTRACT_REPORT_CSV, index=False)

    payload = {
        "discovery_ok": discovery.ok,
        "discovery_errors": list(discovery.errors),
        "generators": [
            {
                **_row_dict(row),
                "errors": list(row.errors),
            }
            for row in rows
        ],
    }
    write_json(CONTRACT_REPORT_JSON, payload)
    return frame, payload
