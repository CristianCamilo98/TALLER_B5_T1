# Canonical experiment data

These five Parquet files are the frozen derived-data snapshot used by the common experiment. They are versioned in their canonical runtime paths so an academic evaluator can execute the common contract, fidelity and utility code from a clean clone without downloading market data or rebuilding splits and features.

They are not raw vendor data. Each row stores a derived feature window using the three common channels, in this order:

1. `log_return`;
2. `log_high_low_range`;
3. `log1p_volume`.

Each `features_flat` value has 195 elements and represents a `65 x 3` window. For the downstream test index, the same 65 sessions contain 60 context sessions followed by the 5-session target horizon.

## Frozen files

| Path | Rows | Logical shape | Official split | Stride | Purpose | SHA256 |
|---|---:|---|---|---:|---|---|
| `data/features/windows/donor_train.parquet` | 4,910 | `(4910, 65, 3)` | donors, 2012-01-03 through 2021-12-31 | 5 | generator training reference, common normalization, baseline and fidelity nearest-neighbour reference | `5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f` |
| `data/features/windows/donor_validation.parquet` | 380 | `(380, 65, 3)` | donors, 2022-01-03 through 2022-12-30 | 5 | held-out real reference for common fidelity | `134f51a2ac9e546bf1a2f21f4efbf56a62bf019a08de14209058563b0a88ae23` |
| `data/features/windows/nvda_visible.parquet` | 62 | `(62, 65, 3)` | NVDA visible, 2022-07-01 through 2022-12-30 | 1 | common NVDA calibration and real-only downstream reference | `0e6f046313b56b046d2e5f19d5cf0b7b8e0b81060a04620c2f5bb6a7b245f6d3` |
| `data/features/test_index.parquet` | 150 | `(150, 65, 3)` | NVDA test targets, 2023-01-03 through 2025-12-31 | 5 | common held-out downstream test | `64fe6f4c316d3746a6c28b233a3d4dddee587851163537bab79f1400a89be2c0` |
| `data/features/windows/nvda_full_history.parquet` | 2,703 | `(2703, 65, 3)` | NVDA full-history benchmark, 2012-01-03 through 2022-12-30 | 1 | benchmark only; not an input to common phases 01/02/03 | `19f651da100e6a304dda77831448d50a015e6eff5b9e019f6b3b2ffc6e908617` |

The exact files total 1,346,941 bytes. Their hashes are also recorded in `data/features/features_manifest.json` and `data/features/checksums.sha256`.

## Provenance boundary

The historical raw OHLCV snapshot is deliberately not versioned. Neither are the cleaned panel, daily split assignments, daily feature table, caches or other runtime Parquets. Their manifests, checksums, configuration and builders remain tracked as lineage and reconstruction evidence.

A new yfinance download can contain vendor corrections or adjusted-price changes and therefore may not reproduce the same bytes. The files listed above, with the stated SHA256 values, are the official frozen inputs for the final common experiment. Byte-exact reconstruction from the beginning of the data pipeline still requires the historical raw snapshot held outside Git.
