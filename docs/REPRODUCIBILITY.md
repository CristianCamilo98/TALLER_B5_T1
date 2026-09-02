# Reproducibility

This document defines the minimum delivery environment and the inputs needed to inspect the published generator outputs and run the common pipeline. It does not change the frozen scientific protocol.

## Environment

Python 3.12 is the recommended and tested delivery version for the data and common-pipeline environment. Historical manifests retain the Python version that actually produced each artifact and must not be rewritten.

The root environment intentionally excludes TensorFlow and PyTorch. Neither framework is imported by the data builders or common phases 01, 02, and 03. Install a generator-specific environment only when inspecting its framework-dependent code or reproducing its training.

## Installation

Create an isolated Python 3.12 environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, the equivalent commands are:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root requirements cover data acquisition/building, Parquet I/O, common fidelity and utility metrics and common tests: NumPy, pandas, PyArrow, PyYAML, yfinance, SciPy, scikit-learn, Matplotlib and pytest.

For generator-specific work, install the root requirements first and then the owner's requirements when present:

```bash
python -m pip install -r generadores/cristian/requirements.txt  # TensorFlow / WGAN-GP
python -m pip install -r generadores/daniel/requirements.txt    # PyTorch / DDPM
python -m pip install -r generadores/david/requirements.txt     # NumPy-only / Normalizing Flow
```

At this repository revision Marco does not have a tracked generator-specific requirements file. Do not infer Marco's final framework dependencies from the root environment.

## Canonical data

The common window contract is 65 timesteps by 3 channels (`log_return`, `log_high_low_range`, `log1p_volume`). The following exact snapshot is certified by `data/features/features_manifest.json` and `data/features/checksums.sha256`.

| Path | Rows | Exact size (bytes) | SHA256 | Tracked | Ignored | Required by |
|---|---:|---:|---|---|---|---|
| `data/features/windows/donor_train.parquet` | 4,910 | 976,450 | `5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f` | Yes | No | baseline, 01 contract, 02 fidelity |
| `data/features/windows/donor_validation.parquet` | 380 | 99,234 | `134f51a2ac9e546bf1a2f21f4efbf56a62bf019a08de14209058563b0a88ae23` | Yes | No | 02 fidelity |
| `data/features/windows/nvda_visible.parquet` | 62 | 9,406 | `0e6f046313b56b046d2e5f19d5cf0b7b8e0b81060a04620c2f5bb6a7b245f6d3` | Yes | No | 03 utility calibration and real-only reference |
| `data/features/test_index.parquet` | 150 | 55,515 | `64fe6f4c316d3746a6c28b233a3d4dddee587851163537bab79f1400a89be2c0` | Yes | No | 03 utility held-out test |
| `data/features/windows/nvda_full_history.parquet` | 2,703 | 206,336 | `19f651da100e6a304dda77831448d50a015e6eff5b9e019f6b3b2ffc6e908617` | Yes | No | full-history benchmark; not an input to 01/02/03 |

These five runtime/benchmark Parquets total 1,346,941 bytes and are tracked in their canonical paths, so they are present in a clean clone. They are frozen derived inputs, not raw OHLCV vendor data. Their complete snapshot boundary is documented in `data/CANONICAL_EXPERIMENT_DATA.md`.

### Rebuild chain and tracked evidence

| Path | Rows | Exact size (bytes) | Tracked | Ignored | Purpose / rebuild |
|---|---:|---:|---|---|---|
| `configs/experiment.yaml` | - | repository file | Yes | No | frozen source of split, feature and path configuration |
| `data/raw/download_manifest.json` | - | 5,976 | Yes | No | tracked acquisition metadata and raw checksums |
| `data/raw/ohlcv_raw.parquet` | 38,720 | 1,726,686 | No | Yes | exact raw snapshot; produced/reused by `scripts/download_ohlcv_raw.py` |
| `data/clean/clean_manifest.json` | - | 1,741 | Yes | No | tracked cleaning lineage |
| `data/clean/checksums.sha256` | - | 289 | Yes | No | tracked clean hashes |
| `data/clean/quality_report.csv` | - | 198 | Yes | No | tracked cleaning evidence |
| `data/clean/ohlcv_clean.parquet` | 38,720 | 1,726,703 | No | Yes | rebuilt by `scripts/clean_ohlcv.py` from the exact raw snapshot |
| `data/splits/split_manifest.json` | - | 2,689 | Yes | No | tracked split rules and counts |
| `data/splits/checksums.sha256` | - | 213 | Yes | No | tracked split hashes |
| `data/splits/daily_split_report.csv` | - | 295 | Yes | No | tracked split evidence |
| `data/splits/daily_split_assignments.parquet` | 38,720 | 266,655 | No | Yes | rebuilt by `scripts/assign_splits.py` |
| `data/features/features_manifest.json` | - | 2,991 | Yes | No | tracked feature/window contract and hashes |
| `data/features/checksums.sha256` | - | 760 | Yes | No | tracked canonical feature hashes |
| `data/features/window_counts.csv` | - | 298 | Yes | No | tracked window-count evidence |
| `data/features/daily_features_by_split.parquet` | 31,431 | 1,129,643 | No | Yes | rebuilt with all canonical windows by `scripts/build_features_windows.py` |

The rebuild commands are documented in the root README and the READMEs under `data/`. A live yfinance download can rebuild a methodologically equivalent dataset, but vendor corrections or adjusted-price changes can alter bytes and hashes. Exact reconstruction therefore requires the certified raw snapshot, not merely network access.

Only the five small canonical runtime/benchmark Parquets are versioned. Raw OHLCV, the cleaned panel, daily split assignments and daily features remain ignored. Exact end-to-end reconstruction still requires the historical raw snapshot outside Git; a new live download is not hash-stable.

## Common pipeline

All commands run from the repository root. Phase 01 is the only contract authority: it validates tracked generator outputs, builds the deterministic baseline from canonical `donor_train`, and refreshes the certified registry.

```bash
python common_pipeline/01_contract/validate_outputs.py
```

### STRICT_FINAL

The default mode is the only final scientific run. It requires all four official generator roles (WGAN-GP, DDPM, VAE and Normalizing Flow) plus the common simple baseline.

```bash
python common_pipeline/02_fidelity/evaluate_fidelity.py
python common_pipeline/02_fidelity/plot_fidelity.py

python -m common_pipeline.03_utility.calibrate_nvda
python -m common_pipeline.03_utility.validate_physical
python -m common_pipeline.03_utility.build_mixtures
python -m common_pipeline.03_utility.downstream_ridge
python -m common_pipeline.03_utility.plot_utility
python -m common_pipeline.03_utility.interpretation_summary
```

`STRICT_FINAL` has been completed: phase 01 certifies all four official roles (WGAN-GP, DDPM, VAE, Normalizing Flow) plus the common simple baseline at commit `2bef27364907d9ae8e01b4bfea1673749a1d8488`, and phases 02 and 03 have each been run exactly once against that certified registry. The frozen scientific state is tagged `strict-final-20260902` (commit `d5b702269b0b28d7fcf965ac1e1659299402ec1f`).

Re-running the commands above regenerates the same kind of local runtime outputs described below; it does not change or supersede the tagged, tracked result snapshot.

### Runtime results vs. the tracked final snapshot

Running phases 02 and 03 writes to local, mostly gitignored runtime locations:

- `common_pipeline/02_fidelity/results/` — manifest, tables and figures for the most recent fidelity run (tables/figures/manifest are gitignored).
- `common_pipeline/03_utility/results/runs/<run_id>/` — one isolated, content-addressed directory per utility run (fully gitignored); `<run_id>` defaults to `utility_<certified-registry-sha256[:12]>`.

These runtime directories are not present, or are empty, in a fresh clone, and re-running the commands overwrites their previous content in place.

The **authoritative, tracked, byte-exact copy of the STRICT_FINAL results** is versioned separately and is present in a fresh clone:

```
artifacts/final/strict_final_20260902/
├── MANIFEST.json      # SHA256 of every copied file, source path, provenance
├── README.md
├── fidelity/          # copy of the STRICT_FINAL 02 tables, figures, manifest
└── utility/           # copy of the STRICT_FINAL 03 tables, figures, manifest (run utility_7308c264cbd0)
```

Use `artifacts/final/strict_final_20260902/` to inspect the final numbers without running anything. Use the runtime locations above only when actually re-executing phases 02/03 locally.

### PROVISIONAL_PARTIAL

Partial execution is integration evidence only, never a final result. It selects satisfied official roles plus the common baseline and excludes wrong-role outputs. Run phase 01 first, then explicitly opt in at the selection entry points:

```bash
python common_pipeline/02_fidelity/evaluate_fidelity.py --allow-partial
python common_pipeline/02_fidelity/plot_fidelity.py

python -m common_pipeline.03_utility.calibrate_nvda --allow-partial
python -m common_pipeline.03_utility.validate_physical
python -m common_pipeline.03_utility.build_mixtures
python -m common_pipeline.03_utility.downstream_ridge
python -m common_pipeline.03_utility.plot_utility
python -m common_pipeline.03_utility.interpretation_summary
```

The registry and every downstream phase verify input freshness and hashes. Do not use a stale registry after generator outputs change.

## Generator outputs

Published Parquets under `generadores/*/outputs/` and the common baseline under `common_pipeline/01_contract/outputs/` are tracked and can be inspected from a clean clone without retraining. Their presence does not by itself make them eligible: phase 01 validates schema, source model, official role, normalization provenance and donor lineage.

The tracked `common_pipeline/01_contract/results/` files are a snapshot of the registry/report at their producing commit. Refresh phase 01 after any official output changes and before a new common run.

## Training reproduction

Training is separate from delivery-level common-pipeline reproduction:

- install the root environment, then the generator-specific requirements;
- use the owner's training configuration, seed, history and provenance;
- do not retrain merely to inspect a tracked official output;
- checkpoints and large intermediate pools are not required by phases 01/02/03.

Cristian's WGAN-GP and Daniel's DDPM have tracked lightweight evidence under `generadores/cristian/evidence/` and `generadores/daniel/evidence/`: histories, manifests, normalizer/provenance, checkpoint SHA256 identities, the effective configuration and a convergence figure. David's Normalizing Flow has equivalent tracked evidence directly under `generadores/david/artifacts/` (checkpoint, loss history, training manifest, convergence figure) plus a dedicated integrity test. The checkpoint binaries and redundant NPZ pools remain local/ignored for all owners. Marco does not yet have an equivalent tracked evidence folder at this revision.

In the full development environment used to produce and certify the STRICT_FINAL run (root requirements plus every generator's own requirements installed), `python -m pytest -q` reports 149 passed tests.

## Clean-clone capability

| Component | From clone with root requirements only | Additional requirement |
|---|---|---|
| Read code, configs, manifests and documentation | Yes | None |
| Inspect tracked official synthetic outputs | Yes | None |
| Run data/common unit tests that use only fixtures | Yes | None |
| Run the complete repository test suite | No | generator-specific frameworks and the ignored clean/split/daily-feature snapshots used by historical certification tests |
| Run 01 contract and baseline | Yes | tracked `donor_train`; phase 01 still enforces output eligibility |
| Run 02 fidelity | Yes | tracked donor data, fresh phase-01 registry and eligible outputs |
| Run 03 utility | Yes | tracked NVDA data, fresh phase-01 registry and eligible outputs |
| Inspect `nvda_full_history` benchmark | Yes | None |
| Rebuild the exact certified data snapshot | No | exact ignored raw snapshot; live download is not hash-stable |
| Reproduce generator training | No | owner-specific dependencies, data and any required local checkpoint/training artifacts |

## Known limitations

- The exact raw snapshot is local/ignored. Network reconstruction may produce a different lineage.
- Historical root certification tests that inspect the clean panel, daily split assignments or daily feature table still require those local ignored intermediates; common phases 01/02/03 and their tests use the five tracked canonical inputs.
- TensorFlow and PyTorch are deliberately absent from root requirements, so a root-only environment does not run framework-specific generator tests or training.
- Marco has no tracked generator-specific requirements file at this revision.
- Large checkpoints and redundant training pools are not versioned. Lightweight evidence and hashes are sufficient for inspecting frozen outputs, but full training/sampling replay may require owner-held artifacts.
- `STRICT_FINAL` has been completed (see above); this document does not claim bit-exact retraining reproducibility for any generator beyond what each owner's evidence explicitly documents.
