# Common phase 01 - certified output contract

This module is the only authority that decides whether a published synthetic
Parquet is eligible for the common experiment. It discovers
`generadores/*/outputs/*.parquet`, validates the canonical `(5000, 65, 3)`
schema and metadata, and records every PASS/FAIL reason in the contract report.
It never repairs, re-normalizes, or modifies a generator output.

The canonical interface for later phases is
`results/certified_outputs.json`. Only PASS outputs appear in that registry.
Each entry contains a stable method ID, family, repository-relative path,
SHA256, logical shape, seed, space, channel order, normalization status, and
contract status. Structural PASS is recorded separately from the official
experiment role (Cristian/WGAN-GP, Daniel/DDPM, Marco/VAE, David/Normalizing
Flow) and donor-lineage status. Phases 02 and 03 verify the registered file
and hash but do not infer missing metadata.

## Common simple baseline

The deterministic `bootstrap_jitter` baseline is generated from canonical
donor train only, with complete-window bootstrap sampling, seed 42, and
Gaussian jitter standard deviation 0.05. It uses the canonical Parquet schema
and enters the registry as `method_family = simple_baseline`; after
certification it follows the same fidelity and utility paths as neural methods.

## Commands

From the repository root:

```bash
python common_pipeline/01_contract/validate_outputs.py
python -m pytest -q common_pipeline/01_contract/tests
```

The validator checks the donor SHA before generating the baseline, writes the
complete PASS/FAIL report, and writes the certified registry. The registry may
be incomplete while owners correct outputs. Final runs in phases 02 and 03 are
strict and require all four official role slots plus the simple baseline.
Explicit partial runs select only satisfied official roles plus that baseline;
they never admit an arbitrary structurally certified method.
