# DDPM training evidence

This directory contains the lightweight, tracked evidence for the three frozen DDPM training runs. It was copied from the local, ignored `generadores/daniel/artifacts/` tree; no model was retrained and no checkpoint or generated pool is published here.

## Frozen runs

| training seed | epochs completed | best epoch | best validation loss | stop reason |
|---:|---:|---:|---:|---|
| 42 | 33 | 13 | 0.5556680957476298 | `early_stopping_patience` |
| 123 | 32 | 12 | 0.5567662715911865 | `early_stopping_patience` |
| 2026 | 32 | 12 | 0.5573557813962301 | `early_stopping_patience` |

Each seed directory includes:

- `training_history.csv`: per-epoch train and validation losses, learning rate, and elapsed time;
- `training_manifest.json`: effective configuration, seeds, data lineage, selected epoch, losses, environment, and checkpoint hashes;
- `normalizer.json`: canonical donor-train-only normalization contract and statistics;
- `checkpoint_hashes.json`: SHA256 identities of the local best and last checkpoints (checkpoint binaries are intentionally not tracked);
- `final_pool_manifest.json`: provenance of the frozen generated pool and its selected checkpoint.

The repository-level training configuration is preserved as `diffusion.yaml`. The effective configuration for each run, including its seed override, is embedded in that run's `training_manifest.json`.

## Data and normalization provenance

All three runs record the canonical `donor_train` SHA256:

`5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f`

Normalization is fitted on `donor_train` only in `float64`, over axes `(0, 1)`, with `ddof=0`; standard deviations below `1e-8` are replaced by `1.0`, and network tensors are `float32`. The normalizer manifest SHA256 is identical for all runs:

`7e0fcce9c67d6a01581df4bed12e130555b164e7e1f846c39b25b4996eecef8e`

## Official seed-42 output

The official output remains:

`generadores/daniel/outputs/diffusion_seed42_normalized.parquet`

Its SHA256 is:

`bb9b5ad6b412fd785f73344cd765c56447b92de2b5827e40b3dc77d06e40a6c2`

`official_output_provenance.json` records the verified chain from training seed 42 and its selected checkpoint to the normalized 5,000-sample pool and official Parquet. The Parquet's `features_flat` values were compared with the local pool in synthetic-id order and are exactly equal (`max_abs_difference = 0.0`).

## Convergence figure

`figures/ddpm_training_convergence.png` plots the recorded train and validation loss for every epoch of each seed, with the selected best validation epoch marked. The figure was produced only from the tracked CSV histories and does not require retraining.
