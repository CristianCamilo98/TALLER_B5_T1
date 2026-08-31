# Provisional Common Pipeline Results

## Status

**PROVISIONAL_PARTIAL — NOT FINAL**

**NOT FOR FINAL SCIENTIFIC CONCLUSIONS**

This curated snapshot records an integration and sanity run of the common
pipeline at Git commit
`343c30eb0424a12b3487667ea137251783c1e9c9`.

Included methods:

- Cristian — WGAN-GP
- Daniel — DDPM
- Marco — VAE
- common bootstrap+jitter baseline

Excluded method:

- David — Normalizing Flow pending

The snapshot demonstrates that the following common path works end to end:

```text
01_contract → 02_fidelity → 03_utility
```

It must not be used for:

- final conclusions;
- model selection;
- hyperparameter selection;
- the final presentation;
- definitive project figures or metrics.

The future strict final run will supersede these results after the fourth
official generator, David's Normalizing Flow, is available and certified.

## Contents

`fidelity/` contains the normalized-space fidelity snapshot:

- `manifest/`: the provisional evaluation manifest;
- `tables/`: 10 metric and diagnostic tables;
- `figures/`: 10 common fidelity figures.

`utility/` contains the downstream utility snapshot:

- `manifest/`: the isolated provisional utility run manifest;
- `tables/`: 6 calibration, validity, design, and evaluation tables;
- `figures/`: 5 common utility figures.

`SNAPSHOT_MANIFEST.csv` records the source path, snapshot path, and SHA256 of
every copied artifact. Large intermediate calibrated pools and all model/data
artifacts are deliberately excluded.
