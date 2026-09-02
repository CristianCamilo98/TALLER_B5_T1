# STRICT_FINAL results snapshot (2026-09-02)

This directory is a curated, tracked, immutable copy of the project's
**STRICT_FINAL** scientific results. It is the material used for the
project's analysis, README, and presentation work. It contains no
provisional results and none of its contents were recalculated,
re-plotted, or re-serialized to produce this copy — every file here is a
byte-exact copy of a file already produced by the certified pipeline runs
(see `MANIFEST.json` for source paths and SHA256 of every file).

It does not replace or modify the original runtime results under
`common_pipeline/02_fidelity/results/` and
`common_pipeline/03_utility/results/runs/utility_7308c264cbd0/`, which
remain the authoritative runtime sources and stay documented in the
manifests copied here (`fidelity/evaluation_manifest.json` and
`utility/run_manifest.json`).

## Provenance

- Certified registry (phase 01) SHA256:
  `7308c264cbd032f82db456b4dd2bdef749aae8c45bfdce80bdca3a9da23aa256`
- Master commit the STRICT_FINAL run was executed against:
  `2bef27364907d9ae8e01b4bfea1673749a1d8488`
- Methods (5): `bootstrap_jitter` (simple baseline), `cristian` (WGAN-GP),
  `daniel` (DDPM), `marco` (VAE), `david` (Normalizing Flow) — 4 neural
  generators + 1 simple baseline, all structurally PASS under phase 01.

## What "real" means in each phase — do not conflate them

**02 fidelity** (`fidelity/`): "real" = 380 held-out **donor_validation**
windows (donor market data, not NVDA). Every method's synthetic pool of
5000 windows is subset to the same 380 row positions (evaluation subset
seed 42) before comparison against those 380 real windows. This phase
never uses NVDA.

**03 utility** (`utility/`): `REAL_ONLY` = 62 visible NVDA training
windows (2022-07-01 to 2022-12-30, calibrated from 126 unique daily NVDA
observations). The held-out test set is 150 NVDA windows from 2023-2025.
Ratios 25/50/75% add 21/62/186 synthetic windows (subsampling seeds 42,
123, 2026) on top of the 62 real windows; the same Ridge(alpha=1.0)
downstream model and the same 150-window test are used for every method,
including the baseline.

## Contents

```
strict_final_20260902/
├── README.md                      (this file)
├── MANIFEST.json                  (delivery metadata: SHA256 of every file, not a new scientific result)
├── fidelity/
│   ├── evaluation_manifest.json   (copy of the 02 STRICT_FINAL run manifest)
│   ├── tables/                    (10 CSV: marginals, Wasserstein-1, return/abs-return ACF,
│   │                                channel correlations + errors, nearest-neighbour,
│   │                                C2ST, joint PCA/t-SNE coordinates, evaluation subset indices)
│   └── figures/                   (10 PNG: marginals per channel, ACF curves, correlation heatmaps,
│                                    nearest-neighbour, C2ST AUC, joint t-SNE, fidelity scorecard)
└── utility/
    ├── run_manifest.json          (copy of the 03 STRICT_FINAL run manifest, run_id=utility_7308c264cbd0)
    ├── tables/                    (6 CSV: NVDA calibration, physical validity, mixture design,
    │                                downstream raw results (46 rows), downstream summary (16 rows),
    │                                interpretation summary)
    └── figures/                   (5 PNG: physical rejection rates, RMSE/MAE vs ratio, RMSE
                                     improvement heatmap, utility summary)
```

The calibrated-pool `.npz` caches from the utility run are intentionally
**not** included: they are runtime/intermediate artifacts (calibrated
synthetic tensors + validity masks used internally between pipeline
steps), fully reproducible from the certified registry and not needed to
read or audit the published tables and figures.

## Scope of this snapshot

- No scientific code was changed to produce this snapshot.
- No methodology, threshold, feature, seed, or model was changed.
- No new metric was computed; every number here already existed in the
  STRICT_FINAL runtime outputs before this copy was made.
- This snapshot does not contain conclusions, rankings, or a declared
  "winner" — that interpretation work has not been done yet.
