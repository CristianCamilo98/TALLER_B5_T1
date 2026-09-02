# Common pipeline

Three phases, run strictly in order, shared by all four generators and the
common simple baseline. Each phase has its own detailed README
(`01_contract/README.md`, `02_fidelity/README.md`, `03_utility/README.md`);
this page is only an index.

```
01_contract  →  02_fidelity  →  03_utility
```

## 01_contract

The only authority that certifies whether a published generator Parquet is
eligible for the common experiment: structural schema, official role
(WGAN-GP / DDPM / VAE / Normalizing Flow), and donor-lineage status. Writes
the certified registry consumed by phases 02 and 03.

## 02_fidelity

Compares each certified method's synthetic windows against **held-out
`donor_validation` real windows** — this is donor market data, **not NVDA**.
Marginals, Wasserstein-1, ACF, channel correlations, nearest-neighbour and
C2ST. Diagnostic and comparative only, never used for model selection.

## 03_utility

Measures the downstream impact of synthetic augmentation on NVDA volatility
forecasting. `REAL_ONLY` = Ridge trained on the 62 visible NVDA windows only;
augmented runs add 25/50/75% synthetic windows from each certified method.
Every configuration is evaluated on the same held-out **150 NVDA test
windows (2023-2025)**.

## Where the final results are

The authoritative, tracked, byte-exact STRICT_FINAL results for both phases
are published under
[`artifacts/final/strict_final_20260902/`](../artifacts/final/strict_final_20260902/README.md),
frozen at git tag `strict-final-20260902`. The `results/` folders inside
`02_fidelity/` and `03_utility/` hold local runtime output when you actually
re-run the phases; see [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)
for the full distinction and the exact commands.
