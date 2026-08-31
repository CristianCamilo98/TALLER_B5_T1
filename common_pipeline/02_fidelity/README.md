# Common synthetic fidelity

This module evaluates every certified method with one normalized-space
implementation. Phase 01 is the sole contract authority: phase 02 reads
`common_pipeline/01_contract/results/certified_outputs.json` and never
discovers or approves generator files independently.

By default a run fails fast unless all four official generator roles and the
simple baseline are satisfied. `--allow-partial` runs only the satisfied
official roles plus the baseline and marks its manifest
`PROVISIONAL_PARTIAL`; it never admits a wrong-role certified output and must
not be used for final scientific results.

## Common protocol

- Real reference: 380 donor-validation windows.
- Normalizer: float64 global channel statistics fit on donor train only;
  validation is transform-only and converted to float32.
- Synthetic subset: the same 380 row positions for every certified method,
  drawn without replacement with evaluation subset seed 42.
- No NVDA calibration, physical filtering, clipping, repair, winsorization, or
  synthetic re-standardization.
- Metrics: marginal statistics, Wasserstein-1, within-window return/absolute
  return ACF, channel correlations, nearest-neighbour distance, out-of-fold
  logistic C2ST, and one joint PCA+t-SNE embedding.

The baseline has no special evaluation path after certification. Plotting
reloads exactly the methods and row positions recorded by the evaluation
manifest.

## Commands

```bash
python common_pipeline/02_fidelity/evaluate_fidelity.py
python common_pipeline/02_fidelity/plot_fidelity.py
python -m pytest -q common_pipeline/02_fidelity/tests
```

For an explicitly provisional, isolated run only:

```bash
python common_pipeline/02_fidelity/evaluate_fidelity.py --allow-partial
```

Runtime tables and figures must be regenerated only when the strict registry
is complete. Metrics remain separate; no arbitrary composite score is used.
