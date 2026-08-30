# Daniel Diffusion — isolated experimental module

This directory contains Daniel's DDPM baseline and its local tests. It is
intentionally isolated from the common core. The permanent model code accepts
only tensors shaped `(N, 65, 3)`; it does not know about parquet files,
tickers, dates, NVDA, downstream models, or test data.

`src/data_adapter.py` is the local boundary adapter. The normalizer follows the
team's global per-channel contract: donor-train is loaded in float64, one mean
and population standard deviation are fitted per channel over all windows and
sessions (`axes=(0, 1)`, `ddof=0`), and the normalized DDPM tensors are then
converted to float32. Validation is transform-only. A standard deviation below
`1e-8` is replaced by `1.0`.

The frozen channel order is:

1. `log_return`
2. `log_high_low_range`
3. `log1p_volume`

The local runtime requires PyTorch in addition to the repository's common
dependencies. On Windows, the audited commands use Python 3.12:

```powershell
py -3.12 -m pip install -r generadores/daniel/requirements.txt
py -3.12 -m pytest -q generadores/daniel/tests
py -3.12 generadores/daniel/scripts/inspect_inputs.py
py -3.12 generadores/daniel/scripts/smoke_test.py
py -3.12 generadores/daniel/scripts/tiny_overfit.py --steps 150
python generadores/daniel/scripts/train.py --seed 42
python generadores/daniel/scripts/train.py --seed 123
python generadores/daniel/scripts/train.py --seed 2026
python generadores/daniel/scripts/plot_training.py --run-id diffusion_seed42_global_channel
python generadores/daniel/scripts/verify_frozen_runs.py
python generadores/daniel/scripts/summarize_frozen_seeds.py
python generadores/daniel/scripts/sample_diagnostics.py
python generadores/daniel/scripts/generate_final_pools.py --seed 42
python generadores/daniel/scripts/generate_final_pools.py --seed 123
python generadores/daniel/scripts/generate_final_pools.py --seed 2026
python generadores/daniel/scripts/summarize_final_pools.py
python generadores/daniel/scripts/evaluate_visual_diagnostics.py
python generadores/daniel/scripts/plot_training_diagnostics.py --run-id diffusion_seed42_global_channel
```

The smoke test performs one optimizer update and mechanical sampling only. The
tiny-overfit diagnostic uses 32 donor-train windows and fixed noise/timesteps;
neither command performs full training or accesses any NVDA artifact.

The final training entrypoint refuses a dirty Git worktree, rejects any
configuration drift from `config/diffusion.yaml`, and accepts only seeds 42,
123, and 2026. The runtime override changes only `reproducibility.seed`; all
other model, diffusion, training, normalization, and validation settings are
frozen. Training timesteps and noise remain stochastic under the run seed.
Validation is ordered and receives a
dedicated generator that is reinitialized to seed `424242` at every epoch;
therefore each validation row sees the same timestep/noise realization across
epochs. Validation runs under `no_grad`, never updates the optimizer, and never
participates in normalization fit.

`sample_diagnostics.py` is a preliminary, generator-local normalized-space
diagnostic. It creates exactly 1,000 samples, checks deterministic sampling,
and compares them with normalized donor validation. Its Wasserstein, ACF,
correlation, nearest-neighbour, and diversity outputs are not the future common
fidelity evaluation and must not be compared as final cross-generator scores.

Final-pool generation uses each global-normalized best checkpoint without
retraining. The
temporary NVDA calibrator reads only unique daily rows from the canonical
`nvda_visible` feature block and applies `mean + std * normalized_sample` with
population statistics. Physical validation rejects an entire window; it never
clips, repairs, takes absolute values, or uses a donor inverse transform. These
calibration and plotting utilities are local preparation for the future common
fidelity phase, not a replacement for it.

Generated checkpoints, histories, samples, manifests, and figures belong
under `artifacts/` and are excluded locally through `.git/info/exclude`.

## Artifact lineage

Artifacts named `diffusion_seed*_frozen`, `diffusion_seed*_normalized_5000`,
or `diffusion_seed*_nvda_like_5000` were produced before normalization
alignment and are retained locally as `LEGACY_PER_TICKER`. Current artifacts
use `global_channel` in every run, pool, manifest, table, and figure name and
form the `CURRENT_GLOBAL_CHANNEL` lineage. Scripts never overwrite the legacy
names.

## Daniel-only visual diagnostics

`evaluate_visual_diagnostics.py` compares normalized donor validation with a
deterministic, balanced subset of the global-channel seed-42 normalized pool. The
marginal panels describe channel centre, dispersion, and tails, but not
temporal dependence. The logistic classifier is a diagnostic rather than part
of the generator: its ROC-AUC and accuracy use five-fold out-of-fold
predictions. AUC near 0.50 means only that this linear classifier has little
distinguishing ability; it does not prove identical distributions. The joint
PCA/t-SNE view is likewise descriptive and seed/parameter dependent, so it is
not a model score or ranking. These local figures do not replace common
fidelity evaluation.

## Seed-42 long-training diagnostic (not run during normalization alignment)

`train_long_diagnostic.py` can derive a separate experiment from the current
configuration. Its only changes are `max_epochs: 200 -> 300` and
`early_stopping_patience: 20 -> 30`; it does not impose a minimum epoch count.
All data, normalization, model, diffusion, optimizer, and validation settings
remain frozen. Outputs use the distinct run ID
`diffusion_seed42_long_training_diagnostic` and cannot overwrite the certified
seed-42 run.

Training loss may continue to decrease while validation loss rises. That
pattern is compatible with overfitting and validation need not improve
monotonically. Early stopping therefore preserves the checkpoint with minimum
deterministic validation loss, not the last epoch. Even if the diagnostic finds
a later minimum, replacing the certified frozen checkpoint remains a separate
manual project decision.
