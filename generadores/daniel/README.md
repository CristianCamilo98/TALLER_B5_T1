# Daniel Diffusion — isolated experimental module

This directory contains Daniel's DDPM baseline and its local tests. It is
intentionally isolated from the common core. The permanent model code accepts
only tensors shaped `(N, 65, 3)`; it does not know about parquet files,
tickers, dates, NVDA, downstream models, or test data.

`src/data_adapter.py` and `src/temporary_normalizer.py` are temporary boundary
adapters. They must be replaced when the project publishes the common loader
and normalizer. The temporary normalizer is fitted only on donor-train values,
with one mean and population standard deviation per ticker and channel.

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
python generadores/daniel/scripts/plot_training.py --run-id diffusion_seed42_frozen
python generadores/daniel/scripts/verify_frozen_runs.py
python generadores/daniel/scripts/summarize_frozen_seeds.py
python generadores/daniel/scripts/sample_diagnostics.py
python generadores/daniel/scripts/generate_final_pools.py --seed 42
python generadores/daniel/scripts/generate_final_pools.py --seed 123
python generadores/daniel/scripts/generate_final_pools.py --seed 2026
python generadores/daniel/scripts/summarize_final_pools.py
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

Final-pool generation uses each frozen best checkpoint without retraining. The
temporary NVDA calibrator reads only unique daily rows from the canonical
`nvda_visible` feature block and applies `mean + std * normalized_sample` with
population statistics. Physical validation rejects an entire window; it never
clips, repairs, takes absolute values, or uses a donor inverse transform. These
calibration and plotting utilities are local preparation for the future common
fidelity phase, not a replacement for it.

Generated checkpoints, histories, samples, manifests, and figures belong
under `artifacts/` and are excluded locally through `.git/info/exclude`.
