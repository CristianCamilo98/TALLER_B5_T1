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
python generadores/daniel/scripts/train.py
python generadores/daniel/scripts/plot_training.py
```

The smoke test performs one optimizer update and mechanical sampling only. The
tiny-overfit diagnostic uses 32 donor-train windows and fixed noise/timesteps;
neither command performs full training or accesses any NVDA artifact.

The real seed-42 entrypoint refuses a dirty Git worktree and rejects any
configuration drift from `config/diffusion.yaml`. Training timesteps and noise
remain stochastic under the run seed. Validation is ordered and receives a
dedicated generator that is reinitialized to seed `424242` at every epoch;
therefore each validation row sees the same timestep/noise realization across
epochs. Validation runs under `no_grad`, never updates the optimizer, and never
participates in normalization fit.

Generated checkpoints, histories, samples, manifests, and figures belong
under `artifacts/` and are excluded locally through `.git/info/exclude`.
