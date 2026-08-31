# Common synthetic utility pipeline

This module consumes only the phase-01 certified registry. Method selection is
driven by the central phase-01 official-role policy. The default final
behavior is strict: WGAN-GP, DDPM, VAE, Normalizing Flow, and the certified
simple baseline are required. `--allow-partial` explicitly creates a
`PROVISIONAL_PARTIAL` run containing only currently satisfied official roles
plus the baseline; wrong-role outputs remain excluded.

## Run isolation

Calibration creates `results/runs/<run_id>/run_manifest.json`. The manifest
pins the registry SHA, input Parquet SHAs, methods, NVDA calibration source,
ratios, subsampling seeds, feature-scaler policy, and downstream model. Every
later stage reads only paths registered inside that run; no global cache glob
is used. Historical versioned tables/figures are not inputs to this flow.

## Scientific protocol (unchanged)

- One common affine calibration: `X = mu_NVDA + sigma_NVDA * Z`, using 126
  unique visible NVDA days and population standard deviation.
- Post-calibration whole-window physical validation, with no repair.
- One `REAL_ONLY` result: 62 real windows, zero synthetic windows, one RMSE
  and one MAE. Every augmented result uses this same delta reference.
- Ratios above zero use 21/62/186 synthetic windows for 25/50/75 percent,
  sampled without replacement with seeds 42, 123, and 2026.
- One downstream StandardScaler fit only on the 62 real visible windows.
- Common eight-feature transformation, Ridge(alpha=1.0), and the same held-out
  NVDA test for every certified method, including the simple baseline.

## Commands

```bash
python -m common_pipeline.03_utility.calibrate_nvda
python -m common_pipeline.03_utility.validate_physical
python -m common_pipeline.03_utility.build_mixtures
python -m common_pipeline.03_utility.downstream_ridge
python -m common_pipeline.03_utility.plot_utility
python -m common_pipeline.03_utility.interpretation_summary
python -m pytest -q common_pipeline/03_utility/tests
```

Official runs must use the default strict mode. Old results remain historical
and are not consumed by the isolated-run pipeline.
