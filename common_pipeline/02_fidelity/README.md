# Common synthetic fidelity

This module evaluates every generator with one implementation, one real
reference, one normalized space, and one deterministic sampling policy. It is
independent of `01_contract` and `03_utility` and imports no generator code.

## Inputs and common space

- Real fidelity reference: `data/features/windows/donor_validation.parquet`
  (380 windows).
- Normalizer fit reference: `data/features/windows/donor_train.parquet`
  (4,910 windows).
- Synthetic inputs: one published `*normalized*.parquet` under each
  `generadores/<owner>/outputs/` directory.
- Logical shape: `(N, 65, 3)` reconstructed from 195 session-major values.
- Channel order: `log_return`, `log_high_low_range`, `log1p_volume`.

The real reference is transformed with the global channel scaler fitted in
float64 on donor train only: one population mean and standard deviation per
channel over axes `(window, time)`, followed by a float32 transform. Validation
never participates in fit. Synthetic pools are evaluated exactly as published;
they are not re-standardized, clipped, calibrated, winsorized, or repaired.
Negative normalized range or volume z-scores are valid here. Physical validity
belongs to utility after NVDA calibration, not to normalized fidelity.

## Deterministic evaluation subset

The pipeline creates one ordered array of 380 row positions using
`np.random.default_rng(evaluation_subset_seed=42).choice(5000, size=380,
replace=False)`. The exact same positions are applied to every method. All
metrics and figures use these balanced 380-window subsets and all 380 real
validation windows.

## Metrics

- Marginal count, mean, population standard deviation, median, p01, p05, p25,
  p75, p95, p99, bias-corrected skewness, and bias-corrected Fisher/excess
  kurtosis by channel.
- Empirical Wasserstein-1 by channel and its descriptive channel mean.
- Return and absolute-return ACF at lags 1-20. ACF is calculated inside each
  65-session window using the window mean and biased denominator, then averaged
  across non-constant windows. No pair crosses a window boundary.
- Pearson channel correlations after flattening only sample and time axes, plus
  mean/max absolute off-diagonal error against real validation.
- Euclidean nearest-neighbour distance in 195-D from each evaluation window to
  normalized donor train. This is a memorization/drift diagnostic, not a pass
  rule.
- Balanced Logistic Regression C2ST with `StandardScaler`, five-fold shuffled
  stratification, and out-of-fold probabilities. ROC-AUC near 0.5 means this
  linear classifier has little discriminatory ability; it does not prove equal
  distributions.
- One joint `StandardScaler -> PCA(30) -> t-SNE(2)` embedding containing real
  and every method. It is a visualization, never a quantitative ranking.

The scorecard displays separate metrics in their own units. It deliberately
does not calculate a composite fidelity score.

## Commands

From the repository root, after four neural outputs are present:

```bash
python common_pipeline/02_fidelity/evaluate_fidelity.py
python common_pipeline/02_fidelity/plot_fidelity.py
```

During parallel development, the available neural count can be stated without
changing any metric definition:

```bash
python common_pipeline/02_fidelity/evaluate_fidelity.py --expected-neural-count 3
python common_pipeline/02_fidelity/plot_fidelity.py
```

Explicit inputs are supported with repeatable `--synthetic METHOD=PATH`. Once
the optional simple baseline exists, add:

```bash
--baseline-path common_pipeline/01_contract/outputs/bootstrap_jitter_seed42_normalized.parquet
```

Passing a missing baseline path fails explicitly; the baseline is never an
implicit dependency.

## Outputs

`results/tables/` contains the metric CSV files, shared subset indices, and the
joint t-SNE coordinates. `results/figures/` contains the three marginal plots,
two ACF plots, correlation matrices, nearest-neighbour summary, C2ST AUC, joint
t-SNE, and the non-composite scorecard. `results/evaluation_manifest.json`
records input paths/hashes, normalization parameters, subset policy, methods,
and the explicit absence of physical filtering or synthetic repair.

These runtime results are ignored by Git because the definitive tables and
figures must be regenerated only after all four conforming neural outputs (and,
when available, the optional baseline) are present. The scripts and tests are
the versioned common contract.
