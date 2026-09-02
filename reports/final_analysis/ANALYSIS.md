# Final quantitative analysis

Derived exclusively from the tracked STRICT_FINAL snapshot
(`artifacts/final/strict_final_20260902/`), via
[`reports/build_final_analysis.py`](../build_final_analysis.py). All numbers
below are read directly from that script's output tables in
`reports/final_analysis/`. No model was retrained, no result was recomputed
at the source, and nothing under `artifacts/final/strict_final_20260902/`
was modified to produce this document.

Model names used throughout: **Bootstrap + Jitter** (simple baseline),
**WGAN-GP** (Cristian), **DDPM** (Daniel), **TimeVAE** (Marco), **Normalizing
Flow** (David). Internal `method_id` values are preserved as a column in
every table but are never used as the primary label here.

## 1. Real-only reference

Ridge trained on the 62 visible NVDA windows only, evaluated on the 150-window
held-out NVDA test set (2023-2025):

- **RMSE = 1.479584**
- **MAE = 1.146934**

This is the single common reference (`REAL_ONLY`) against which every
augmented configuration below is compared. It is a reference baseline, not a
generative method.

## 2. Effect of synthetic augmentation

For every one of the 5 methods, adding synthetic windows lowers both RMSE and
MAE at all three ratios relative to `REAL_ONLY` (see
`master_utility_table.csv`, 15 rows). Improvement is monotonic or
near-monotonic with ratio for most methods:

| Method | RMSE @25% | RMSE @50% | RMSE @75% |
|---|---:|---:|---:|
| Bootstrap + Jitter | 0.2855 | 0.2840 | 0.2627 |
| WGAN-GP | 0.2833 | 0.2749 | 0.2537 |
| DDPM | 0.3098 | 0.3093 | 0.2446 |
| TimeVAE | 0.3815 | 0.3560 | 0.3223 |
| Normalizing Flow | 0.2839 | 0.2720 | 0.2529 |

Every method reaches its lowest RMSE and lowest MAE at **75%** synthetic
share (`model_summary.csv`: `best_rmse_ratio` = `best_mae_ratio` = 0.75 for
all 5 methods). DDPM is the only method whose RMSE at 50% is essentially flat
relative to 25% (0.3093 vs 0.3098) before dropping sharply at 75%.

## 3. Best post-hoc observed configurations

These are **post-hoc descriptive** observations (`post_hoc_description_only = true`
in `model_summary.csv`), not model selection or tuning.

- **Best RMSE overall**: DDPM @ 75% — RMSE = 0.244600 (std 0.007319), 83.47% improvement over REAL_ONLY.
- **Best MAE overall**: Normalizing Flow @ 75% — MAE = 0.174023 (std 0.005471), 84.83% improvement over REAL_ONLY.
- Per-model best RMSE (all at 75%): Bootstrap + Jitter 0.2627 (82.24%), WGAN-GP 0.2537 (82.86%), DDPM 0.2446 (83.47%), TimeVAE 0.3223 (78.22%), Normalizing Flow 0.2529 (82.91%).

DDPM and Normalizing Flow are within 0.003 RMSE of each other at 75%; TimeVAE
is the clear laggard on both RMSE and MAE at every ratio.

## 4. Seed stability

From `seed_stability.csv` (3 subsampling seeds per method x ratio,
descriptive only — no inferential test):

- **Most stable**: TimeVAE @ 25% — `rmse_std` = 0.000648 (`rmse_cv` = 0.0021, `rmse_range` = 0.00139). TimeVAE is the most stable method across all three of its ratios (std between 0.0006 and 0.0040), consistently the tightest of the five methods.
- **Least stable**: DDPM @ 50% — `rmse_std` = 0.063834 (`rmse_cv` = 0.253, `rmse_range` = 0.1394), a coefficient of variation roughly two orders of magnitude larger than TimeVAE's. DDPM @ 25% is also comparatively unstable (`rmse_std` = 0.0347).
- Bootstrap + Jitter and WGAN-GP sit in between, with `rmse_std` shrinking sharply at 75% for every method (the 75% column is the most stable ratio for 4 of 5 methods; TimeVAE is the exception, most stable at 25%).

Stability and accuracy are not the same axis here: TimeVAE is the most
seed-stable method but also has the worst RMSE/MAE of the five.

## 5. Neural models vs simple baseline

From `baseline_vs_neural.csv` (12 rows: 4 neural methods x 3 ratios), positive
`rmse_improvement_pct_difference_vs_baseline` means the neural method
improved more than Bootstrap + Jitter over REAL_ONLY:

- **WGAN-GP** beats the baseline at all three ratios (+0.14, +0.62, +0.61 percentage points of improvement).
- **Normalizing Flow** beats the baseline at all three ratios (+0.10, +0.81, +0.67 points).
- **DDPM** underperforms the baseline at 25% and 50% (-1.64, -1.71 points) and only overtakes it at 75% (+1.23 points).
- **TimeVAE** underperforms the baseline at every ratio, by a wide margin (-6.49, -4.86, -4.02 points).

So the answer is **method-dependent, not uniform**: two of the four neural
generators (WGAN-GP, Normalizing Flow) consistently add value over the
simple bootstrap+jitter baseline on this task; DDPM only does so once enough
synthetic volume is added (75%); TimeVAE does not surpass the baseline at
any tested ratio.

## 6. Fidelity

From `fidelity_master.csv` (held-out `donor_validation`, 380 real vs 380
synthetic windows per method — this is donor market data, **not NVDA**). No
composite score; leader/laggard per individual metric:

| Metric (lower = closer to real) | Closest to real | Furthest from real |
|---|---|---|
| C2ST ROC-AUC | Bootstrap + Jitter (0.897) | TimeVAE (0.999) |
| Mean correlation error | Bootstrap + Jitter (0.046) | DDPM (0.080) |
| Wasserstein, log_return | DDPM (0.308) | TimeVAE (0.899) |
| Wasserstein, log_high_low_range | DDPM (0.590) | TimeVAE (0.673) |
| Wasserstein, log1p_volume | WGAN-GP (0.118) | DDPM (0.493) |
| Return ACF MAE | WGAN-GP (0.035) | TimeVAE (0.095) |
| Abs-return ACF MAE | DDPM (0.028) | TimeVAE (0.042) |

No single method leads on every metric. **TimeVAE is furthest from real on 5
of the 7 metrics**, most sharply on the return channel (Wasserstein
log_return nearly 3x the next-worst method), consistent with the variance
collapse on `log_return` already documented in Marco's own README. Bootstrap
+ Jitter's strong C2ST/correlation results are expected: it is bootstrapped
real donor windows plus small jitter, so it is structurally close to the
reference data rather than "well generated." Nearest-neighbour distance to
`donor_train` is not directly comparable across methods on a common
better/worse scale here: Bootstrap + Jitter's is near-zero (0.70) for the
same structural reason, while the four neural generators range from 6.7
(TimeVAE) to 10.7 (DDPM).

## 7. Fidelity vs utility

`fidelity_utility_correlations.csv`, n=5, exploratory only
(`exploratory_only=true`, `statistical_inference=false`, no p-value used as
evidence):

- Several return-channel fidelity metrics move together with utility improvement in this 5-point sample: **Wasserstein log_return** (Pearson r = -0.98 vs `best_rmse_improvement_pct`), **return ACF MAE** (r = -0.97), **abs-return ACF MAE** (r = -0.99), and **C2ST ROC-AUC** (r = -0.93). In all four cases, worse fidelity on the return channel co-occurs with smaller RMSE improvement — driven largely by TimeVAE sitting at the extreme on both axes.
- Other fidelity metrics show **no consistent relationship**: nearest-neighbour distance (r = 0.31, Spearman rho = 0.10-0.15) and the volume-channel Wasserstein distance (r = 0.01 to -0.33 depending on the utility target, sign flips between targets) track utility improvement weakly or inconsistently.
- Spearman rho is markedly weaker than Pearson r for most pairs (e.g. Wasserstein log_return: Pearson -0.98 vs Spearman -0.60/0.00), which is expected with n=5 and one influential point (TimeVAE) — a reminder that these Pearson values are not robust evidence of a monotonic relationship, just a description of this specific 5-point sample.

**High fidelity does not necessarily mean high downstream utility, and the
reverse also does not hold cleanly here**: Bootstrap + Jitter has the best
C2ST and correlation-error fidelity of the five but only middling utility
improvement (81.25% mean), while DDPM has mediocre-to-poor fidelity on
several metrics (worst correlation error and volume-channel Wasserstein) yet
the best RMSE overall. The one fairly consistent pattern in this sample is
specific to the **return channel**: the method that is worst there
(TimeVAE) is also the method with the smallest downstream improvement. This
is a descriptive association in 5 points, not a causal claim.

## 8. Main findings

1. Every one of the 5 methods improves RMSE and MAE substantially over the 62-real-window baseline (74-85% relative improvement), and every method's best result is at the highest tested ratio (75%).
2. DDPM has the best overall RMSE (0.2446 @ 75%); Normalizing Flow has the best overall MAE (0.1740 @ 75%); both sit within ~0.003 RMSE of the WGAN-GP/DDPM/Normalizing Flow cluster at 75%.
3. TimeVAE is the utility laggard at every ratio (worst RMSE and MAE of the five methods) and is also furthest from real on 5 of 7 fidelity metrics, most severely on the log_return channel.
4. Only WGAN-GP and Normalizing Flow beat the simple Bootstrap + Jitter baseline at all three tested ratios; DDPM only overtakes it at 75%; TimeVAE never does.
5. TimeVAE is simultaneously the most seed-stable method (lowest RMSE std/CV) and the least accurate one — stability and accuracy are separate axes in this data.
6. DDPM is the least seed-stable configuration observed (RMSE std = 0.064 @ 50%, CV = 0.25), an order of magnitude more variable than TimeVAE's most stable point.
7. No fidelity metric or generator dominates across the board; return-channel fidelity metrics (Wasserstein log_return, return/abs-return ACF error, C2ST) track utility improvement fairly strongly in this 5-point sample, while nearest-neighbour distance and volume-channel Wasserstein do not.
8. The observed 74-85% RMSE/MAE improvements are arithmetically confirmed against the STRICT_FINAL source tables (`sanity_check_improvements.csv`, 15/15 rows match the formula, single non-duplicated `REAL_ONLY` reference) — the size of the improvement reflects how poorly a Ridge model generalizes from only 62 real training windows, not a computation error.

## 9. Limitations

- Only 3 subsampling seeds (42, 123, 2026) per method x ratio; stability statistics (`rmse_std`, `rmse_cv`, ranges) are descriptive over n=3, not inferential.
- Fidelity-vs-utility correlations use n=5 methods; Pearson/Spearman values in `fidelity_utility_correlations.csv` are exploratory only and are not evidence of statistical significance or causality.
- All comparisons in Sections 2-5 use a single downstream model (Ridge, alpha=1.0, fixed features/scaler) on a single forecasting target (5-session annualized realized volatility) and a single held-out test set (150 NVDA windows, 2023-2025); they do not generalize to other models, targets, or assets.
- All comparisons in Sections 3-4 are post-hoc descriptive ("best", "most stable"); none of it was used to select, tune, or retrain any generator.
- Fidelity (Section 6) is measured against held-out `donor_validation` (donor market data), not NVDA; it answers a different question than the utility results and the two are only related descriptively in Section 7.
- No causal claim is made anywhere in this document between fidelity and utility, or between any generator's architecture and its results.
