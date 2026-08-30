# Common Synthetic Utility Pipeline (`03_utility`)

Pipeline unico y compartido para que TODOS los generadores (VAE/GAN/Diffusion/
baseline) pasen por la misma calibracion, validacion fisica, mezclas real+
sintetico, features, scaler y Ridge -- la unica variable experimental es que
metodo genero el sintetico y en que proporcion.

No modifica `01_contract/` ni `02_fidelity/`, ni ningun `generadores/*`.

## Pipeline

## Metodologia clave

- **Calibracion**: `X = mu_NVDA + sigma_NVDA * Z_syn`, afin pura, SIN
  re-estandarizar el pool generado -- los errores del generador se ven tal
  cual, no se enmascaran. `mu`/`sigma` calculados sobre 126 observaciones
  diarias unicas de NVDA visible (no las 62 ventanas solapadas), `ddof=0`.
- **Validacion fisica**: post-calibracion. `log_return` solo exige finitud
  (sin restriccion de signo); `log_high_low_range` y `log1p_volume` exigen
  finitud y `>= 0`. Ventanas invalidas se descartan enteras, sin reparar
  (sin `clip`/`abs`/winsorize -- verificado por test via AST).
- **Mezclas**: 62 reales fijas + 0/21/62/186 sinteticas (ratios 0/25/50/75%).
  Subsampling sin reemplazo, 3 seeds (42/123/2026) para ratios > 0.
- **Scaler**: `StandardScaler` ajustado UNA sola vez con las 62 reales
  visibles -- nunca reajustado por mezcla ni con sintetico.
- **Modelo**: `Ridge(alpha=1.0, fit_intercept=True)` identico para todos,
  sin busqueda de hiperparametros.
- **Test**: 150 ventanas canonicas de NVDA 2023-2025, stride=5, nunca usado
  para calibrar/ajustar scaler/elegir ratio.

## Como ejecutar

```powershell
python -m common_pipeline.03_utility.calibrate_nvda
python -m common_pipeline.03_utility.validate_physical
python -m common_pipeline.03_utility.build_mixtures
python -m common_pipeline.03_utility.downstream_ridge
python -m common_pipeline.03_utility.plot_utility
python -m common_pipeline.03_utility.interpretation_summary
python -m pytest common_pipeline/03_utility/tests/ -v
```

`results/calibrated_pools/*.npz` no esta versionado (regenerable via
`validate_physical.py`, ver `.gitignore`).

## Resultados (con los generator outputs disponibles al momento del merge)

| method | best_ratio | best_mean_rmse | delta_rmse_vs_real | invalid_rate |
|---|---:|---:|---:|---:|
| cristian | 0.75 | 0.2537 | -82.9% | 0.0% |
| daniel | 0.75 | 0.2446 | -83.5% | 0.0% |
| marco | 0.75 | 0.3509 | -76.3% | 0.0% |

`real_only` identico en los 3 (RMSE=1.4796) -- confirma consistencia del
pipeline antes de que entre la variable experimental. Detalle completo en
`results/tables/downstream_results_summary.csv` e `interpretation_summary.csv`.

## Blockers de input conocidos (no corregidos aqui, ver Parte 11 del protocolo)

- Output del cuarto generador (baseline `01_contract`) aun no integrado via
  path opcional -- el pipeline lo acepta pero no se ha ejecutado con el.
- Ejecutado con 3 de 4 metodos (cristian, daniel, marco). El run definitivo
  con los 4 y cifras oficiales queda pendiente de alineacion de todos los
  generator outputs.
