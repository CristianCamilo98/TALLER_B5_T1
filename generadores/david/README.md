# Generador David - Normalized Temporal Jitter

Cuarto generador simple del taller. Produce ventanas sinteticas en el espacio
comun `global_channel_normalized` mediante `bootstrap_resample +
temporal_correlated_gaussian_jitter`: remuestrea ventanas reales de
`donor_train` ya normalizadas y anade ruido gaussiano AR(1) correlacionado por
canales.

## Contrato

| Elemento | Valor |
|---|---|
| Entrada | `data/features/windows/donor_train.parquet` |
| Normalizacion | z-score global por canal, ajustado solo con `donor_train` |
| Shape logico | `5000 x 65 x 3` |
| Canales | `log_return`, `log_high_low_range`, `log1p_volume` |
| Seed oficial | `42` |
| Source model | `temporal_jitter_0p40_rho0p85` |
| Ruido | `noise_scale = 0.40` |
| Persistencia temporal | `rho = 0.85` |
| Espacio | `global_channel_normalized` |
| Output | `outputs/bootstrap_jitter_seed42_normalized.parquet` |

El parquet oficial contiene exactamente las columnas canonicas:
`synthetic_id`, `source_model`, `training_seed`, `space`, `window_length`,
`n_channels`, `channel_order`, `features_flat`.

El nombre del fichero conserva `bootstrap_jitter` por compatibilidad con el
pipeline comun ya cableado; la columna `source_model` y el JSON de provenance
identifican el algoritmo real usado.

## Ejecutar

Desde la raiz del repositorio:

```powershell
python generadores/david/scripts/generate_normalized.py
python common_pipeline/01_contract/validate_outputs.py
```

Para reproducir el sweep de mejoras:

```powershell
python generadores/david/scripts/experiment_normalized.py
```

Si faltan los parquets canonicos de datos, regenerar primero:

```powershell
python scripts/download_ohlcv_raw.py --config configs/experiment.yaml --reuse-snapshot
python scripts/clean_ohlcv.py --config configs/experiment.yaml
python scripts/assign_splits.py --config configs/experiment.yaml
python scripts/build_features_windows.py --config configs/experiment.yaml
```

Si no existe snapshot raw local, la primera linea debe ejecutarse sin
`--reuse-snapshot` para descargar de yfinance.

## Interpretacion

Este modelo no intenta superar arquitectonicamente a VAE, WGAN-GP o Diffusion.
Sirve como linea base fuerte y reproducible: mide cuanto aporta un modelo
neuronal frente a copiar estructura empirica donor y perturbarla localmente.

Se probaron variantes de ruido independiente, ruido correlacionado, ruido
temporal, bootstrap por regimen, mixup entre vecinos de regimen y GMM en PCA.
La candidata promovida fue `temporal_jitter_0p40_rho0p85` porque, entre las
variantes con 0% de rechazo fisico, fue la que mejoro mas C2ST, Wasserstein y
RMSE downstream frente al baseline inicial. El coste observado es una peor
ACF de retornos: el modelo gana diversidad y utilidad, pero suaviza/perturba
mas la estructura temporal fina.

Comparacion principal frente al baseline inicial `noise_scale=0.05`:

| metrica | baseline inicial | David mejorado |
|---|---:|---:|
| C2ST AUC, menor es mejor hacia 0.5 | 0.8971 | 0.8651 |
| Wasserstein medio | 0.3743 | 0.3439 |
| return ACF MAE | 0.0358 | 0.0575 |
| ventanas invalidas post-calibracion | 0/5000 | 0/5000 |
| mejor RMSE downstream | 0.2627 | 0.2401 |

El generador no usa `nvda_visible` ni `nvda_test` para construir ventanas. La
calibracion a NVDA-like, el rechazo de ventanas fisicamente invalidas y la
evaluacion downstream sobre `nvda_test` pertenecen al pipeline comun
`common_pipeline/03_utility`.
