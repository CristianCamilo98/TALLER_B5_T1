# Generador David - RealNVP Normalizing Flow

Cuarto generador oficial del taller. Produce ventanas sinteticas en el espacio
comun `global_channel_normalized` mediante un Normalizing Flow RealNVP con
ActNorm aprendible, capas de acoplamiento afin y permutaciones fijas entre
bloques.

El modelo tiene arquitectura neuronal entrenable, optimizador Adam,
checkpoint, likelihood bajo prior normal estandar y calculo exacto del
log-det-Jacobian de cada transformacion invertible. El muestreo oficial usa
rechazo determinista contra las restricciones fisicas NVDA, sin clipping ni
reparacion de valores.

## Contrato

| Elemento | Valor |
|---|---|
| Entrada | `data/features/windows/donor_train.parquet` |
| Normalizacion | z-score global por canal, ajustado solo con `donor_train` |
| Shape logico | `5000 x 65 x 3` |
| Canales | `log_return`, `log_high_low_range`, `log1p_volume` |
| Seed oficial | `42` |
| Source model | `normalizing_flow` |
| Arquitectura | RealNVP, ActNorm, affine coupling, fixed permutations, MLP tanh |
| Objetivo | negative log-likelihood |
| Entrenamiento default | 10000 epochs, early stopping patience 200 |
| Regularizacion default | Adam lr 5e-4, weight decay 5e-5 |
| Sampling default | temperature 1.0, rechazo fisico determinista |
| Output | `outputs/bootstrap_jitter_seed42_normalized.parquet` |

El nombre del parquet oficial se conserva por compatibilidad con el pipeline
existente, pero la columna `source_model` y la provenance identifican el
metodo real como `normalizing_flow`.

## Ejecutar con la base local del Explorer

Desde la raiz del repositorio:

```powershell
python scripts/import_local_data_snapshot.py --source-data-root "C:\Users\david\Desktop\Taller 5\data"
python generadores/david/scripts/train_normalizing_flow.py
python generadores/david/scripts/generate_normalized.py
python common_pipeline/01_contract/validate_outputs.py
```

Despues del contrato, el run comun estricto ya puede incluir el rol oficial de
David:

```powershell
python -m common_pipeline.02_fidelity.evaluate_fidelity
python -m common_pipeline.03_utility.calibrate_nvda
python -m common_pipeline.03_utility.validate_physical
python -m common_pipeline.03_utility.build_mixtures
python -m common_pipeline.03_utility.downstream_ridge
python -m common_pipeline.03_utility.plot_utility
python -m common_pipeline.03_utility.interpretation_summary
```

## Artefactos

El entrenamiento crea:

- `artifacts/checkpoints/normalizing_flow_seed42.npz`
- `artifacts/training_history_seed42.csv`
- `artifacts/training_manifest_seed42.json`

La generacion crea:

- `outputs/bootstrap_jitter_seed42_normalized.parquet`
- `outputs/bootstrap_jitter_seed42_normalized.provenance.json`

## Notas

El antiguo `temporal_jitter_0p40_rho0p85` queda como experimento historico y
baseline fuerte, pero no satisface el rol oficial `normalizing_flow`.
