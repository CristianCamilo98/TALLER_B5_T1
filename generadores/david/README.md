# Generador David - RealNVP Normalizing Flow

Cuarto generador oficial del taller. Produce ventanas sinteticas en el espacio
comun `global_channel_normalized` mediante un Normalizing Flow RealNVP con
ActNorm aprendible, capas de acoplamiento afin y permutaciones fijas entre
bloques.

El modelo tiene arquitectura neuronal entrenable, optimizador Adam,
checkpoint, likelihood bajo prior normal estandar y calculo exacto del
log-det-Jacobian de cada transformacion invertible. El entrenamiento usa solo
`donor_train` y la seleccion de checkpoint usa solo NLL sobre `donor_validation`.

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
| Sampling default | temperature 1.0 desde la distribucion base normal |
| Output oficial | `outputs/normalizing_flow_seed42_normalized.parquet` |

## Modelo oficial

- **Official model:** Normalizing Flow / RealNVP
- **Official output:** `normalizing_flow_seed42_normalized.parquet`
- **Architecture:** RealNVP + ActNorm + affine coupling + fixed permutations
- **Training:** configured epochs = 10000, completed epochs = 289, best epoch = 89, early stopping patience = 200
- **Selection:** NLL minima sobre `donor_validation` (donor_validation negative log-likelihood)
- **Data:** `donor_train` canonico + `donor_validation` canonico (SHA256 verificados en training)

### Sin leakage

- NO NVDA
- NO utility
- NO RMSE/MAE
- NO test selection
- NO physical filtering
- NO calibration antes de fidelity

## Ejecutar

Desde la raiz del repositorio:

```powershell
python generadores/david/scripts/train_normalizing_flow.py
python generadores/david/scripts/generate_normalized.py
python common_pipeline/01_contract/validate_outputs.py
```

No se usa NVDA para training, validation, sampling, filtro, seleccion de
checkpoint ni ajuste de hiperparametros.

## Artefactos

El entrenamiento crea:

- `artifacts/checkpoints/normalizing_flow_seed42.npz`
- `artifacts/loss_history.csv`
- `artifacts/training_manifest_seed42.json`
- `artifacts/normalizing_flow_convergence.png`

La generacion crea:

- `outputs/normalizing_flow_seed42_normalized.parquet`
- `outputs/normalizing_flow_seed42_normalized.provenance.json`

## Notas

`generadores/david/scripts/experiment_normalized.py` y
`generadores/david/scripts/plot_experiment_diagnostics.py`, junto con todo lo
que hay bajo `experiments/`, pertenecen al antiguo experimento temporal-jitter
/ candidate-search (incluido `temporal_jitter_0p40_rho0p85`).

**LEGACY. NOT FINAL MODEL. NOT USED IN FINAL EXPERIMENT.**

Sus metricas de RMSE/utility/fidelity no son oficiales y no deben
presentarse como resultados del modelo final. Ese script ya no tiene
capacidad de escribir ni sustituir el output oficial (`--promote` fue
eliminado).
