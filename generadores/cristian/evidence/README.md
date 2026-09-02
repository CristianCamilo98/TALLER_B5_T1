# Cristian WGAN-GP — training evidence (seed 42)

Evidencia verificable del run que produjo el output oficial:

`generadores/cristian/outputs/synthetic_seed42_n5000_normalized.parquet`

SHA256: `135503132f19f8d856e89b595cbc283baafc29864e2fb042d99bc12bfe892123`

## Run oficial de entrenamiento

| Campo | Valor |
|-------|-------|
| Modelo | WGAN-GP |
| Epochs ejecutadas | **300** (índices 0..299) |
| Batch size | 64 |
| Seed declarada | 42 |
| Checkpoint final | `generator_epoch_00299.keras` |
| Checkpoint SHA256 | `5392083d1cf0e9c8306c2e3136c21608aad8f26f43e7578beeab398322dc7cc0` |

**Nota:** `configs/wgan_gp.yaml` lista `epochs: 5000` como configuración histórica/default.
Ese valor **no** corresponde al run ejecutado que generó el output publicado.

## Archivos en este directorio

| Archivo | Contenido |
|---------|-----------|
| `training_manifest.json` | Metadatos completos del run y auditoría |
| `provenance.json` | Cadena de provenance y límites declarados |
| `loss_history.csv` | Copia de la history real (300 epochs) |
| `figures/training_convergence.png` | Curva de convergencia (300 epochs reales) |

Los checkpoints `.keras` no se publican (gitignored, ~1 MB); su SHA256 está en el manifest.

## Donor train — provenance

| Campo | Valor |
|-------|-------|
| SHA local (training) | `b92a36a01b1931397d9a2c88c0f3473f224586b0193b0aefbb4e47f12eb00eca` |
| SHA canónico (master) | `5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f` |
| Bytes idénticos | **NO** |
| Same window identifiers | **YES** |
| Shape | `(4910, 65, 3)` ambos |
| max_abs_diff | `1.506e-06` |
| mean_abs_diff | `6.742e-08` |
| fraction > 1e-6 | 0.06% |
| allclose atol=1e-5 | **YES** |

**Verdict:** `SAME_SCIENTIFIC_DATA` — **RETRAIN_REQUIRED = NO**

## Limitación de seed

`SEED_REPRODUCIBILITY = NOT_BITWISE_GUARANTEED`

El generator y el critic se construyen en `WGAN_GP.__init__()` **antes** de
`tf.keras.utils.set_random_seed()` en `train()`. Esto limita la reproducibilidad
bit-a-bit de un retrain, pero **no invalida** el output ya publicado.

## Cadena de generación

```
donor_train.parquet
  → normalizer.json (artifacts/seed_42/)
  → entrenamiento WGAN-GP, 300 epochs, seed 42
  → generator_epoch_00299.keras
  → generate_synthetic.py --seed 42
  → synthetic_seed42_n5000_normalized.parquet
```
