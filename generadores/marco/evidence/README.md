# Evidence — Marco VAE

Artefactos verificables de entrenamiento, generados por auditoria de solo
lectura (`generadores/marco/audit_training_evidence.py`,
`investigate_donor_provenance.py`, `write_*_evidence.py`). Nada aqui fue
inventado — cada campo referencia un archivo real o declara explicitamente
su ausencia/estado no verificable.

## Contenido

| Archivo | Contenido |
|---|---|
| `training_manifest.json` | Arquitectura, hiperparametros, resultado de entrenamiento, hash del checkpoint, estado de provenance del donor |
| `scaler_stats.json` | mean/std reales del scaler, fit_dtype, estado del guard de varianza cero |
| `convergence_curve.png` | Curva reproducida directamente desde `loss_history.npz` (no la figura antigua) |
| `loss_vae_donors_reference.png` | Copia de la figura ya publicada en `figures/`, para comparar |

## Estado de verificacion (resumen)

- **Checkpoint**: disponible, SHA256 registrado en `training_manifest.json`.
- **Loss history (raw)**: disponible (`loss_history.npz`, local, no versionado).
- **Scaler**: disponible, `float64` estricto (fix aplicado y documentado en
  el README principal).
- **Donor SHA**: **NO verificable**. El SHA canonico registrado en
  `data/features/features_manifest.json` (`5f1e33f6...`) no coincide con el
  SHA del archivo `donor_train.parquet` presente localmente en el momento
  de esta auditoria (`a9c8f76d...`). El archivo no esta versionado en git
  (excluido por `.gitignore` de la base comun), por lo que no hay historial
  que permita reconstruir cuando o por que cambio. No se puede confirmar
  que el checkpoint actual se entreno con el donor exacto certificado como
  oficial. No se ha reentrenado para resolver esto (fuera de scope de esta
  auditoria).
- **Discrepancia del guard de varianza cero** (`std==0 -> 1e-8` en la
  implementacion propia vs. `sigma<1e-8 -> 1.0` en el contrato comun): sin
  impacto numerico confirmado — ningun canal del scaler actual tiene
  `std < 1e-8`.

## Reproducibilidad

```powershell
python -m generadores.marco.audit_training_evidence
python -m generadores.marco.investigate_donor_provenance
python -m generadores.marco.write_scaler_evidence
python -m generadores.marco.write_training_manifest
python -m generadores.marco.write_convergence_evidence
```
