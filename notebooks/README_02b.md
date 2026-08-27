# Notebook 02b — Auditoría de limpieza OHLCV

> **LEGACY SNAPSHOT:** previo a `common-core-1.0.0`; no es evidencia de certificación.

Diagnóstico interactivo **raw vs clean**. No regenera `data/clean/`; solo lectura.

## Cómo ejecutarlo

Desde la raíz del repo (`TALLER_B5_T1/`):

```bash
# Dependencias base + visualización del notebook
uv pip install --python .venv/bin/python -r requirements.txt matplotlib ipykernel

# Kernel Jupyter (opcional)
.venv/bin/python -m ipykernel install --user --name=taller-cristian --display-name="taller-cristian (.venv)"

# Abrir
jupyter notebook notebooks/02b_audit_cleaning.ipynb
# o
jupyter lab notebooks/02b_audit_cleaning.ipynb
```

Seleccionar el kernel del `.venv` del proyecto.

## Entradas (solo lectura)

| Artefacto | Ruta |
|---|---|
| Raw | `data/raw/ohlcv_raw.parquet` |
| Clean | `data/clean/ohlcv_clean.parquet` |
| Manifest | `data/clean/clean_manifest.json` |

Fuente de verdad de la limpieza: `scripts/clean_ohlcv.py` + panel limpio canónico.

## Alcance

Dentro: NaNs, `Volume==0`, invariantes OHLC, drops, cobertura, cruce con `clean_manifest.json`.
