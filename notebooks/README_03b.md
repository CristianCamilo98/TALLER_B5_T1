# Notebook 03b — Inspección de features y ventanas multi-stride (`features-0.2.0`)

Notebook **explicativo / de inspección** de los artefactos `features-0.2.0`.
**Solo lectura.** No regenera `data/features/`, no recalcula features, no hace splits ni modelos.

## Cómo ejecutarlo

Desde la raíz del repo (`taller_cristian/`):

```bash
uv pip install --python .venv/bin/python -r requirements.txt matplotlib ipykernel

jupyter notebook notebooks/03b_inspect_features_windows.ipynb
# o
jupyter lab notebooks/03b_inspect_features_windows.ipynb
```

Seleccionar el kernel del `.venv` del proyecto.

Ejecución headless:

```bash
.venv/bin/python -m jupyter execute notebooks/03b_inspect_features_windows.ipynb
```

## Controles (celda de setup)

| Variable | Default | Efecto |
|---|---|---|
| `TICKER` | `"NVDA"` | Serie `log_return`, densidades, ventana ejemplo |
| `STRIDE` | `1` | Parquet a inspeccionar (`primary_stride` / default recomendado) |
| `STRIDE_B` | `65` | Segunda ventana + segundo check de consistencia |
| `SAVE_FIGS` | `False` | Si `True`, guarda PNGs en `notebooks/figures/03b_*` |

## Entradas (solo lectura; SHA vía `checksums.sha256`)

| Artefacto | Ruta |
|---|---|
| Manifest | `data/features/features_manifest.json` (`data_version: features-0.2.0`) |
| Checksums | `data/features/checksums.sha256` |
| Features diarias | `data/features/daily_features.parquet` |
| Menú de ventanas | `data/features/windows_65_stride{1,10,30,65}.parquet` |
| README canónico | `data/features/README.md` |

Si `data_version != features-0.2.0`, faltan strides o falla un SHA → el notebook **aborta**.

## Alcance

Dentro: significado de las 3 features, tensor `[65,3]`, menú de strides / `primary_stride=1`,
conteos al variar stride, plots simples, check reshape↔daily (stride 1 y otro).

Fuera: regenerar `data/features/`, splits, estandarización/calibración NVDA, entrenamiento.

Fuente de verdad: artefactos `features-0.2.0`, no este notebook.
