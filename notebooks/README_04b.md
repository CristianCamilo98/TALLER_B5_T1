# Notebook 04b — Inspección de splits temporales (`splits-0.1.0`)

> **LEGACY / NO EJECUTAR:** referencia la asignación post-ventana retirada por leakage.

Notebook **explicativo / de inspección** de los artefactos `splits-0.1.0`.
**Solo lectura.** No reasigna splits, no escribe `data/splits/` nuevos, no entrena,
no genera sintéticos. No muta `data/features/` ni `data/clean/`.

## Cómo ejecutarlo

Desde la raíz del repo:

```bash
uv pip install --python .venv/bin/python -r requirements.txt matplotlib ipykernel

jupyter notebook notebooks/04b_inspect_splits.ipynb
# o
jupyter lab notebooks/04b_inspect_splits.ipynb
```

Seleccionar el kernel del `.venv` del proyecto.

Ejecución headless:

```bash
.venv/bin/python -m jupyter execute notebooks/04b_inspect_splits.ipynb
```

## Controles (celda de setup)

| Variable | Default | Efecto |
|---|---|---|
| `STRIDE` | `1` | Parquet de splits a inspeccionar en profundidad (`primary_stride`) |
| `STRIDE_COMPARE` | `65` | Segundo stride para comparar composición / `nvda_visible` |
| `SAVE_FIGS` | `False` | Si `True`, guarda PNGs en `notebooks/figures/04b_*` |

## Entradas (solo lectura; SHA vía `checksums.sha256`)

| Artefacto | Ruta |
|---|---|
| Manifest | `data/splits/split_manifest.json` (`data_version: splits-0.1.0`) |
| Checksums | `data/splits/checksums.sha256` |
| Splits por stride | `data/splits/window_splits_stride{1,10,30,65}.parquet` |
| Join demo (opcional) | `data/features/windows_65_stride1.parquet` |
| README canónico | `data/splits/README.md` |

Si `data_version != splits-0.1.0` o falla un SHA → el notebook **aborta**.

## Alcance

Dentro: significado de cada split, contrato de fechas, conteos split × stride,
composición (barras/stacked), timeline de `window_end_date`, assert 0 NVDA en
donors train/val, demo join → filtrar `donor_train`, advertencia `nvda_visible`
con stride alto, handoff a generadores.

Fuera: reasignar splits, entrenar, sintéticos, calibración NVDA, mutar features/clean.

Fuente de verdad: artefactos `splits-0.1.0`, no este notebook.
