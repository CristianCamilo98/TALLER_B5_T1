# Notebook 03b — Inspección de features diarias y ventanas T=65

Notebook **explicativo / de inspección** de los artefactos `features-0.1.0`
**Solo lectura.** No recalcula features, no reconstruye ventanas desde OHLCV como pipeline,
no escribe en `data/features/`, no hace splits ni modelos.

## Cómo ejecutarlo

Desde la raíz del repo (`TALLER_B5_T1/`):

```bash
# Dependencias base + visualización del notebook
uv pip install --python .venv/bin/python -r requirements.txt matplotlib ipykernel

# Abrir
jupyter notebook notebooks/03b_inspect_features_windows.ipynb
# o
jupyter lab notebooks/03b_inspect_features_windows.ipynb
```

Seleccionar el kernel del `.venv` del proyecto.

También se puede ejecutar de punta a punta (sin UI):

```bash
.venv/bin/python -m jupyter execute notebooks/03b_inspect_features_windows.ipynb
```

## Controles (celda de setup)

| Variable | Default | Efecto |
|---|---|---|
| `TICKER` | `"NVDA"` | Serie de `log_return` + densidades + ventana ejemplo A |
| `TICKER_B` | `"AMD"` | Segunda ventana (comparar escala) |
| `SAVE_FIGS` | `False` | Si `True`, guarda PNGs en `notebooks/figures/03b_*` |

## Entradas (solo lectura; SHA verificados en el notebook)

| Artefacto | Ruta | SHA256 esperado |
|---|---|---|
| Features diarias | `data/features/daily_features.parquet` | `86e07598ed9e45c6e4b1362b12ad73967f852fcdb13570dc7746496093d10118` |
| Ventanas T=65 | `data/features/windows_65.parquet` | `58bf4c4788cc4ae4feed14c2173419dea876a322c9e6f66d510c9dfb6c00bccf` |
| Manifest | `data/features/features_manifest.json` | (metadatos / conteos / políticas) |
| README canónico | `data/features/README.md` | fórmulas y layout del tensor |

Si algún SHA falla, el notebook **aborta**.

## Alcance

Dentro: dtypes, head/tail, conteos vs manifest, significado de columnas, plots simples,
reshape `features_flat` → `[65, 3]`, check didáctico ventana ↔ slice de `daily_features`.

Fuera: regenerar `data/features/`, splits, estandarización/calibración NVDA, entrenamiento.

Fuente de verdad: artefactos (`features-0.1.0`), no este notebook.
