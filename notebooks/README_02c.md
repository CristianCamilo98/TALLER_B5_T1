# Notebook 02c — EDA exploratoria del panel limpio

Playground interactivo sobre **OHLCV ya limpio**. Sirve para entender la naturaleza de los datos ticker a ticker y en conjunto (añadir celdas libremente).

**No** regenera ni corrige `data/clean/`. Si el SHA no coincide, el notebook se detiene.

## Cómo ejecutarlo

Desde la raíz del repo (`TALLER_B5_T1/`):

```bash
# Dependencias base + kernel
uv pip install --python .venv/bin/python -r requirements.txt matplotlib ipykernel

# Kernel Jupyter (opcional)
.venv/bin/python -m ipykernel install --user --name=taller-cristian --display-name="taller-cristian (.venv)"

# Abrir
jupyter notebook notebooks/02c_eda_clean_panel.ipynb
# o
jupyter lab notebooks/02c_eda_clean_panel.ipynb
```

Seleccionar el kernel del `.venv` del proyecto. También funciona si el cwd es `notebooks/` (resuelve `ROOT` hacia el padre).

### Controles rápidos (celda de setup)

| Variable | Uso |
|---|---|
| `USE_LOG_SCALE` | Close en escala log |
| `DATE_START` / `DATE_END` | Zoom temporal exploratorio |
| `TICKER` | Drill-down (§8), p. ej. `"NVDA"` |
| `SAVE_FIGS` | `True` → exporta `notebooks/figures/02c_*.png` |

## Entradas (solo lectura)

| Artefacto | Ruta | SHA256 esperado |
|---|---|---|
| Clean | `data/clean/ohlcv_clean.parquet` | `fb1d9e60853743fff8cf6a2b17fb7588915d4213022c75b7311f33944a974f25` |
| Manifest | `data/clean/clean_manifest.json` | (contexto) |

## Alcance

Dentro: forma del panel, precios, volumen, returns exploratorios $\ln(C_t/C_{t-1})$, rango intradía, correlaciones, drill-down.


## Nota playground

Este notebook **puede crecer**. Lo que no puede hacer es cambiar el artefacto canónico clean. Problemas graves de datos → documentar y escalar al orquestador.
