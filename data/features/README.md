# Features diarias + ventanas T=65 (`features-0.1.0`)

Panel de features generativas diarias y ventanas de 65 días a partir de
`data/clean/ohlcv_clean.parquet`. **Solo features + ventanas.**

Sin splits train/val/test, sin calibración a NVDA, sin estandarización, sin
entrenamiento (VAE/GAN/Diffusion/Ridge). Sin re-limpieza ni escritura en
`data/clean/`.

> **Mantenimiento:** cualquier cambio en fórmulas o políticas debe actualizar
> **este README** y `scripts/build_features_windows.py` (y regenerar artefactos +
> `features_manifest.json`).

## Entrada (solo lectura)

| Campo | Valor |
|---|---|
| Ruta | `data/clean/ohlcv_clean.parquet` |
| SHA256 esperado | `fb1d9e60853743fff8cf6a2b17fb7588915d4213022c75b7311f33944a974f25` |
| `data/clean/` | **no se modifica** |

Si el SHA del parquet de entrada no coincide, el script **aborta**.

## Features (por ticker, orden temporal)

1. `log_return_t = ln(Close_t / Close_{t-1})`
2. `log_high_low_range_t = ln(High_t / Low_t)`; si `High == Low` → `0.0`
3. `log_volume_t = ln(Volume_t)`; filas con `Volume == 0` → **DROP** (no `ln(1)`)

Reglas:

- Independiente por ticker.
- Filas `Volume == 0` se eliminan **antes** de calcular features (el `log_return`
  siguiente usa el `Close` de la fila previa restante).
- Primera fila de cada ticker sin `log_return` → fuera.
- Drop NaN/inf en las 3 features.
- **No** ffill. **No** winsorize. **No** estandarizar (calibración NVDA = otro chat).
- Huecos vs unión de fechas: las ventanas usan **filas consecutivas del panel de
  features**, no días de calendario rellenados.

## Ventanas

| Parámetro | Valor |
|---|---|
| `T` | 65 |
| `stride` | 1 |
| Canales (orden fijo) | `['log_return', 'log_high_low_range', 'log_volume']` → shape `[65, 3]` |
| Ámbito | un solo ticker por ventana |
| Metadatos | `ticker`, `window_start_date`, `window_end_date` |

## Ejecutar

```bash
cd TALLER_B5_T1
.venv/bin/python scripts/build_features_windows.py
```

## Artefactos

| Ruta | Contenido |
|---|---|
| `data/features/daily_features.parquet` | Features diarias |
| `data/features/daily_features.csv` | Misma tabla en CSV |
| `data/features/windows_65.parquet` | Ventanas + `features_flat` |
| `data/features/features_manifest.json` | Metadatos, conteos, SHA, políticas |
| `data/features/checksums.sha256` | SHA256 de parquet/csv de salida |
| `data/features/README.md` | Este documento |

`data_version`: `features-0.1.0`

## Reconstruir tensor `[65, 3]`

En `windows_65.parquet`, la columna `features_flat` es una lista de
`195` floats en orden row-major: para cada `t` en `0..64`, los
canales `['log_return', 'log_high_low_range', 'log_volume']`.

```python
import numpy as np
import pandas as pd

w = pd.read_parquet("data/features/windows_65.parquet")
row = w.iloc[0]
X = np.asarray(row["features_flat"], dtype=np.float64).reshape(65, 3)
# X.shape == (65, 3); X[:, 0] == log_return, etc.
assert list(X.shape) == [65, 3]
```

## Verificar checksums

```bash
sha256sum -c data/features/checksums.sha256
```

## Usar features diarias

```python
import pandas as pd
f = pd.read_parquet("data/features/daily_features.parquet")
# columnas: date, ticker, log_return, log_high_low_range, log_volume
```

Fuera de alcance: splits de experimento, calibración/estandarización a NVDA,
entrenamiento de modelos.
