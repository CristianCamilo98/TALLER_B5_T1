# Features diarias + ventanas T=65 multi-stride (`features-0.2.0`)

Panel de features generativas diarias y **menú de datasets de ventanas** (varios
strides) a partir de `data/clean/ohlcv_clean.parquet`.

**Solo features + ventanas.** Sin splits, sin calibración a NVDA, sin
estandarización, sin entrenamiento (VAE/GAN/Diffusion/Ridge). Sin re-limpieza
ni escritura en `data/clean/`.

Este repo prepara datos comunes para que Marco (VAE), Cristian (GAN) y Dani
(Diffusion) entrenen sintéticos **sin reconstruir el pipeline**. Cada uno elige
más adelante qué stride(s) usar; este chat **no** decide el entrenamiento.

> **Mantenimiento:** cualquier cambio en fórmulas o políticas debe actualizar
> **este README** y `scripts/build_features_windows.py` (y regenerar artefactos +
> `features_manifest.json`).

## Canónico en 0.2.0 (vs legado 0.1.0)

| Versión | Artefacto de ventanas |
|---|---|
| **0.2.0 (canónico)** | `windows_65_stride{1,10,30,65}.parquet` — **un fichero por stride** |
| 0.1.0 (legado) | `windows_65.parquet` (stride=1 implícito) — **eliminado** en esta versión |

No uses `windows_65.parquet`. No mezcles distintos strides en un mismo train
sin acuerdo explícito del equipo (y sin columna `stride` si fuera un fichero único;
aquí preferimos ficheros separados).

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
- Filas `Volume == 0` se eliminan **antes** de calcular features.
- Primera fila de cada ticker sin `log_return` → fuera.
- Drop NaN/inf en las 3 features.
- **No** ffill. **No** winsorize. **No** estandarizar.
- Huecos vs unión de fechas: ventanas sobre **filas consecutivas del panel de
  features**, no días de calendario rellenados.

## Menú de ventanas (multi-stride)

| Parámetro | Valor |
|---|---|
| `T` | 65 |
| `strides` | `[1, 10, 30, 65]` |
| `primary_stride` | **1** (DEFAULT RECOMENDADO para corrida oficial comparable) |
| Canales (orden fijo) | `['log_return', 'log_high_low_range', 'log_volume']` → shape `[65, 3]` |
| Ámbito | un solo ticker por ventana |
| Metadatos | `ticker`, `window_start_date`, `window_end_date` |

### Qué significa el stride

El stride es cuántas filas de features avanzas al crear la **siguiente** ventana.
Con `T=65` y `stride=1`, las ventanas se solapan en 64 días. Con `stride=65`,
son bloques disjuntos (sin solape).

### Fórmula de conteo (por ticker y stride)

Si un ticker tiene `n` filas de features:

```
n_windows(ticker, stride) = floor((n - T) / stride) + 1   si n >= T
n_windows(ticker, stride) = 0                              si n < T
```

Total por stride = suma sobre tickers.

### Ficheros del menú

| `1` | `data/features/windows_65_stride1.parquet` | **← primary / default recomendado**
| `10` | `data/features/windows_65_stride10.parquet` |
| `30` | `data/features/windows_65_stride30.parquet` |
| `65` | `data/features/windows_65_stride65.parquet` |

`primary_stride=1` es solo el default recomendado comparable del
equipo; los otros strides quedan disponibles para experimentos individuales.

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
| `data/features/windows_65_stride*.parquet` | Ventanas por stride + `features_flat` |
| `data/features/features_manifest.json` | Metadatos, conteos, SHA, políticas |
| `data/features/checksums.sha256` | SHA256 de salidas |
| `data/features/README.md` | Este documento |

`data_version`: `features-0.2.0`

## Reconstruir tensor `[65, 3]`

En cada `windows_65_stride*.parquet`, la columna `features_flat` es una lista de
`195` floats en orden row-major: para cada `t` en `0..64`, los
canales `['log_return', 'log_high_low_range', 'log_volume']`.

```python
import numpy as np
import pandas as pd

w = pd.read_parquet("data/features/windows_65_stride1.parquet")  # o stride10/30/65
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
entrenamiento de modelos generativos.
