# OHLCV limpio (`clean-0.1.0`)

Panel long validado a partir de `data/raw/ohlcv_raw.parquet`. **Solo limpieza/validación.**  
Sin features, sin `log_return`, sin ventanas, sin splits, sin entrenamiento.

> **Mantenimiento:** cualquier cambio futuro en reglas de limpieza o validación debe
> actualizar **este README** y `scripts/clean_ohlcv.py` (y regenerar artefactos +
> `clean_manifest.json`).

## Entrada (solo lectura)

| Campo | Valor |
|---|---|
| Ruta | `data/raw/ohlcv_raw.parquet` |
| SHA256 esperado | `6ecd4c929ecd3bdca32c646aec8210a7757b566843a90102f21bd86d2da036d6` |
| `data/raw/` | **no se modifica** |

Si el SHA del parquet de entrada no coincide, el script **aborta** (no re-descarga).

## Reglas de limpieza / validación

Aplicadas en este orden por `scripts/clean_ohlcv.py`:

1. **Sin forward-fill** de precios ni volumen.
2. **Drop filas** con NaN en `Open` / `High` / `Low` / `Close`.
3. **Drop filas** que rompan invariantes OHLC:
   - `High >= Low`
   - `High >= Open` y `High >= Close`
   - `Low <= Open` y `Low <= Close`
   - `Volume >= 0`
4. **Normalizar `date`** a fecha naive (sin timezone) y **ordenar** por `(date, ticker)`.
5. **Cobertura por ticker:**
   - Calendario = **unión de fechas del panel tras** drops NaN + invariantes
     (antes de excluir tickers por cobertura).
   - `cobertura(ticker) = n_fechas_distintas(ticker) / |calendario|`
   - Si `cobertura < 0.95` → **excluir** el ticker del panel limpio y registrarlo
     en `clean_manifest.json` → `dropped_tickers_low_coverage`.
6. **No inventar filas.** **No interpolar.**

Fuera de alcance de este paso: features, ventanas de 65 días, splits de experimento,
descarga yfinance, entrenamiento.

## Columnas del panel limpio

`date`, `ticker`, `Open`, `High`, `Low`, `Close`, `Volume`  
Formato long: una fila = un ticker × un día de sesión.

## Ejecutar

```bash
cd taller_cristian
.venv/bin/python scripts/clean_ohlcv.py
```

## Artefactos

| Ruta | Contenido |
|---|---|
| `data/clean/ohlcv_clean.parquet` | Panel limpio |
| `data/clean/ohlcv_clean.csv` | Misma tabla en CSV |
| `data/clean/clean_manifest.json` | Metadatos, drops, cobertura, SHA |
| `data/clean/checksums.sha256` | SHA256 de parquet y CSV limpios |
| `data/clean/README.md` | Este documento |

`data_version`: `clean-0.1.0`

## Verificar checksums

Desde la raíz del repo:

```bash
sha256sum -c data/clean/checksums.sha256
```

O contra el manifest:

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path("data/clean/clean_manifest.json").read_text())
ok = True
for rel, expected in m["checksums_sha256"].items():
    h = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    status = "OK" if h == expected else "MISMATCH"
    ok = ok and status == "OK"
    print(f"{status}  {rel}")
print("input verified:", m["input_sha256_verified"] == m["input_sha256_expected"])
raise SystemExit(0 if ok else 1)
PY
```

## Usar el limpio

```python
import pandas as pd
df = pd.read_parquet("data/clean/ohlcv_clean.parquet")
# columnas: date, ticker, Open, High, Low, Close, Volume
```

Para el siguiente paso del pipeline (features / ventanas), consumir **solo** estos
artefactos limpios, no `data/raw/`.
