# Descarga OHLCV raw (`raw-0.1.0`)

Pipeline mínimo: yfinance → panel long crudo. **Sin limpieza, sin features.**

## Contrato

| Campo | Valor |
|---|---|
| `data_version` | `raw-0.1.0` |
| Fuente | yfinance |
| `auto_adjust` | `true` |
| Target | NVDA |
| Donors | AMD, INTC, QCOM, AVGO, MU, TXN, ADI, MCHP, MRVL, NXPI |
| Ventana | start=`2012-01-03`, end inclusivo solicitado=`2025-12-31` |
| Columnas | `date`, `ticker`, `Open`, `High`, `Low`, `Close`, `Volume` |

**End exclusivo de yfinance:** la API trata `end` como exclusivo. El script pasa `end=2026-01-01` para incluir sesiones del `2025-12-31` cuando existan. Detalle en `download_manifest.json` → `yfinance_params`.

Parámetros de universo/calendario: [`configs/data_contract.yaml`](../../configs/data_contract.yaml).

## Entorno

```bash
cd TALLER_B5_T1
# opción A: uv
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python

# opción B: venv + pip
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencias: `requirements.txt` (`yfinance`, `pandas`, `pyarrow`, `PyYAML`).

## Ejecutar descarga

```bash
.venv/bin/python scripts/download_ohlcv_raw.py
```

Si un ticker falla o vuelve vacío, el script **termina con error** (no silencia).

## Artefactos

| Ruta | Contenido |
|---|---|
| `data/raw/ohlcv_raw.parquet` | Panel long (todos los tickers) |
| `data/raw/ohlcv_raw.csv` | Misma tabla en CSV |
| `data/raw/by_ticker/<TICKER>.parquet` | Un fichero por ticker |
| `data/raw/download_manifest.json` | Metadatos, n_rows, nulls, rutas, SHA256 |

## Verificar checksums

Desde la raíz del repo:

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path("data/raw/download_manifest.json").read_text())
ok = True
for rel, expected in m["checksums_sha256"].items():
    h = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    status = "OK" if h == expected else "MISMATCH"
    ok = ok and status == "OK"
    print(f"{status}  {rel}")
raise SystemExit(0 if ok else 1)
PY
```

Solo el panel principal:

```bash
sha256sum data/raw/ohlcv_raw.parquet
# comparar con checksums_sha256["data/raw/ohlcv_raw.parquet"] en download_manifest.json
```

Si el SHA no coincide tras re-descargar: yfinance puede haber cambiado revisiones; repartid el parquet del equipo como fuente de verdad.
