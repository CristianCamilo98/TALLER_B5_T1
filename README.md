# Taller B5-T1 — Datos compartidos (prep. pre-sintéticos)

Pipeline común de **preparación de datos** para el taller de generación de datos financieros sintéticos.

Este repo deja listos paneles, features, ventanas y splits para que cada integrante se centre en su **modelo generativo**, sin reconstruir la base de datos.


| Persona  | Modelo generativo |
| -------- | ----------------- |
| Marco    | Time VAE          |
| Cristian | GAN               |
| Dani     | Diffusion         |
| David    | Normaflow         |


**Estado actual:** preparación de datos **antes** de sintéticos = **cerrada** (`raw` → `clean` → `features-0.2.0` → `splits-0.1.0`).

Enunciado: `[Taller_B5_T1.pdf](Taller_B5_T1.pdf)`.

---



## Pipeline por bloques

```text
configs/data_contract.yaml
        │
        ▼
[1] Descarga          scripts/download_ohlcv_raw.py
        │               → data/raw/                 (raw-0.1.0)
        ▼
[2] Limpieza          scripts/clean_ohlcv.py
        │               → data/clean/               (clean-0.1.0)
        ▼
[3] Features+ventanas scripts/build_features_windows.py
        │               → data/features/            (features-0.2.0)
        │                 daily_features + windows_65_stride{1,10,30,65}
        ▼
[4] Splits            scripts/assign_splits.py
                        → data/splits/              (splits-0.1.0)
                          window_splits_stride{1,10,30,65}
```

Cada bloque:

- tiene **script canónico** (reproduce el artefacto),
- escribe **manifest + checksums SHA256**,
- documenta reglas en su `data/<bloque>/README.md`,
- no debe mutar salidas de bloques anteriores (solo lectura aguas arriba).



### Notebooks de inspección (no regeneran datos)


| Notebook                                                                                       | Qué explica                       |
| ---------------------------------------------------------------------------------------------- | --------------------------------- |
| `[notebooks/02b_audit_cleaning.ipynb](notebooks/02b_audit_cleaning.ipynb)`                     | Auditoría raw vs clean            |
| `[notebooks/02c_eda_clean_panel.ipynb](notebooks/02c_eda_clean_panel.ipynb)`                   | EDA del panel limpio (playground) |
| `[notebooks/03b_inspect_features_windows.ipynb](notebooks/03b_inspect_features_windows.ipynb)` | Features + menú multi-stride      |
| `[notebooks/04b_inspect_splits.ipynb](notebooks/04b_inspect_splits.ipynb)`                     | Splits temporales                 |


Cómo abrir cada uno: `notebooks/README_02b.md`, `README_02c.md`, `README_03b.md`, `README_04b.md`.

---



## Estructura del repositorio

```text
taller_cristian/
├── README.md                 ← este fichero (mapa global)
├── Taller_B5_T1.pdf
├── requirements.txt
├── configs/
│   └── data_contract.yaml    ← universo / fechas descarga
├── scripts/
│   ├── download_ohlcv_raw.py
│   ├── clean_ohlcv.py
│   ├── build_features_windows.py
│   └── assign_splits.py
├── data/
│   ├── raw/                  + README.md, manifest, checksums
│   ├── clean/                + README.md, manifest, checksums
│   ├── features/             + README.md, manifest, checksums
│   └── splits/               + README.md, manifest, checksums
└── notebooks/
    ├── 02b_… 03b_… 04b_… 02c_…
    ├── README_0xb.md
    └── figures/              (opcionales, p.ej. EDA 02c)
```

---



## Cómo usar los datos (handoff generadores)

**Recomendado para la corrida oficial comparable entre VAE/GAN/Diffusion:**


| Pieza                            | Ruta                                        |
| -------------------------------- | ------------------------------------------- |
| Ventanas                         | `data/features/windows_65_stride1.parquet`  |
| Splits                           | `data/splits/window_splits_stride1.parquet` |
| Features diarias (si hace falta) | `data/features/daily_features.parquet`      |


Protocolo:

1. Elige **un** stride y usa el **mismo** en features y splits. No mezcles strides en un mismo train sin acuerdo del equipo.
2. Join ventanas ↔ splits (p. ej. vía `window_row` 0..n-1 alineado, como en el notebook 04b).
3. Entrena el generador solo con `split == donor_train`.
4. Checkpoint / early stop con `donor_val` (donors 2022; **sin NVDA**).
5. **No** uses `nvda_test` para tunear.
6. `nvda_visible` y calibración / mixes = fase posterior.
7. `unused` no entra en el protocolo oficial (es grande a propósito).

Menú de strides disponibles: `1`, `10`, `30`, `65`.  
`primary_stride=1` = default recomendado (más muestra). Strides altos dejan muy pocas ventanas `nvda_visible` (p. ej. stride 65 → solo 2).

Forma de una ventana: `[65, 3]` con canales  
`[log_return, log_high_low_range, log_volume]`  
(ver `data/features/README.md` y `features_manifest.json`).

Detalle de splits y fechas: `[data/splits/README.md](data/splits/README.md)`.

---



## Entorno

```bash
cd TALLER_B5_T1
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python
# o: python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Regenerar un bloque (solo si cambias reglas; ver sección siguiente):

```bash
.venv/bin/python scripts/download_ohlcv_raw.py      # → data/raw/
.venv/bin/python scripts/clean_ohlcv.py             # → data/clean/
.venv/bin/python scripts/build_features_windows.py  # → data/features/
.venv/bin/python scripts/assign_splits.py           # → data/splits/
```

Los scripts de bloques 2–4 **verifican SHA** de su entrada y abortan si no coincide (no “arreglan” re-descargando).

---



## Documentación por bloque (fuente de verdad local)


| Bloque              | README de detalle                                    | Script                              | Versión          |
| ------------------- | ---------------------------------------------------- | ----------------------------------- | ---------------- |
| Descarga            | `[data/raw/README.md](data/raw/README.md)`           | `scripts/download_ohlcv_raw.py`     | `raw-0.1.0`      |
| Limpieza            | `[data/clean/README.md](data/clean/README.md)`       | `scripts/clean_ohlcv.py`            | `clean-0.1.0`    |
| Features / ventanas | `[data/features/README.md](data/features/README.md)` | `scripts/build_features_windows.py` | `features-0.2.0` |
| Splits              | `[data/splits/README.md](data/splits/README.md)`     | `scripts/assign_splits.py`          | `splits-0.1.0`   |


Este `README.md` raíz = **mapa**. Las reglas exactas (fórmulas, drops, fechas de split) viven en el README + manifest de cada bloque.

---



## Si quieres cambiar el pipeline con IA (o a mano)



### Principios

1. **Un bloque por cambio.** No pidas “arregla features y splits y limpia” en el mismo chat.
2. **Lee antes el README del bloque** que vas a tocar (`data/<bloque>/README.md`) y el script asociado.
3. **Aguas abajo se invalidan.** Si cambias `clean`, debes regenerar `features` y luego `splits` (y actualizar notebooks de inspección si aplica).
4. **No mutar canónicos aguas arriba.** El bloque N solo lee el artefacto N−1.
5. **Bump de** `data_version` en manifest + README del bloque si cambias semántica (no solo un bugfix cosmético).
6. **SHA:** tras regenerar, actualiza checksums/manifests; quien consuma datos debe verificar SHA.
7. **Notebooks no son la fuente de verdad.** Explican/inspeccionan; los scripts generan artefactos.



### Prompt mínimo recomendado para un chat de IA

Copia y adapta:

```text
Estás en el repo del Taller B5-T1 (prep. datos pre-sintéticos).
Alcance: SOLO el bloque <raw|clean|features|splits>.
1) Lee data/<bloque>/README.md y scripts/<script>.py
2) Lee el README raíz para ver dependencias aguas abajo
3) Aplica el cambio pedido: <describe el cambio>
4) Regenera artefactos de ESE bloque; actualiza README + manifest + checksums
5) NO toques otros bloques salvo que diga explícitamente regenerar la cadena
6) Al final: lista ficheros tocados, nuevo data_version, nuevos SHA, y qué bloques hay que regenerar después
```



### Cadena de regeneración


| Si cambias…                                   | Regenerar después                        |
| --------------------------------------------- | ---------------------------------------- |
| `raw` (contrato tickers/fechas / re-descarga) | clean → features → splits                |
| `clean` (reglas de limpieza)                  | features → splits                        |
| `features` (fórmulas, T, strides)             | splits                                   |
| `splits` (fechas de protocolo)                | (nada de datos; sí revisar notebook 04b) |


**Nota:** re-descargar `raw` puede cambiar SHA aunque el código sea igual (yfinance). Para trabajo en equipo, el parquet versionado + SHA es la fuente de verdad compartida; no asumas que dos máquinas regeneran bytes idénticos sin verificar.
