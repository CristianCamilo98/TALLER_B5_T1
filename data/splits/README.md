# Splits temporales de ventanas (`splits-0.1.0`)

Etiquetas de split por `window_end_date` sobre ventanas `features-0.2.0`.

**Solo asignación de splits.** Sin entrenamiento, sin sintéticos, sin calibración
NVDA, sin mixes, sin Ridge, sin recalcular features. No se modifica
`data/clean/` ni `data/features/`.

> **Mantenimiento:** cambios de protocolo deben actualizar **este README** y
> `scripts/assign_splits.py` (y regenerar artefactos + `split_manifest.json`).

## Contrato (congelado)

| Rol | Tickers |
|---|---|
| Donors | AMD, INTC, QCOM, AVGO, MU, TXN, ADI, MCHP, MRVL, NXPI |
| Target | NVDA |

Asignación por `window_end_date` (inclusivo), **idéntica en cada stride**:

1. `donor_train`: ticker ∈ donors AND `window_end_date` ∈ [2012-01-01, 2021-12-31]
2. `donor_val`: ticker ∈ donors AND `window_end_date` ∈ [2022-01-01, 2022-12-31]
3. `nvda_visible`: ticker == NVDA AND `window_end_date` ∈ [2022-07-01, 2022-12-31]
4. `nvda_test`: ticker == NVDA AND `window_end_date` ∈ [2023-01-01, 2025-12-31]
5. `unused`: todo lo demás

Reglas:

- Una ventana → una sola etiqueta (mutuamente excluyentes).
- NVDA **nunca** en `donor_train` ni `donor_val`.
- No se borran ventanas de features; el split es artefacto aparte.
- Hay un parquet de splits **por stride** (menú 1/10/30/65).

## Entrada (solo lectura; SHA verificados)

| Fichero | SHA256 |
|---|---|
| `windows_65_stride1.parquet` | `58bf4c4788cc4ae4feed14c2173419dea876a322c9e6f66d510c9dfb6c00bccf` |
| `windows_65_stride10.parquet` | `5bb0cd6ce8dd0fd1ff09fcafd56f0d124f6c8b8ab4a9e0a1ae21c006da754e9d` |
| `windows_65_stride30.parquet` | `6caf6893c6ab526d5ed95caa8d51f95cdd896211e070adc2f88bdcf34bf5225c` |
| `windows_65_stride65.parquet` | `1ce567ee7074050840e032a42c93d6e17d1b3523afb02b3ae098296db3556827` |

`features_manifest.json` debe reportar `data_version == features-0.2.0`.

## Salidas

| Artefacto | Descripción |
|---|---|
| `window_splits_stride{1,10,30,65}.parquet` | etiquetas por ventana |
| `split_manifest.json` | manifest `splits-0.1.0` |
| `checksums.sha256` | digests de salidas |
| `README.md` | este documento |

### Columnas de cada `window_splits_stride*.parquet`

| Columna | Notas |
|---|---|
| `ticker` | |
| `window_start_date` | |
| `window_end_date` | criterio de split |
| `window_row` | índice 0..n-1 alineado 1:1 con el parquet `windows_65_stride*` |
| `split` | una de: donor_train, donor_val, nvda_visible, nvda_test, unused |

Join canónico: `window_row` (mismo orden de filas que el windows del stride).
Join alternativo: `(ticker, window_start_date, window_end_date)` (única por stride).

## Conteos (n_windows por split × stride)

| stride | donor_train | donor_val | nvda_visible | nvda_test | unused |
|---|---:|---:|---:|---:|---:|
| 1 | 24512 | 2510 | 127 | 752 | 10096 |
| 10 | 2458 | 251 | 13 | 75 | 1009 |
| 30 | 820 | 89 | 5 | 25 | 337 |
| 65 | 380 | 40 | 2 | 12 | 160 |

### n_unused por stride

| stride | n_unused |
|---|---:|
| 1 | 10096 |
| 10 | 1009 |
| 30 | 337 |
| 65 | 160 |

## Protocolo de uso (para generadores)

- Entrenar generadores **solo** con `donor_train` (donors).
- Validar hiperparámetros / early-stop con `donor_val` (donors).
- `nvda_visible`: NVDA H2-2022 reservado (visible para inspección / calibración
  **fuera de este chat**).
- `nvda_test`: hold-out NVDA 2023–2025.
- `unused`: no usar en el protocolo oficial (p. ej. NVDA pre-2022-07, donors post-2022).

`primary_stride = 1` es solo la nota de dataset comparable del menú
de features; **existen splits para todos los strides**. Un compañero elige stride
y consume el `window_splits_stride{N}` correspondiente.

## Asserts

| Check | Resultado |
|---|---|
| Join/cardinalidad 1:1 con cada `windows_65_stride*` | OK |
| Partición exhaustiva por stride | OK |
| 0 NVDA en donor_train ∪ donor_val (cada stride) | OK |

## Fuera de alcance

Generación de sintéticos, calibración NVDA, mixes, Ridge y entrenamiento de
VAE/GAN/Diffusion **no** se hacen aquí (Chat de generadores / notebooks
posteriores).
