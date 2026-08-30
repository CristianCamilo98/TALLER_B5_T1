# Taller B5-T1 — Synthetic NVDA Common Core

Common core certificado para el experimento de ampliación sintética de NVDA.
Esta rama contiene exclusivamente configuración, descarga, limpieza, splits
diarios, features, ventanas y pruebas anti-leakage. **No implementa todavía**
VAE, WGAN-GP, Diffusion, Gaussian Noise, Ridge ni evaluación downstream.

La única fuente de verdad metodológica es
[`configs/experiment.yaml`](configs/experiment.yaml).

## Orden obligatorio del pipeline

```text
configs/experiment.yaml
        │
        ▼
1. download_ohlcv_raw.py       snapshot OHLCV + manifest
        │
        ▼
2. clean_ohlcv.py              tolerancia OHLC + quality report
        │
        ▼
3. assign_splits.py            asignación temporal a NIVEL DIARIO
        │
        ▼
4. build_features_windows.py   features DENTRO de cada bloque
                                y después ventanas con stride oficial
        │
        ▼
5. pytest                      certificación de fronteras y leakage
```

Está prohibido construir ventanas globales y etiquetarlas posteriormente por
`window_end_date`.

## Contrato congelado

| Elemento | Valor |
|---|---|
| Target | NVDA |
| Donors | AMD, INTC, QCOM, AVGO, MU, TXN, ADI, MCHP, MRVL, NXPI |
| Donor train | 2012-01-03..2021-12-31, stride 5 |
| Donor validation | 2022-01-03..2022-12-30, stride 5 |
| NVDA visible | 2022-07-01..2022-12-30, stride 1 |
| Full-history benchmark | 2012-01-03..2022-12-30, stride 1 |
| Test targets | 2023-01-03..2025-12-31, context 60, horizon 5, stride 5 |
| Canales | log_return, log_high_low_range, log1p_volume |
| Ventana | 65 × 3 |
| Seeds futuras | 42, 123, 2026 |
| Ratios futuros | 25 %, 50 %, 75 % |
| Downstream futuro | Ridge alpha=1.0; contrato únicamente, no implementado |

## Recuentos del snapshot certificado

| Split | Ventanas | Stride |
|---|---:|---:|
| donor_train | 4.910 | 5 |
| donor_validation | 380 | 5 |
| nvda_visible | 62 | 1 |
| nvda_full_history | 2.703 | 1 |
| nvda_test (`test_index`) | 150 | 5 |

Los 150 targets de test contienen exactamente cinco sesiones, no se solapan y
están íntegramente dentro de 2023-01-03..2025-12-31.

## Artefactos canónicos

| Etapa | Artefactos |
|---|---|
| Raw | `data/raw/ohlcv_raw.*`, `download_manifest.json` |
| Clean | `data/clean/ohlcv_clean.*`, `quality_report.csv`, manifest/checksums |
| Split diario | `data/splits/daily_split_assignments.parquet`, reporte/manifest/checksums |
| Features | `data/features/daily_features_by_split.parquet` |
| Ventanas | `data/features/windows/{split}.parquet` |
| Test común | `data/features/test_index.parquet` |
| Certificación | `COMMON_CORE_CERTIFICATION.md`, `tests/` |

Los artefactos globales pre-realineación se conservan localmente, fuera del
pipeline canónico, en `data/legacy_pre_realignment/`.

## Ejecutar sin volver a descargar

```bash
python scripts/download_ohlcv_raw.py --config configs/experiment.yaml --reuse-snapshot
python scripts/clean_ohlcv.py --config configs/experiment.yaml
python scripts/assign_splits.py --config configs/experiment.yaml
python scripts/build_features_windows.py --config configs/experiment.yaml
python -m pytest -q
```

Omitir `--reuse-snapshot` realiza una descarga nueva de yfinance.

## Política de checksums

Los SHA256 certifican el snapshot observado por cada manifest. Cada etapa
verifica que su entrada coincide con el manifest inmediatamente anterior. Los
scripts no contienen hashes metodológicos hardcodeados y una descarga futura
válida puede producir un SHA diferente: al regenerar la cadena se genera un
nuevo linaje de manifests.

## Estado

Common core `CERTIFIED` para el snapshot observado. La evidencia exacta y las condiciones de
certificación están en [`COMMON_CORE_CERTIFICATION.md`](COMMON_CORE_CERTIFICATION.md).
