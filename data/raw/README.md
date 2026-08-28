# Snapshot raw (`raw-0.2.0`)

Descarga OHLCV diaria de los once tickers definidos exclusivamente en
`configs/experiment.yaml`, con `auto_adjust=true` y final inclusivo
`2025-12-31` (`end=2026-01-01` enviado a yfinance).

`download_manifest.json` registra parámetros, versión de protocolo y SHA256
del snapshot observado. El hash describe esa descarga; no se exige que una
descarga futura de yfinance produzca los mismos bytes.

Para reconstruir solo el manifest sin red:

```bash
python scripts/download_ohlcv_raw.py --config configs/experiment.yaml --reuse-snapshot
```

Para descargar un snapshot nuevo, omitir `--reuse-snapshot` y regenerar después
clean → daily splits → features/windows.
