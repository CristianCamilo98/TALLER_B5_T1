# OHLCV limpio (`clean-0.2.0`)

Entrada: snapshot certificado por `data/raw/download_manifest.json`.

Reglas:

- sin forward-fill, interpolación ni filas inventadas;
- fechas y clave `(date,ticker)` únicas;
- NaN en OHLCV se eliminan y registran;
- OHLC se valida con tolerancia absoluta `1e-10` y relativa `1e-12`;
- volumen debe ser no negativo; `Volume=0` es válido;
- cobertura mínima por ticker: 95 %.

En el snapshot actual las diferencias OHLC de orden `1e-15` registradas en
`quality_report.csv` quedan correctamente aceptadas y no hay violaciones
materiales. Resultado:
38.720 → 38.720 filas.

Artefactos: `ohlcv_clean.parquet`, `ohlcv_clean.csv`, `quality_report.csv`,
`clean_manifest.json` y `checksums.sha256`.
