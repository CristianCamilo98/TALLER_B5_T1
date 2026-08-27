# Splits diarios (`daily-splits-1.0.0`)

Este bloque asigna el rol temporal a cada fila del panel limpio **antes** de
calcular ventanas. Produce `daily_split_assignments.parquet` con clave única
`(date,ticker)` y una de estas etiquetas:

- `donor_train`
- `donor_validation`
- `nvda_hidden`
- `nvda_visible`
- `nvda_test`
- `unused`

No crea features ni ventanas. Las reglas proceden de
`configs/experiment.yaml`; no existen fechas o tickers duplicados en el script.

Artefactos: `daily_split_assignments.parquet`, `daily_split_report.csv`,
`split_manifest.json` y `checksums.sha256`.
