# Common phase 01 — output contract

Audita los outputs oficiales en `generadores/*/outputs/*.parquet`, certifica
el contrato lógico `(5000, 65, 3)` en espacio `global_channel_normalized` y
genera el baseline obligatorio `bootstrap_jitter`.

## Ejecución

Desde la raíz del repositorio:

```bash
.venv/bin/python common_pipeline/01_contract/validate_outputs.py
.venv/bin/python common_pipeline/01_contract/build_simple_baseline.py
.venv/bin/pytest -q common_pipeline/01_contract/tests
```

## Salidas

- `results/output_contract_report.csv`
- `results/output_contract_report.json`
- `outputs/bootstrap_jitter_seed42_normalized.parquet`

## Reglas de descubrimiento

- Debe existir **exactamente un** parquet oficial por generador.
- Deben existir **cuatro** generadores con output.
- Si hay más/menos outputs, el módulo **falla** y lo reporta explícitamente.

## Alcance

Este módulo **no** hace fidelity, calibración NVDA, downstream, C2ST ni t-SNE.
