# LEGACY — DO NOT EXECUTE FOR CURRENT EXPERIMENT

`04b_inspect_splits.ipynb` conserva exclusivamente trazabilidad histórica del
protocolo `splits-0.1.0`, anterior a `common-core-1.0.0`.

Las referencias a ventanas globales, al menú de strides `1, 10, 30, 65`, a
`primary_stride=1` y a la asignación posterior por `window_end_date` describen
únicamente el diseño retirado por riesgo de leakage. No son reglas ni entradas
válidas del experimento actual.

Todas las antiguas celdas de código se conservan como Markdown no ejecutable.
Este notebook no forma parte del pipeline canónico y no debe usarse para
regenerar, inspeccionar ni validar los artefactos actuales.

Fuentes vigentes:

- `configs/experiment.yaml`
- `scripts/assign_splits.py`
- `scripts/build_features_windows.py`
- `data/splits/README.md`
- `COMMON_CORE_CERTIFICATION.md`
- `tests/`
