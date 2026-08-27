# Common Core Certification — Synthetic NVDA

Fecha de certificación: 2026-08-27  
Protocolo: `common-core-1.0.0`  
Rama: `feature/common-protocol-realignment`

## Alcance

Certifica configuración, snapshot raw, limpieza, asignación diaria, features,
ventanas y test index. Excluye expresamente modelos generativos, Gaussian Noise,
Ridge y cualquier evaluación downstream.

## Evidencia del snapshot

- Raw: 38.720 filas, 11 tickers, 0 duplicados y 0 NaN OHLCV.
- Clean: 38.720 filas; 7 bordes flotantes aceptados; 0 violaciones materiales.
- `log1p_volume` mantiene la fila AMD 2015-01-02 con volumen cero.
- Las features se reinician dentro de cada bloque diario.

## Fronteras certificadas

| Split | Primera fecha de ventana | Última fecha de ventana/target | N | Resultado |
|---|---|---|---:|---|
| donor_train | 2012-01-04 | 2021-12-30 | 4.910 | PASS |
| donor_validation | 2022-01-04 | 2022-12-30 | 380 | PASS |
| nvda_visible | 2022-07-05 | 2022-12-30 | 62 | PASS |
| nvda_full_history | 2012-01-04 | 2022-12-30 | 2.703 | PASS |
| nvda_test targets | 2023-01-03 | 2025-12-29 | 150 | PASS |

La diferencia entre la fecha de inicio configurada y la primera feature de cada
bloque se debe al retorno inicial indefinido. No se utiliza el cierre previo al
corte para completarlo.

## Pruebas

Comando: `python -m pytest -q`

Resultado: `9 passed`.

Cobertura funcional:

- contrato YAML congelado;
- tolerancia OHLC frente a error material;
- `log1p_volume` y conservación de volumen cero;
- unicidad y fronteras del split diario;
- shape, finitud, stride y fronteras de cada ventana;
- 60 días de contexto + 5 targets;
- targets íntegramente dentro de test y no solapados;
- recuentos y checksums contra manifests;
- ausencia de artefactos globales obsoletos en rutas canónicas.

## Dictamen

`CERTIFIED`, condicionado a que no se cambien `configs/experiment.yaml`, el
snapshot de datos o los scripts sin regenerar toda la cadena y volver a ejecutar
los tests. La implementación de modelos puede comenzar en una fase posterior,
pero no forma parte de esta certificación.
