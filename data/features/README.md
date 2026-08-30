# Features y ventanas post-split (`features-windows-1.0.0`)

Las features se calculan separadamente dentro de cada bloque diario. La primera
fila de cada ticker/bloque no tiene `log_return` y se elimina dentro del bloque;
nunca se toma el cierre anterior a una frontera.

Canales, en orden fijo:

1. `log_return = ln(Close_t / Close_t-1)`
2. `log_high_low_range = ln(High_t / Low_t)`
3. `log1p_volume = ln(1 + Volume_t)`

Artefactos:

- `daily_features_by_split.parquet`
- `windows/donor_train.parquet`
- `windows/donor_validation.parquet`
- `windows/nvda_visible.parquet`
- `windows/nvda_full_history.parquet`
- `test_index.parquet`
- `window_counts.csv`, manifest y checksums

Cada ventana estándar contiene 195 floats (`65×3`). `test_index` añade las 60
fechas de contexto y las cinco fechas target explícitas; targets consecutivos
no se solapan.

Los artefactos **LEGACY** `windows_65_stride{1,10,30,65}` no son canónicos y se archivaron
en `data/legacy_pre_realignment/`.
