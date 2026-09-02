# DDPM — Denoising Diffusion Probabilistic Model

**Generación de ventanas financieras sintéticas mediante difusión temporal.**

El DDPM es uno de los cuatro generadores neuronales comparados en el proyecto. Su función es aprender la distribución de ventanas financieras de compañías donantes del sector semiconductor y producir escenarios sintéticos en el mismo espacio normalizado común.

El problema experimental es de escasez de datos: **si sólo dispusiéramos de seis meses de historia visible de NVDA, ¿puede el _synthetic data augmentation_ mejorar la predicción de volatilidad futura?** Las ventanas generadas por el DDPM se incorporan posteriormente al entrenamiento de un modelo downstream común, manteniendo fija su arquitectura y su test.

El DDPM se entrena exclusivamente con datos de empresas donantes. **NVDA no interviene en el entrenamiento, la validación ni la selección del checkpoint del generador.**

Responsable de implementación: Daniel.

## Papel en el experimento

```mermaid
flowchart TD
    A[Compañías donantes<br/>sector semiconductor] --> B[Ventanas canónicas<br/>65 sesiones × 3 canales]
    B --> C[Normalización global por canal<br/>fit sólo en donor_train]
    C --> D[DDPM temporal<br/>noise prediction]
    D --> E[5.000 ventanas sintéticas<br/>seed oficial 42]
    E --> F[Fidelity común<br/>vs donor_validation, no NVDA]
    E --> G[Utility común<br/>calibración y validación física]
    G --> H[Augmentation<br/>25% / 50% / 75%]
    H --> I[Ridge alpha=1<br/>misma arquitectura y 8 features]
    I --> J[Test futuro NVDA<br/>150 ventanas, 2023–2025]

    K[donor_validation<br/>380 ventanas] --> L[Selección por mínima<br/>validation loss]
    L --> D
```

La frontera de aprendizaje es explícita: `donor_train` ajusta el modelo y el normalizador; `donor_validation` selecciona el checkpoint. NVDA aparece sólo después del DDPM, durante la calibración y evaluación downstream, nunca para aprender sus parámetros ni escoger su checkpoint.

## Contrato de datos

Los inputs congelados se describen en [`data/CANONICAL_EXPERIMENT_DATA.md`](../../data/CANONICAL_EXPERIMENT_DATA.md).

| Split | Ventanas | Tensor lógico por ventana | Uso en el DDPM |
|---|---:|---:|---|
| `donor_train` | 4.910 | `65 × 3` | Entrenamiento y fit del normalizador |
| `donor_validation` | 380 | `65 × 3` | Validation loss, early stopping y checkpoint selection |

Los tres canales aparecen siempre en este orden:

| Índice | Canal | Definición |
|---:|---|---|
| 0 | `log_return` | $\log(\mathrm{Close}_t / \mathrm{Close}_{t-1})$ |
| 1 | `log_high_low_range` | $\log(\mathrm{High}_t / \mathrm{Low}_t)$ |
| 2 | `log1p_volume` | $\log(1 + \mathrm{Volume}_t)$ |

El tensor público conserva el layout `(batch, 65, 3)`. La red transpone internamente a `(batch, 3, 65)` para aplicar `Conv1D` y devuelve la predicción al layout original.

## Normalización canónica

La implementación [`temporary_normalizer.py`](src/temporary_normalizer.py) aplica un z-score global por canal. Las estadísticas se ajustan **sólo con `donor_train`**, en `float64`, sobre los ejes `(0, 1)` y con `ddof=0`. Si $\sigma < 10^{-8}$, se sustituye por `1.0`; el tensor entregado a la red es `float32`. `donor_validation` es únicamente transformado con esas estadísticas congeladas.

| Canal | Mean | Standard deviation |
|---|---:|---:|
| `log_return` | `0.00081142897100880656` | `0.023515504591060377` |
| `log_high_low_range` | `0.026025805148914841` | `0.016724288791728319` |
| `log1p_volume` | `16.06027218135258` | `1.0933253360280637` |

La normalización pone canales con escalas muy distintas en un rango numérico comparable. Esto estabiliza la predicción del ruido y evita que la loss quede dominada por el canal de volumen.

## Concepto DDPM

El **forward process** añade ruido gaussiano progresivamente a una ventana real normalizada $x_0$. Para un timestep $t$, la implementación usa la forma cerrada:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
$$

La red aprende a predecir el ruido aplicado:

$$
\mathcal{L} = \mathbb{E}_{x_0,t,\epsilon}
\left[\left\lVert \epsilon - \epsilon_\theta(x_t,t) \right\rVert_2^2\right].
$$

En este proyecto:

- $x_0$ es una ventana financiera normalizada de `65 × 3`;
- $x_t$ es esa ventana después de añadir el nivel de ruido correspondiente a $t$;
- $t$ identifica uno de los 100 pasos de difusión;
- $\epsilon$ es el ruido gaussiano conocido usado para construir $x_t$;
- $\epsilon_\theta(x_t,t)$ es el ruido estimado por la red temporal.

Durante el **reverse process**, el sampler parte de $x_T \sim \mathcal{N}(0,I)$ y recorre los timesteps en orden inverso. En cada paso utiliza $\epsilon_\theta$ para calcular la media posterior y añade la varianza posterior correspondiente; en $t=0$ no añade ruido adicional.

## Arquitectura implementada

La red definida en [`network.py`](src/network.py) es un denoiser temporal Conv1D que preserva la longitud. No contiene attention, Transformer, U-Net, latent diffusion ni conditioning externo.

```mermaid
flowchart TD
    X["x_t: B × 65 × 3"] --> T1[Transpose<br/>B × 3 × 65]
    T1 --> IC[Input Conv1D<br/>3 → 64, kernel 3]

    TS[timestep t] --> SE[Sinusoidal embedding<br/>dimensión 128]
    SE --> TM[Linear 128 → 128<br/>SiLU<br/>Linear 128 → 128]

    IC --> R1[Residual block 1<br/>64 → 64]
    R1 --> R2[Residual block 2<br/>64 → 128]
    R2 --> R3[Residual block 3<br/>128 → 128]
    R3 --> R4[Residual block 4<br/>128 → 64]
    TM -. timestep projection .-> R1
    TM -. timestep projection .-> R2
    TM -. timestep projection .-> R3
    TM -. timestep projection .-> R4
    R4 --> ON[GroupNorm + SiLU]
    ON --> OC[Output Conv1D<br/>64 → 3, kernel 3]
    OC --> Y["epsilon_hat: B × 65 × 3"]
```

| Componente | Configuración real | Propósito |
|---|---|---|
| Timestep embedding | Sinusoidal, dimensión 128; MLP `Linear–SiLU–Linear` | Representar el nivel de ruido $t$ |
| Input projection | `Conv1D(3, 64, kernel=3, padding=1)` | Proyectar los tres canales al espacio oculto |
| Residual block 1 | `64 → 64`, kernel 3 | Procesamiento temporal condicionado por $t$ |
| Residual block 2 | `64 → 128`, kernel 3, skip 1×1 | Aumentar capacidad sin cambiar longitud |
| Residual block 3 | `128 → 128`, kernel 3 | Procesamiento temporal en alta dimensión |
| Residual block 4 | `128 → 64`, kernel 3, skip 1×1 | Volver a la dimensión base |
| Normalización/activación | GroupNorm y SiLU | Estabilizar las activaciones residuales |
| Output projection | `Conv1D(64, 3, kernel=3, padding=1)` | Predecir un ruido por sesión y canal |
| Parámetros entrenables | `336259` | Total de parámetros de `TemporalDenoiser` |

Cada bloque aplica `GroupNorm → SiLU → Conv1D`, suma una proyección aprendible del timestep, aplica un segundo `GroupNorm → SiLU → Conv1D` y añade la conexión residual.

## Diffusion schedule y sampling

[`diffusion.py`](src/diffusion.py) implementa un schedule lineal:

| Parámetro | Valor |
|---|---:|
| Timesteps $T$ | `100` |
| Schedule | Lineal |
| $\beta_1$ / `beta_start` | `0.0001` |
| $\beta_T$ / `beta_end` | `0.02` |
| $\alpha_t$ | $1-\beta_t$ |
| $\bar{\alpha}_t$ | $\prod_{s=1}^{t}\alpha_s$ |
| Objective | `epsilon_prediction` |

La transición inversa implementada usa:

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)\right),
$$

con varianza posterior
$\beta_t(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t)$.

## Configuración de entrenamiento

La configuración congelada está en [`config/diffusion.yaml`](config/diffusion.yaml) y se valida contra el baseline inmutable antes de entrenar.

| Parámetro | Valor |
|---|---|
| Objective | MSE de `epsilon_prediction` |
| Optimizer | AdamW |
| Learning rate | `0.0002` |
| Weight decay | `0.0001` |
| Batch size | `64` |
| Maximum epochs configurados | `200` |
| Early stopping patience | `20` epochs |
| Gradient clipping | Norma máxima `1.0` |
| Diffusion timesteps | `100` |
| Beta schedule | Lineal, `0.0001 → 0.02` |
| Train tensor dtype | `float32` después del fit `float64` |
| Validation seed | `424242`, reinicializado cada epoch |
| Device registrado | CPU en los tres training manifests |
| Entorno registrado | Python `3.14.3`, PyTorch `2.13.0+cpu` |

La validation loss se calcula sin gradientes, sin shuffle y sin pasos del optimizer. El checkpoint `best_model.pt` se actualiza únicamente cuando mejora esa validation loss; `last_model.pt` no es el modelo seleccionado para generación.

## Evidencia de entrenamiento

Los tres training seeds reproducen la misma configuración; el único cambio experimental permitido es el seed. Los checkpoints binarios permanecen fuera de Git, pero sus SHA256, manifests, historiales y provenance están versionados en [`evidence/`](evidence/).

| Training seed | Epochs completados | Best epoch | Best validation loss | Checkpoint y provenance |
|---:|---:|---:|---:|---|
| 42 | 33 | 13 | `0.5556680957476298` | [manifest](evidence/seed42/training_manifest.json) · [hashes](evidence/seed42/checkpoint_hashes.json) · [pool](evidence/seed42/final_pool_manifest.json) |
| 123 | 32 | 12 | `0.5567662715911865` | [manifest](evidence/seed123/training_manifest.json) · [hashes](evidence/seed123/checkpoint_hashes.json) · [pool](evidence/seed123/final_pool_manifest.json) |
| 2026 | 32 | 12 | `0.5573557813962301` | [manifest](evidence/seed2026/training_manifest.json) · [hashes](evidence/seed2026/checkpoint_hashes.json) · [pool](evidence/seed2026/final_pool_manifest.json) |

Estos son **training seeds del DDPM**. Los valores `42`, `123` y `2026` reaparecen en la fase downstream como seeds de subsampling/mezcla sobre el pool oficial; son ejecuciones conceptualmente distintas y no corresponden a reentrenamientos del generador.

## Convergencia

![Convergencia train/validation de los tres training seeds](evidence/figures/ddpm_training_convergence.png)

La figura procede exclusivamente de los CSV versionados. Muestra train loss, validation loss y el epoch de mínima validation loss seleccionado para cada seed. El seed 42 alcanza su mínimo en el epoch 13 y el entrenamiento finaliza en el 33 por patience; los seeds 123 y 2026 alcanzan su mínimo en el 12 y finalizan en el 32. La separación posterior entre train y validation justifica conservar el mejor checkpoint, pero no se interpreta por sí sola como una demostración general de overfitting.

## Generación sintética

El sampler [`sampler.py`](src/sampler.py) parte de:

$$x_T \sim \mathcal{N}(0,I)$$

y ejecuta los 100 pasos de reverse diffusion hasta obtener una ventana sintética $x_0$. El generador de PyTorch se inicializa explícitamente con el sampling seed y la salida se valida como finita y con shape `(N, 65, 3)`.

El output oficial común es [`outputs/diffusion_seed42_normalized.parquet`](outputs/diffusion_seed42_normalized.parquet):

| Propiedad | Valor verificado |
|---|---|
| `source_model` | `diffusion_ddpm` |
| Training/sampling seed oficial | `42` / `42` |
| Ventanas | `5000` |
| Shape lógico | `(5000, 65, 3)` |
| Space | `global_channel_normalized` |
| SHA256 | `bb9b5ad6b412fd785f73344cd765c56447b92de2b5827e40b3dc77d06e40a6c2` |

El Parquet no contiene tickers ni fechas reales. Su tensor `float32` coincide exactamente con el pool normalizado seed 42 identificado en [`official_output_provenance.json`](evidence/official_output_provenance.json), con diferencia máxima `0.0`.

## Fidelity — STRICT_FINAL

En fidelity, “real held-out donors” significa las **380 ventanas de `donor_validation`**, no NVDA. Cada método se compara con el mismo subconjunto de 380 ventanas sintéticas. No existe un composite score ni una clasificación global única.

| Métrica DDPM | Resultado STRICT_FINAL |
|---|---:|
| C2ST ROC-AUC | `0.907015` |
| Wasserstein — `log_return` | `0.308172` |
| Wasserstein — `log_high_low_range` | `0.589615` |
| Wasserstein — `log1p_volume` | `0.492661` |
| Mean correlation error | `0.080053` |
| Return ACF MAE | `0.039155` |
| Abs-return ACF MAE | `0.028037` |
| Nearest-neighbour mean / median | `10.701458` / `10.712345` |

El DDPM es el método más cercano al real en Wasserstein para `log_return`, Wasserstein para `log_high_low_range` y abs-return ACF MAE. No lidera todas las métricas: presenta el mayor error de correlación y la mayor distancia Wasserstein en volumen entre los métodos comparados. Fidelity y downstream utility responden a preguntas distintas.

Fuente: [`fidelity_master.csv`](../../reports/final_analysis/fidelity_master.csv), derivado exclusivamente del snapshot STRICT_FINAL.

## Downstream utility — STRICT_FINAL

El protocolo común mantiene todo lo demás fijo:

- `REAL_ONLY`: 62 ventanas visibles de NVDA para training;
- augmentation: 21, 62 o 186 ventanas sintéticas, equivalentes a shares de 25%, 50% y 75%;
- representación downstream: 8 features derivados de cada ventana;
- modelo: `Ridge(alpha=1)` y el mismo `StandardScaler`;
- target: future 5-session annualized realized volatility;
- test: 150 ventanas NVDA de 2023–2025;
- referencia única `REAL_ONLY`: RMSE `1.479584`, MAE `1.146934`.

| Synthetic share | Ventanas sintéticas | RMSE mean | RMSE SD | MAE mean | MAE SD | Mejora RMSE vs REAL_ONLY |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 21 | `0.309787` | `0.034746` | `0.229393` | `0.032008` | `79.062549%` |
| 50% | 62 | `0.309301` | `0.063834` | `0.242643` | `0.068618` | `79.095423%` |
| 75% | 186 | **`0.244600`** | **`0.007319`** | **`0.179606`** | **`0.011175`** | **`83.468303%`** |

DDPM @75% es la **mejor configuración observada post-hoc por RMSE** en el experimento. Es una descripción del resultado congelado, no una configuración “óptima” ni una regla de selección para datos futuros.

![Mejora RMSE por método y synthetic share](../../reports/final_analysis/figures/github/rmse_improvement_heatmap.png)

![Comparación de modelos al 75% sintético](../../reports/final_analysis/figures/github/performance_at_75pct.png)

Fuente numérica: [`master_utility_table.csv`](../../reports/final_analysis/master_utility_table.csv) y [`model_summary.csv`](../../reports/final_analysis/model_summary.csv).

## DDPM frente a Bootstrap + Jitter

La comparación contra el método simple forma parte del diseño experimental. Un valor positivo en la última columna indica que DDPM mejora más que Bootstrap + Jitter frente a `REAL_ONLY`.

| Synthetic share | RMSE Bootstrap + Jitter | RMSE DDPM | Diferencia de mejora RMSE | Lectura |
|---:|---:|---:|---:|---|
| 25% | `0.285467` | `0.309787` | `-1.643714 pp` | DDPM no supera al baseline simple |
| 50% | `0.284047` | `0.309301` | `-1.706810 pp` | DDPM no supera al baseline simple |
| 75% | `0.262744` | **`0.244600`** | **`+1.226262 pp`** | DDPM supera al baseline simple |

![Modelos neuronales frente a Bootstrap + Jitter](../../reports/final_analysis/figures/github/neural_vs_simple_baseline.png)

Fuente: [`baseline_vs_neural.csv`](../../reports/final_analysis/baseline_vs_neural.csv).

## Estabilidad de los seeds downstream

![Estabilidad por seed de subsampling](../../reports/final_analysis/figures/github/seed_stability.png)

Los seeds representados aquí (`42`, `123`, `2026`) son **downstream mixture/subsampling seeds aplicados al mismo pool sintético oficial**, no los tres training runs de la tabla de evidencia.

DDPM @50% tiene `rmse_std = 0.063834`, la dispersión más alta de todas las configuraciones del experimento. En cambio, DDPM @75% reduce la dispersión a `0.007319` y obtiene el mejor RMSE medio observado. Con sólo tres seeds, estas medidas son descriptivas, no inferenciales.

Fuente: [`seed_stability.csv`](../../reports/final_analysis/seed_stability.csv).

## Hallazgos principales

1. DDPM @75% obtiene el mejor RMSE global observado: `0.244600`.
2. Esa configuración mejora el RMSE un `83.468303%` frente a `REAL_ONLY`.
3. El DDPM no supera a Bootstrap + Jitter al 25% ni al 50%; sólo lo supera al 75%.
4. DDPM @50% presenta la mayor variabilidad downstream del experimento (`rmse_std = 0.063834`).
5. DDPM @75% es mucho más estable (`rmse_std = 0.007319`) que sus configuraciones de menor ratio.
6. El modelo destaca en algunas métricas de fidelity, pero no en correlación ni volumen.
7. Una fidelity elevada en una métrica no garantiza automáticamente mejor downstream utility.

## Limitaciones

1. El reverse diffusion utiliza 100 pasos y es computacionalmente más costoso que Bootstrap + Jitter.
2. El denoiser es una arquitectura Conv1D compacta sin attention ni estructura multiescala.
3. La estabilidad downstream se estima con sólo tres seeds de subsampling.
4. La evaluación downstream utiliza una única arquitectura, `Ridge(alpha=1)`.
5. Se estudia un único target: volatilidad realizada anualizada a cinco sesiones.
6. La validación externa se limita a un único activo objetivo, NVDA, y a su test 2023–2025.
7. El resultado al 75% es una observación descriptiva post-hoc, no tuning ni evidencia de optimalidad universal.
8. DDPM no domina al baseline simple en todos los ratios: queda por detrás al 25% y al 50%.
9. Fidelity y utility no son equivalentes; ninguna métrica aislada sustenta una conclusión universal sobre generación financiera.

## Reproducibilidad y provenance

| Elemento | Referencia |
|---|---|
| Canonical donor SHA256 | `5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f` |
| Normalización | [Evidence y estadísticas](evidence/README.md) |
| Histories y manifests | [`evidence/`](evidence/) |
| Checkpoint hashes | [`seed42`](evidence/seed42/checkpoint_hashes.json), [`seed123`](evidence/seed123/checkpoint_hashes.json), [`seed2026`](evidence/seed2026/checkpoint_hashes.json) |
| Output oficial | [`outputs/`](outputs/) — SHA256 `bb9b5ad6b412fd785f73344cd765c56447b92de2b5827e40b3dc77d06e40a6c2` |
| Dependencias | [`requirements.txt`](requirements.txt) |
| Guía común | [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md) |
| Freeze científico | Tag `strict-final-20260902` |
| Snapshot final | [`artifacts/final/strict_final_20260902/`](../../artifacts/final/strict_final_20260902/) |
| Análisis validado | [`reports/final_analysis/ANALYSIS.md`](../../reports/final_analysis/ANALYSIS.md) |

Los checkpoints no están versionados; sus hashes sí. El output oficial, los historiales, manifests, normalizadores y la curva de convergencia son visibles desde un clone limpio. No es necesario reentrenar para revisar la entrega final.

## Cómo ejecutar

Todos los comandos se ejecutan desde la raíz del repositorio.

### Instalar dependencias DDPM

Python 3.12 es la versión de referencia documentada para la entrega. PyTorch se instala sólo para trabajar con el DDPM:

```bash
python -m pip install -r requirements.txt
python -m pip install -r generadores/daniel/requirements.txt
```

### Retraining opcional

No es necesario para inspeccionar los resultados publicados. El training requiere un working tree limpio y escribe checkpoints/histories locales ignorados:

```bash
python generadores/daniel/scripts/train.py --seed 42
python generadores/daniel/scripts/train.py --seed 123
python generadores/daniel/scripts/train.py --seed 2026
```

### Generar los pools congelados

Este paso requiere los checkpoints y manifests locales producidos por training. Genera el pool normalizado y su pareja calibrada a NVDA asociados a cada seed; no sustituye al common pipeline:

```bash
python generadores/daniel/scripts/generate_final_pools.py --seed 42
python generadores/daniel/scripts/generate_final_pools.py --seed 123
python generadores/daniel/scripts/generate_final_pools.py --seed 2026
```

El Parquet oficial seed 42 ya está versionado en [`outputs/`](outputs/) y puede inspeccionarse sin ejecutar esos comandos.

### Tests

```bash
python -m pytest generadores/daniel/tests -q
```

### Inspeccionar los resultados finales

No ejecutar training ni el common pipeline para revisar la entrega. Los resultados publicados están en:

- [`artifacts/final/strict_final_20260902/`](../../artifacts/final/strict_final_20260902/): snapshot científico inmutable;
- [`reports/final_analysis/`](../../reports/final_analysis/): tablas, figuras e interpretación derivadas y validadas.

## Estructura del módulo

```text
generadores/daniel/
├── config/          # configuración DDPM congelada
├── src/             # red, diffusion, trainer, sampler y contratos locales
├── scripts/         # entrypoints de training, sampling y diagnósticos
├── tests/           # guards unitarios y de reproducibilidad
├── evidence/        # histories, manifests, hashes y curva de convergencia
├── outputs/         # Parquet oficial normalizado seed 42
├── requirements.txt # dependencias específicas de PyTorch
└── README.md        # este informe técnico
```

## Fuentes finales

Todos los resultados cuantitativos de este documento proceden del snapshot `STRICT_FINAL` y de su capa derivada validada:

- [`fidelity_master.csv`](../../reports/final_analysis/fidelity_master.csv)
- [`master_utility_table.csv`](../../reports/final_analysis/master_utility_table.csv)
- [`model_summary.csv`](../../reports/final_analysis/model_summary.csv)
- [`seed_stability.csv`](../../reports/final_analysis/seed_stability.csv)
- [`baseline_vs_neural.csv`](../../reports/final_analysis/baseline_vs_neural.csv)
- [`ANALYSIS.md`](../../reports/final_analysis/ANALYSIS.md)

No se utilizan resultados provisionales, diagnósticos individuales legacy ni métricas recalculadas para este README.
