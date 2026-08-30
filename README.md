# Generador B — VAE (Marco)

TimeVAE (Desai et al., 2022) adaptado al problema de cold-start de NVDA: aprende
de 10 semiconductores donors (2012-2021) y genera ventanas sintéticas calibradas
a la escala de NVDA, para complementar sus 6 meses reales visibles.

## Arquitectura

![Arquitectura](figures/arquitectura_vae.png)

Encoder: 2×(Conv1D → BatchNorm → ReLU → Dropout) → Dense(mu), Dense(logvar).
Decoder: Dense → Reshape → 2×Conv1DTranspose → Cropping1D(0,3) (ajusta 68→65).
`latent_dim=8`. Sin condicionante — a diferencia de un CVAE, aquí no hay
variable exógena externa que mantener aparte de lo generado.

## Datos

| Split | Ventanas | Uso |
|---|---:|---|
| `donor_train` | 4910 | entrenamiento VAE |
| `donor_validation` | 380 | early stopping |
| `nvda_visible` | 62 | NVDA "cold-start" (jul-dic 2022) |
| `nvda_full_history` (Oracle) | 2703 | referencia con historia completa |
| `nvda_test` | 150 | hold-out real, 2023-2025, targets sin solape |

NVDA nunca entra en `donor_train`/`donor_validation`.

## Entrenamiento

Huber loss + KL con free-bits (0.25 nats/dim), warmup de KL en 15 épocas,
Adam lr=1e-3, early stopping (paciencia 10). Mejor época: 10, `val_recon=0.312`.

![Loss](figures/loss_vae_donors.png)

## Generación y calibración

Se genera muestreando `z~N(0,1)` directamente (sin encoder) y decodificando.
El resultado se recalibra (media/std) a los valores reales de NVDA visible,
para que "patrón de donors + escala de NVDA". La calibración se calcula sobre
los **126 días únicos** reconstruidos de `nvda_visible` (no sobre las 62
ventanas aplanadas) — ver nota en Limitaciones.

Sintético exportado en `outputs/nvda_synthetic_windows.parquet` (25000 ventanas).

## Experimento: ¿ayuda el sintético con pocos datos reales?

Mismo Ridge entrenado 5 veces (solo real, +25/50/75% sintético, Oracle),
evaluado siempre contra `nvda_test` real. Media de 3 seeds:

| Mezcla | RMSE |
|---|---:|
| 100% real (62 ventanas) | 1.4796 |
| +25% sintético | 0.2749 |
| +50% sintético | 0.2520 |
| +75% sintético | 0.2395 |
| Oracle (2703 reales) | 0.2248 |

![RMSE vs sintético](figures/rmse_vs_sintetico.png)

Con solo 62 ventanas reales el modelo generaliza muy mal (RMSE 1.48). Añadir
sintético lo corrige de forma drástica y monótona, acercándose al Oracle
según sube el ratio. Confirmado también a nivel de coeficientes: la
similitud coseno de los pesos del Ridge frente al Oracle pasa de **0.26**
(solo real) a **0.65** (50% sintético) — el sintético no solo baja el error,
empuja a aprender la relación correcta.

## Métricas de calidad

- **Distribuciones marginales**: solapan razonablemente en las 3 variables.

![Distribuciones](figures/dist_real_vs_sintetico.png)

- **t-SNE**: solapamiento parcial, no total como en el paper original —
  aparecen dos regiones sintéticas sin contrapartida real, probablemente
  porque los donors cubren 10 años de regímenes de mercado y NVDA visible
  es un único semestre.

![t-SNE](figures/tsne_calidad.png)

- **Discriminative score**: 0.105 (rango 0-0.15 del paper para un generador
  que funciona razonablemente). La matriz de confusión muestra que el
  clasificador confunde sobre todo lo *real* con sintético, no al revés —
  coherente con el solapamiento parcial del t-SNE.

![Clasificador](figures/clasificador_calidad.png)

## Limitaciones

- **Colapso de varianza no uniforme por canal**: `log_return` retiene solo
  ~6% de la varianza de entrenamiento al generar (vs 43-57% en las otras
  2 variables) — justo la variable más importante para el downstream.
  Limitación conocida de los VAE al generar desde el prior, no del
  entrenamiento en sí (con `z` real, la reconstrucción es buena).
- La brecha train/validation en el entrenamiento no es sobreajuste: 2022
  es el 2º año más volátil de la década en los donors (ligado a las subidas
  de tipos de la Fed, -36% del índice SOX), y 2020 (dentro de train)
  también reconstruye peor — mismo patrón, año atípico, no memorización.
- Un bug de `free_bits` en la implementación Keras (el suelo se aplicaba al
  KL total en vez de por dimensión) fue detectado y corregido; impacto
  menor en el resultado final.
- Un bug en la base de datos común (`nvda_visible` incluía ventanas cuyo
  contexto se apoyaba en historia oculta) fue detectado, reportado, y
  corregido por el equipo antes del experimento final.
- **Corrección de calibración (detectada en revisión de equipo)**: la
  calibración inicial promediaba sobre las 62 ventanas aplanadas (4030
  posiciones), pesando los días centrales del semestre hasta 62x más que
  los de los bordes (por el solapamiento de ventanas con stride=1). Se
  corrigió calibrando sobre los 126 días únicos reales. El sesgo era
  moderado (~9% en `log1p_volume`, marginal en las otras 2 variables) y no
  afectó al resultado central del experimento de mezclas, pero sí mejoró
  el discriminative score (0.158 → 0.105).

## Trabajo futuro

- Embedding de ticker o condicionante de régimen, para que el generador no
  tenga que "adivinar" en qué tipo de año está la ventana.
- Repetir el experimento de mezclas con más seeds para intervalos de
  confianza más ajustados.

## Reproducibilidad

```powershell
python -m generadores.marco.cargar_datos_compartidos
python -m generadores.marco.entrenar_vae
python -m generadores.marco.generar_sintetico
python -m generadores.marco.exportar_sintetico
python -m generadores.marco.experimento_mezclas
```

Seeds: 42, 123, 2026 (experimento de mezclas). `scaler_donors.npz` y los
`.npz` de caché son regenerables, no están versionados (ver `.gitignore`).
El sintético final sí se versiona, en `outputs/nvda_synthetic_windows.parquet`.