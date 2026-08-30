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
El resultado se recalibra (media/std) a los valores reales de `nvda_visible`,
para que "patrón de donors + escala de NVDA".

## Experimento: ¿ayuda el sintético con pocos datos reales?

Mismo Ridge entrenado 5 veces (solo real, +25/50/75% sintético, Oracle),
evaluado siempre contra `nvda_test` real. Media de 3 seeds:

| Mezcla | RMSE |
|---|---:|
| 100% real (62 ventanas) | 1.4796 |
| +25% sintético | 0.2748 |
| +50% sintético | 0.2519 |
| +75% sintético | 0.2396 |
| Oracle (2703 reales) | 0.2248 |

![RMSE vs sintético](figures/rmse_vs_sintetico.png)

Con solo 62 ventanas reales el modelo generaliza muy mal (RMSE 1.48). Añadir
sintético lo corrige de forma drástica y monótona, acercándose al Oracle
según sube el ratio. Confirmado también a nivel de coeficientes: la
similitud coseno de los pesos del Ridge frente al Oracle pasa de **0.26**
(solo real) a **0.64** (50% sintético) — el sintético no solo baja el error,
empuja a aprender la relación correcta.

## Métricas de calidad

- **Distribuciones marginales**: solapan razonablemente en las 3 variables.
- **t-SNE**: solapamiento parcial, no total como en el paper original —
  aparecen dos regiones sintéticas sin contrapartida real, probablemente
  porque los donors cubren 10 años de regímenes de mercado y NVDA visible
  es un único semestre.
- **Discriminative score**: 0.158 (rango 0-0.15 del paper para un generador
  que funciona razonablemente). La matriz de confusión muestra que el
  clasificador confunde sobre todo lo *real* con sintético, no al revés —
  coherente con el solapamiento parcial del t-SNE.

![Clasificador](figures/clasificador_calidad.png)


## Sintético vs. donors (antes de calibrar) — validación de la generación pura

Comparación adicional, en el espacio normalizado (z-score sobre `donor_train`,
sin calibrar a NVDA) — aísla "¿aprendió bien el VAE la distribución de
entrenamiento?" de "¿tiene sentido la calibración a NVDA?".

| Referencia | Discriminative score |
|---|---:|
| `donor_train` | 0.013 (casi indistinguible) |
| `donor_validation` | 0.268 |

La brecha no es un fallo del generador: `donor_validation` es 2022 completo,
ya identificado como año atípico (2º más volátil de la década, ligado a las
subidas de tipos de la Fed) — el sintético se parece mucho a lo que
entrenó (`donor_train`), y algo menos a un año que ya se desvía de esa
distribución por sí mismo.

### Autocorrelación temporal

![Autocorrelación](figures/heatmap_autocorrelacion.png)

`log_return` retiene solo el 0.5% de su varianza al generar (`log_high_low_range`:
17.3%; `log1p_volume`: 49.2%). Causa: el decoder produce las 65 ventanas a
partir de un único vector latente compartido, mediante convoluciones que
generan salidas suaves y correlacionadas en el tiempo. Esto reproduce (e
incluso **exagera**) canales con memoria real día a día (`log_high_low_range`,
autocorrelación real 0.26 → el sintético la sostiene más lags de los reales;
`log1p_volume`, 0.47 → mismo patrón), pero no puede replicar ruido
genuinamente independiente como `log_return` (autocorrelación real ≈ -0.04,
consistente con eficiencia de mercado) — ahí, en vez de sobre-suavizar, se
repliega hacia la media y colapsa la varianza.

## Comparabilidad entre generadores (VAE/GAN/Diffusion)

`outputs/donor_synthetic_normalized_seed42.parquet` — 5000 ventanas sintéticas,
espacio normalizado (z-score `donor_train`, **sin calibrar a NVDA**), seed=42.
Protocolo de comparación en dos niveles, para separar "¿funciona la
arquitectura?" de "¿sirve para el problema real?":

1. **Generación pura**: cada generador compara su `*_normalized_seed42.parquet`
   contra `donor_train`/`donor_validation` (marginales, t-SNE, discriminative
   score) — mismo protocolo que arriba.
2. **Tarea final**: cada generador calibrado a NVDA, mismo experimento de
   mezclas contra `nvda_test` real.

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
python -m generadores.marco.experimento_mezclas
```

Seeds: 42, 123, 2026 (experimento de mezclas). `scaler_donors.npz` y los
`.npz` de caché son regenerables, no están versionados (ver `.gitignore`).