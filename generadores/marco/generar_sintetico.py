import numpy as np
import tensorflow as tf
from generadores.marco.architecture import TimeVAE

LATENT_DIM = 8
N_SAMPLES = 5000   # volumen alto a propósito: el equipo decidirá luego cuántas
                     # usar en cada ratio de mezcla (25/50/75%); mejor generar de más
                     # ahora que quedarse cortos y tener que repetir el paso

print("Cargando arquitectura y pesos del VAE...")
model = TimeVAE(latent_dim=LATENT_DIM, dropout_rate=0.2, l2_reg=1e-4)
_ = model(tf.zeros((1, 65, 3)))  # pasada "dummy": Keras necesita conocer las formas
                                  # de las capas antes de poder cargarles pesos guardados
model.load_weights("vae_donors_weights.weights.h5")

# --- Generación pura: muestreamos z directamente del prior N(0,1) ---
# Esto es lo que convierte al VAE en generador de verdad: NO usamos el encoder
# (que necesitaría una ventana real de entrada) -- solo el decoder, alimentado
# con ruido gaussiano. Es la misma mecánica que "sample_raw" del proyecto anterior.
print(f"\nGenerando {N_SAMPLES} ventanas sintéticas desde el espacio latente...")
tf.random.set_seed(42)
z_sample = tf.random.normal(shape=(N_SAMPLES, LATENT_DIM))

# training=False: desactiva Dropout y usa las medias/varianzas POBLACIONALES
# de BatchNorm (no las de un mini-batch) -- imprescindible en inferencia real.
x_generated_scaled = model.decoder(z_sample, training=False).numpy()

# Guardamos TAMBIÉN la versión sin calibrar (espacio escalado de donor_train).
# Sirve para comparar el sintético directamente contra los donors, separando
# "¿el VAE aprendió bien su propia distribución?" de "¿tiene sentido la
# calibración a NVDA?" -- son dos preguntas distintas.
np.savez_compressed("synthetic_scaled_raw.npz", values=x_generated_scaled.astype(np.float32))
print("Guardado (sin calibrar, espacio escalado): synthetic_scaled_raw.npz")

# --- Calibración a NVDA ---
# El generador aprendió el PATRÓN de los semiconductores donors, pero su escala
# (nivel medio, dispersión) es la de "un semiconductor genérico", no la de NVDA
# en concreto. Reescalamos usando los 6 meses reales de NVDA (los válidos, tras
# tu propio fix de la fecha de inicio) para que "patrón de donors + escala de NVDA"
# -- exactamente la receta que el equipo acordó desde el diseño original.
print("\nCalibrando muestras a la distribución de NVDA (jul-dic 2022, ventanas válidas)...")
nvda_visible = np.load("cache_nvda_visible_shared.npz")["values"]

mean_nvda = np.mean(nvda_visible, axis=(0, 1))
std_nvda = np.std(nvda_visible, axis=(0, 1))
std_nvda[std_nvda == 0] = 1e-8

mean_gen = np.mean(x_generated_scaled, axis=(0, 1))
std_gen = np.std(x_generated_scaled, axis=(0, 1))
std_gen[std_gen == 0] = 1e-8

print(f"  -> Media NVDA real:   {mean_nvda}")
print(f"  -> Std NVDA real:     {std_nvda}")
print(f"  -> Media generado:    {mean_gen}")
print(f"  -> Std generado:      {std_gen}")

x_synthetic_nvda = ((x_generated_scaled - mean_gen) / std_gen) * std_nvda + mean_nvda

output_file = "nvda_sintetico_calibrado.npz"
np.savez_compressed(output_file, values=x_synthetic_nvda.astype(np.float32))
print(f"\n¡Generación completada! Guardado en: {output_file}")
print(f"Shape final: {x_synthetic_nvda.shape}")