import numpy as np
import tensorflow as tf
import os

from generadores.marco.architecture import TimeVAE
from generadores.marco.losses import reconstruction_loss, kl_divergence_free_bits
from generadores.marco.scaling import fit_scaler, apply_scaler

# Configurar GPU si está disponible
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Device: GPU")
else:
    print("Device: CPU")

# 1. Cargar y escalar datos
train_data = np.load("cache_donor_train_shared.npz")["values"]
val_data = np.load("cache_donor_validation_shared.npz")["values"]

scaler = fit_scaler(train_data)
print("media por canal:", scaler["mean"])
print("std por canal:", scaler["std"])
np.savez("scaler_donors.npz", mean=scaler["mean"], std=scaler["std"])

train_data_scaled = apply_scaler(train_data, scaler).astype(np.float32)
val_data_scaled = apply_scaler(val_data, scaler).astype(np.float32)

# Crear Datasets de tf.data (Keras usa N, 65, 3 nativamente, no hay que transponer)
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices(train_data_scaled).shuffle(10000).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices(val_data_scaled).batch(128)

# 2. Inicializar Modelo y Optimizador
LATENT_DIM = 8
model = TimeVAE(latent_dim=LATENT_DIM, dropout_rate=0.2, l2_reg=1e-4)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 3. Hiperparámetros de entrenamiento
MAX_EPOCHS, KL_WARMUP_EPOCHS, KL_WEIGHT_MAX = 60, 15, 0.01
PATIENCE, FREE_BITS = 10, 0.25

tf.random.set_seed(42)
np.random.seed(42)

train_loss_history, val_loss_history = [], []
best_val, best_epoch, no_improve = float("inf"), 0, 0

# Para guardar los mejores pesos
checkpoint_path = "vae_donors_weights.weights.h5"

for epoch in range(MAX_EPOCHS):
    # KL Warmup
    kl_weight = KL_WEIGHT_MAX * min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS)
    
    # -- Entrenamiento --
    running_recon = 0.0
    for batch in train_dataset:
        with tf.GradientTape() as tape:
            # training=True activa el Dropout
            x_hat, mu, logvar = model(batch, training=True)
            recon = reconstruction_loss(batch, x_hat)
            kl = kl_divergence_free_bits(mu, logvar, free_bits=FREE_BITS)
            
            # Pérdida total = Recon + KL + L2 Regularization (interna del modelo)
            loss = recon + kl_weight * kl + sum(model.losses)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        running_recon += recon.numpy() * batch.shape[0]
        
    train_recon_epoch = running_recon / len(train_data_scaled)
    train_loss_history.append(train_recon_epoch)

    # -- Validación --
    running_val_recon = 0.0
    for batch in val_dataset:
        # training=False desactiva el Dropout
        x_hat, mu, logvar = model(batch, training=False)
        recon = reconstruction_loss(batch, x_hat)
        running_val_recon += recon.numpy() * batch.shape[0]
        
    val_recon_epoch = running_val_recon / len(val_data_scaled)
    val_loss_history.append(val_recon_epoch)

    print(f"epoch {epoch:02d} | train={train_recon_epoch:.4f} | val={val_recon_epoch:.4f}")

    # -- Early Stopping --
    if val_recon_epoch < best_val:
        best_val, best_epoch, no_improve = val_recon_epoch, epoch, 0
        model.save_weights(checkpoint_path)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping en epoch {epoch}, mejor fue epoch {best_epoch}")
            break

# Cargar los mejores pesos al final
model.load_weights(checkpoint_path)
np.savez("loss_history.npz", train=np.array(train_loss_history), val=np.array(val_loss_history), best_epoch=best_epoch)

print(f"\nMejor val_recon: {best_val:.4f} en epoch {best_epoch}")
print(f"Guardado: {checkpoint_path}, scaler_donors.npz, loss_history.npz")