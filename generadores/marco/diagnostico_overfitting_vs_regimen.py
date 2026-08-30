import numpy as np
import pandas as pd
import tensorflow as tf

from generadores.marco.architecture import TimeVAE
from generadores.marco.scaling import fit_scaler, apply_scaler

LATENT_DIM = 8

df_features = pd.read_parquet("data/features/windows_65_stride1.parquet").reset_index(drop=True)
df_features["window_row"] = df_features.index

df_splits = pd.read_parquet("data/splits/window_splits_stride1.parquet")
df_splits_slim = df_splits[["window_row", "split"]]  # solo lo que necesitamos, sin columnas duplicadas

df = pd.merge(df_features, df_splits_slim, on="window_row", how="inner")

df_train = df[df["split"] == "donor_train"].copy()
df_train["window_end_date"] = pd.to_datetime(df_train["window_end_date"])
df_train["year"] = df_train["window_end_date"].dt.year

X_train = np.stack([np.asarray(row, dtype="float32").reshape(65, 3) for row in df_train["features_flat"]])

cached_train = np.load("cache_donor_train_shared.npz")["values"]
scaler = fit_scaler(cached_train)
X_train_scaled = apply_scaler(X_train, scaler).astype(np.float32)

model = TimeVAE(latent_dim=LATENT_DIM, dropout_rate=0.2, l2_reg=1e-4)
_ = model(tf.zeros((1, 65, 3)))
model.load_weights("vae_donors_weights.weights.h5")

x_hat, mu, logvar = model(X_train_scaled, training=False)
x_hat = x_hat.numpy()

delta = 1.0
diff = np.abs(X_train_scaled - x_hat)
huber_elem = np.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
df_train["recon_error"] = huber_elem.mean(axis=(1, 2))

print(df_train.groupby("year")["recon_error"].mean())
print("\n¿2020 destaca DENTRO de train, igual que 2022 destaca en validation?")