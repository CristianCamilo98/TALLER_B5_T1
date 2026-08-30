import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generadores.marco.scaling import apply_scaler

scaler_npz = np.load("scaler_donors.npz")
scaler = {"mean": scaler_npz["mean"], "std": scaler_npz["std"]}

donor_train_raw = np.load("cache_donor_train_shared.npz")["values"]
donor_train_scaled = apply_scaler(donor_train_raw, scaler)
synthetic = np.load("synthetic_scaled_raw.npz")["values"]

canales = ["log_return", "log_high_low_range", "log1p_volume"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nombre in enumerate(canales):
    axes[i].hist(donor_train_scaled[:, :, i].flatten(), bins=40, alpha=0.5, density=True, label="donor_train (real)", color="tab:blue")
    axes[i].hist(synthetic[:, :, i].flatten(), bins=40, alpha=0.5, density=True, label="sintetico (sin calibrar)", color="tab:orange")
    axes[i].set_title(nombre)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("dist_donor_train_vs_sintetico.png", dpi=120)
print("Guardado: dist_donor_train_vs_sintetico.png")
