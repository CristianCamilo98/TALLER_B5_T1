import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

real_windows = np.load("cache_nvda_visible_shared.npz")["values"]      # [62, 65, 3]
synthetic_windows = np.load("nvda_sintetico_calibrado.npz")["values"]   # [25000, 65, 3]

canales = ["log_return", "log_high_low_range", "log1p_volume"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nombre in enumerate(canales):
    real_vals = real_windows[:, :, i].flatten()
    synth_vals = synthetic_windows[:, :, i].flatten()
    axes[i].hist(real_vals, bins=40, alpha=0.5, density=True, label="real (NVDA visible)", color="tab:blue")
    axes[i].hist(synth_vals, bins=40, alpha=0.5, density=True, label="sintético calibrado", color="tab:orange")
    axes[i].set_title(nombre)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("dist_real_vs_sintetico.png", dpi=120)
print("Guardado: dist_real_vs_sintetico.png")