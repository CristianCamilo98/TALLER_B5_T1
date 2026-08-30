import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

real_windows = np.load("cache_nvda_visible_shared.npz")["values"]
synthetic_windows = np.load("nvda_sintetico_calibrado.npz")["values"]

n_real = len(real_windows)
rng = np.random.default_rng(42)
idx_synth = rng.choice(len(synthetic_windows), size=n_real * 2, replace=False)
synthetic_sample = synthetic_windows[idx_synth]

real_flat = real_windows.reshape(n_real, -1)
synth_flat = synthetic_sample.reshape(len(synthetic_sample), -1)

combined = np.concatenate([real_flat, synth_flat], axis=0)
print(f"Total ventanas a proyectar: {combined.shape[0]} (real={n_real}, sintetico={len(synth_flat)})")

pca_result = PCA(n_components=min(50, combined.shape[0] - 1), random_state=42).fit_transform(combined)
tsne_result = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(pca_result)

real_2d = tsne_result[:n_real]
synth_2d = tsne_result[n_real:]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(real_2d[:, 0], real_2d[:, 1], c="red", alpha=0.6, s=25, label="real")
ax.scatter(synth_2d[:, 0], synth_2d[:, 1], c="blue", alpha=0.3, s=15, label="sintetico")
ax.set_title("t-SNE -- NVDA visible real vs. sintetico calibrado")
ax.legend()
plt.tight_layout()
plt.savefig("tsne_calidad.png", dpi=120)
print("Guardado: tsne_calidad.png")
