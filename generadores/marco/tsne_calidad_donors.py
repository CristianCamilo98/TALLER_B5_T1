import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from generadores.marco.scaling import apply_scaler

scaler_npz = np.load("scaler_donors.npz")
scaler = {"mean": scaler_npz["mean"], "std": scaler_npz["std"]}

donor_val_raw = np.load("cache_donor_validation_shared.npz")["values"]
donor_val_scaled = apply_scaler(donor_val_raw, scaler)
synthetic = np.load("synthetic_scaled_raw.npz")["values"]

n_real = len(donor_val_scaled)
rng = np.random.default_rng(42)
idx = rng.choice(len(synthetic), size=n_real, replace=False)
synth_sample = synthetic[idx]

combined = np.concatenate([donor_val_scaled.reshape(n_real, -1), synth_sample.reshape(n_real, -1)], axis=0)
pca_result = PCA(n_components=min(50, combined.shape[0] - 1), random_state=42).fit_transform(combined)
tsne_result = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(pca_result)

real_2d, synth_2d = tsne_result[:n_real], tsne_result[n_real:]
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(real_2d[:, 0], real_2d[:, 1], c="red", alpha=0.5, s=15, label="donor_validation (real)")
ax.scatter(synth_2d[:, 0], synth_2d[:, 1], c="blue", alpha=0.3, s=15, label="sintetico (sin calibrar)")
ax.set_title("t-SNE -- donor_validation real vs. sintetico (sin calibrar)")
ax.legend()
plt.tight_layout()
plt.savefig("tsne_donor_vs_sintetico.png", dpi=120)
print("Guardado: tsne_donor_vs_sintetico.png")
