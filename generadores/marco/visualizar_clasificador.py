import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

real_windows = np.load("cache_nvda_visible_shared.npz")["values"]
synthetic_windows = np.load("nvda_sintetico_calibrado.npz")["values"]

n_real = len(real_windows)
rng = np.random.default_rng(42)
idx_synth = rng.choice(len(synthetic_windows), size=n_real, replace=False)
synthetic_sample = synthetic_windows[idx_synth]

X = np.concatenate([real_windows.reshape(n_real, -1), synthetic_sample.reshape(n_real, -1)], axis=0)
y = np.concatenate([np.ones(n_real), np.zeros(n_real)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
pca = PCA(n_components=min(20, X_train.shape[0] - 1), random_state=42).fit(X_train)
X_train_pca, X_test_pca = pca.transform(X_train), pca.transform(X_test)

clf = LogisticRegression(max_iter=2000, random_state=42).fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
y_proba = clf.predict_proba(X_test_pca)[:, 1]

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["sintetico", "real"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["sintetico", "real"])
axes[0].set_xlabel("Prediccion"); axes[0].set_ylabel("Real")
axes[0].set_title("Matriz de confusion")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center")

axes[1].hist(y_proba[y_test == 0], bins=15, alpha=0.6, label="sintetico (verdad)", color="tab:orange")
axes[1].hist(y_proba[y_test == 1], bins=15, alpha=0.6, label="real (verdad)", color="tab:blue")
axes[1].axvline(0.5, color="black", linestyle="--", alpha=0.5)
axes[1].set_xlabel("P(clasificador dice 'real')")
axes[1].set_title("Distribucion de probabilidades predichas")
axes[1].legend()

plt.tight_layout()
plt.savefig("clasificador_calidad.png", dpi=120)
print("Guardado: clasificador_calidad.png")
print("Matriz de confusion (filas=verdad, columnas=prediccion):")
print(cm)
