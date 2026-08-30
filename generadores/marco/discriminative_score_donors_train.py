import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generadores.marco.scaling import apply_scaler

scaler_npz = np.load("scaler_donors.npz")
scaler = {"mean": scaler_npz["mean"], "std": scaler_npz["std"]}

donor_train_raw = np.load("cache_donor_train_shared.npz")["values"]
rng = np.random.default_rng(42)
idx_donor = rng.choice(len(donor_train_raw), size=380, replace=False)
donor_train_scaled = apply_scaler(donor_train_raw[idx_donor], scaler)

synthetic = np.load("synthetic_scaled_raw.npz")["values"]
n_real = len(donor_train_scaled)
idx_synth = rng.choice(len(synthetic), size=n_real, replace=False)
synth_sample = synthetic[idx_synth]

X = np.concatenate([donor_train_scaled.reshape(n_real, -1), synth_sample.reshape(n_real, -1)], axis=0)
y = np.concatenate([np.ones(n_real), np.zeros(n_real)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
pca = PCA(n_components=min(20, X_train.shape[0] - 1), random_state=42).fit(X_train)
X_train_pca, X_test_pca = pca.transform(X_train), pca.transform(X_test)

clf = LogisticRegression(max_iter=2000, random_state=42).fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
y_proba = clf.predict_proba(X_test_pca)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f"Accuracy en test: {accuracy:.4f}")
print(f"Discriminative score |accuracy - 0.5|: {abs(accuracy - 0.5):.4f}")
print("Matriz de confusion (filas=verdad, columnas=prediccion):")
print(cm)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["sintetico", "real"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["sintetico", "real"])
axes[0].set_xlabel("Prediccion"); axes[0].set_ylabel("Real")
axes[0].set_title("Matriz de confusion (donor_train)")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center")

axes[1].hist(y_proba[y_test == 0], bins=15, alpha=0.6, label="sintetico (verdad)", color="tab:orange")
axes[1].hist(y_proba[y_test == 1], bins=15, alpha=0.6, label="real (verdad)", color="tab:blue")
axes[1].axvline(0.5, color="black", linestyle="--", alpha=0.5)
axes[1].set_xlabel("P(clasificador dice real)")
axes[1].set_title("Distribucion de probabilidades")
axes[1].legend()
plt.tight_layout()
plt.savefig("clasificador_donor_train_vs_sintetico.png", dpi=120)
print("Guardado: clasificador_donor_train_vs_sintetico.png")
