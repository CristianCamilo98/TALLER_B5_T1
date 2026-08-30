import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

real_windows = np.load("cache_nvda_visible_shared.npz")["values"]
synthetic_windows = np.load("nvda_sintetico_calibrado.npz")["values"]

n_real = len(real_windows)
rng = np.random.default_rng(42)
idx_synth = rng.choice(len(synthetic_windows), size=n_real, replace=False)
synthetic_sample = synthetic_windows[idx_synth]

X_real = real_windows.reshape(n_real, -1)
X_synth = synthetic_sample.reshape(n_real, -1)

X = np.concatenate([X_real, X_synth], axis=0)
y = np.concatenate([np.ones(n_real), np.zeros(n_real)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pca = PCA(n_components=min(20, X_train.shape[0] - 1), random_state=42).fit(X_train)
X_train_pca, X_test_pca = pca.transform(X_train), pca.transform(X_test)

clf = LogisticRegression(max_iter=2000, random_state=42)
clf.fit(X_train_pca, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test_pca))

print(f"Accuracy en test: {accuracy:.4f}")
print(f"Discriminative score |accuracy - 0.5|: {abs(accuracy - 0.5):.4f}  (0 = ideal, indistinguible)")
