import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from downstream_features import build_supervised_pairs

real_windows = np.load("cache_nvda_visible_shared.npz")["values"]
synth_windows = np.load("nvda_sintetico_calibrado.npz")["values"]
oracle_windows = np.load("cache_oracle.npz")["values"]
test_windows = np.load("cache_nvda_test.npz")["values"]

X_real, y_real = build_supervised_pairs(real_windows)
X_synth_pool, y_synth_pool = build_supervised_pairs(synth_windows)
X_oracle, y_oracle = build_supervised_pairs(oracle_windows)
X_test, y_test = build_supervised_pairs(test_windows)

SEEDS = [42, 123, 2026]
n_real = len(X_real)
ratios = [0.0, 0.25, 0.50, 0.75]

def evaluate(X_train, y_train, seed):
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(X_train_s, y_train)
    return np.sqrt(mean_squared_error(y_test, model.predict(X_test_s)))

means, stds = [], []
for ratio in ratios:
    rmses = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n_synth = int(round(n_real * ratio / (1 - ratio))) if ratio > 0 else 0
        if n_synth == 0:
            X_train, y_train = X_real, y_real
        else:
            idx = rng.choice(len(X_synth_pool), size=n_synth, replace=False)
            X_train = np.concatenate([X_real, X_synth_pool[idx]], axis=0)
            y_train = np.concatenate([y_real, y_synth_pool[idx]], axis=0)
        rmses.append(evaluate(X_train, y_train, seed))
    means.append(np.mean(rmses))
    stds.append(np.std(rmses))

oracle_mean = np.mean([evaluate(X_oracle, y_oracle, s) for s in SEEDS])
pct = [r * 100 for r in ratios]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(pct, means, yerr=stds, marker="o", capsize=4, label="real + sintetico (media +/- std, 3 seeds)")
ax.axhline(oracle_mean, color="green", linestyle="--", label=f"Oracle (RMSE={oracle_mean:.3f})")
ax.set_yscale("log")  # el salto real_only (~1.48) vs el resto (~0.22-0.27) aplastaria la escala lineal
ax.set_xlabel("% de la mezcla que es sintetico")
ax.set_ylabel("RMSE sobre nvda_test real (escala log)")
ax.set_title("RMSE vs. proporcion de datos sinteticos")
ax.legend()
plt.tight_layout()
plt.savefig("rmse_vs_sintetico.png", dpi=120)
print("Guardado: rmse_vs_sintetico.png")
for p, m, s in zip(pct, means, stds):
    print(f"{p:5.1f}% sintetico -> RMSE = {m:.4f} +/- {s:.4f}")
print(f"Oracle -> RMSE = {oracle_mean:.4f}")
