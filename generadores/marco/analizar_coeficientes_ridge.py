import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from generadores.marco.downstream_features import build_supervised_pairs

FEATURE_NAMES = ["rv_5", "rv_20", "rv_60", "mean_abs_return_20", "momentum_20", "mean_range_20", "mean_volume_20", "std_volume_20"]

real_windows = np.load("cache_nvda_visible_shared.npz")["values"]
synth_windows = np.load("nvda_sintetico_calibrado.npz")["values"]
oracle_windows = np.load("cache_oracle.npz")["values"]

X_real, y_real = build_supervised_pairs(real_windows)
X_synth_pool, y_synth_pool = build_supervised_pairs(synth_windows)
X_oracle, y_oracle = build_supervised_pairs(oracle_windows)

rng = np.random.default_rng(42)
idx = rng.choice(len(X_synth_pool), size=len(X_real), replace=False)
X_mix50 = np.concatenate([X_real, X_synth_pool[idx]], axis=0)
y_mix50 = np.concatenate([y_real, y_synth_pool[idx]], axis=0)

def fit_coefs(X, y):
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=1.0, random_state=42).fit(scaler.transform(X), y)
    return model.coef_

coef_real = fit_coefs(X_real, y_real)
coef_mix50 = fit_coefs(X_mix50, y_mix50)
coef_oracle = fit_coefs(X_oracle, y_oracle)

df = pd.DataFrame({"feature": FEATURE_NAMES, "real_only": coef_real, "mix_50": coef_mix50, "oracle": coef_oracle})
print(df.to_string(index=False))

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\nSimilitud coseno real_only vs oracle: {cosine_sim(coef_real, coef_oracle):.4f}")
print(f"Similitud coseno mix_50   vs oracle: {cosine_sim(coef_mix50, coef_oracle):.4f}")
print("(mas cerca de 1.0 = la relacion aprendida apunta en la misma direccion que la real)")
