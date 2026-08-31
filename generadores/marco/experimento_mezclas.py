import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from generadores.marco.downstream_features import build_supervised_pairs

real_windows = np.load("cache_nvda_visible_shared.npz")["values"]
synth_windows = np.load("nvda_sintetico_calibrado.npz")["values"]
oracle_windows = np.load("cache_oracle.npz")["values"]
test_windows = np.load("cache_nvda_test.npz")["values"]

X_real, y_real = build_supervised_pairs(real_windows)
X_synth_pool, y_synth_pool = build_supervised_pairs(synth_windows)
X_oracle, y_oracle = build_supervised_pairs(oracle_windows)
X_test, y_test = build_supervised_pairs(test_windows)

print(f"real={len(X_real)}  synth_pool={len(X_synth_pool)}  oracle={len(X_oracle)}  test={len(X_test)}")

SEEDS = [42, 123, 2026]
n_real = len(X_real)
ratios = {"real_only": 0.0, "mix_25": 0.25, "mix_50": 0.50, "mix_75": 0.75}

all_results = {name: [] for name in list(ratios.keys()) + ["oracle"]}

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    def sample_synthetic(n):
        idx = rng.choice(len(X_synth_pool), size=n, replace=False)
        return X_synth_pool[idx], y_synth_pool[idx]

    datasets = {}
    for name, ratio in ratios.items():
        n_synth = int(round(n_real * ratio / (1 - ratio))) if ratio > 0 else 0
        if n_synth == 0:
            X_mix, y_mix = X_real, y_real
        else:
            X_s, y_s = sample_synthetic(n_synth)
            X_mix = np.concatenate([X_real, X_s], axis=0)
            y_mix = np.concatenate([y_real, y_s], axis=0)
        datasets[name] = (X_mix, y_mix)
    datasets["oracle"] = (X_oracle, y_oracle)

    for name, (X_train, y_train) in datasets.items():
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        all_results[name].append(rmse)

print(f"\n{'variante':12s} {'media_RMSE':>10s} {'std':>8s}   valores por seed")
for name, valores in sorted(all_results.items(), key=lambda x: np.mean(x[1])):
    print(f"{name:12s} {np.mean(valores):10.4f} {np.std(valores):8.4f}   {[round(v, 4) for v in valores]}")
