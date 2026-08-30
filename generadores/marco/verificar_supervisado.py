import numpy as np
from downstream_features import build_supervised_pairs

nvda_visible = np.load("cache_nvda_visible_fixed.npz")["values"]
synthetic = np.load("nvda_sintetico_calibrado.npz")["values"]

X_real, y_real = build_supervised_pairs(nvda_visible)
X_synth, y_synth = build_supervised_pairs(synthetic)

print("X_real:", X_real.shape, "| y_real:", y_real.shape)
print("X_synth:", X_synth.shape, "| y_synth:", y_synth.shape)
print("\n¿algún NaN/Inf en real?:", not np.isfinite(X_real).all() or not np.isfinite(y_real).all())
print("¿algún NaN/Inf en sintético?:", not np.isfinite(X_synth).all() or not np.isfinite(y_synth).all())

print("\ny_real  -- media:", y_real.mean(), "| std:", y_real.std())
print("y_synth -- media:", y_synth.mean(), "| std:", y_synth.std())