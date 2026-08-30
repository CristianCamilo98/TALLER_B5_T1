import numpy as np

from generadores.marco.scaling import fit_scaler

cached_train = np.load("cache_donor_train_shared.npz")["values"]
scaler = fit_scaler(cached_train)

nvda_visible = np.load("cache_nvda_visible_fixed.npz")["values"]
synthetic_calibrated = np.load("nvda_sintetico_calibrado.npz")["values"]

canales = ["log_return", "log_high_low_range", "log_volume"]

print("=== 1) ¿El escalador funcionó como debía? std de donor_train YA escalado (debe ser ~1.0 en las 3) ===")
train_scaled_std = ((cached_train - scaler["mean"]) / scaler["std"]).std(axis=(0, 1))
print(dict(zip(canales, train_scaled_std)))

print("\n=== 2) Colapso de varianza POR CANAL (salida cruda del decoder / std esperado=1.0) ===")
raw_gen_std = np.load("nvda_sintetico_calibrado.npz")  # no sirve, recalculamos con el decoder directamente
# (recalculado a partir de lo que ya imprimió el script anterior, para no repetir la generación)
std_gen_scaled = np.array([0.06422286, 0.43054414, 0.565098])  # de tu print anterior
for canal, std_val in zip(canales, std_gen_scaled):
    print(f"  {canal:20s}: std generado (escalado) = {std_val:.3f}  ->  retiene {std_val*100:.1f}% de la varianza de donor_train")

print("\n=== 3) ¿El std FINAL calibrado coincide con el de NVDA real? (debe ser prácticamente idéntico) ===")
std_nvda_real = nvda_visible.std(axis=(0, 1))
std_synthetic_final = synthetic_calibrated.std(axis=(0, 1))
for canal, real_val, synth_val in zip(canales, std_nvda_real, std_synthetic_final):
    print(f"  {canal:20s}: real={real_val:.5f}  sintético_calibrado={synth_val:.5f}")