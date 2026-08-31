import numpy as np
import pandas as pd

# 1) Estadisticas REALMENTE usadas para entrenar (el sidecar que ya tienes guardado)
scaler_used = np.load("scaler_donors.npz")
mean_used, std_used = scaler_used["mean"].astype(np.float64), scaler_used["std"].astype(np.float64)

# 2) Estadisticas canonicas que da el equipo (contrato comun)
CHANNEL_ORDER = ["log_return", "log_high_low_range", "log1p_volume"]
mean_canonical = np.array([0.0008114289710088066, 0.02602580514891484, 16.06027218135258], dtype=np.float64)
std_canonical = np.array([0.023515504591060377, 0.01672428879172832, 1.0933253360280637], dtype=np.float64)

# 3) Recalculo INDEPENDIENTE, en float64 estricto, DESDE EL ORIGEN oficial
#    (no desde ninguna cache propia) -- para aislar si el problema esta en
#    el pipeline de Marco o es algo mas profundo en los datos en si.
df = pd.read_parquet("data/features/windows/donor_train.parquet")
windows_f64 = np.stack([
    np.asarray(row, dtype=np.float64).reshape(65, 3) for row in df["features_flat"]
])
assert windows_f64.shape[0] == 4910, f"esperadas 4910 ventanas, encontradas {windows_f64.shape[0]}"
assert windows_f64.dtype == np.float64

mean_fresh64 = windows_f64.mean(axis=(0, 1), dtype=np.float64)
std_fresh64 = windows_f64.std(axis=(0, 1), ddof=0, dtype=np.float64)

print("=== AUDITORIA DEL SCALER -- SOLO LECTURA, SIN CAMBIOS ===\n")
print(f"{'canal':22s} {'usado_por_VAE':>18s} {'recalc_float64':>18s} {'canonico_equipo':>18s}")
for i, ch in enumerate(CHANNEL_ORDER):
    print(f"{ch:22s} {mean_used[i]:18.10f} {mean_fresh64[i]:18.10f} {mean_canonical[i]:18.10f}   (mean)")
for i, ch in enumerate(CHANNEL_ORDER):
    print(f"{ch:22s} {std_used[i]:18.10f} {std_fresh64[i]:18.10f} {std_canonical[i]:18.10f}   (std)")

print("\n=== DELTAS (mean) ===")
for i, ch in enumerate(CHANNEL_ORDER):
    d_used_canon = mean_used[i] - mean_canonical[i]
    d_fresh_canon = mean_fresh64[i] - mean_canonical[i]
    d_used_fresh = mean_used[i] - mean_fresh64[i]
    print(f"\n{ch}:")
    print(f"  usado_VAE  vs canonico : delta={d_used_canon: .3e}  rel={d_used_canon/mean_canonical[i]: .3e}")
    print(f"  recalc_f64 vs canonico : delta={d_fresh_canon: .3e}  rel={d_fresh_canon/mean_canonical[i]: .3e}")
    print(f"  usado_VAE  vs recalc_f64: delta={d_used_fresh: .3e}  rel={d_used_fresh/mean_fresh64[i]: .3e}")

print("\n=== DIAGNOSTICO ===")
fresh_matches_canonical = np.allclose(mean_fresh64, mean_canonical, rtol=1e-6) and np.allclose(std_fresh64, std_canonical, rtol=1e-6)
used_matches_canonical = np.allclose(mean_used, mean_canonical, rtol=1e-6) and np.allclose(std_used, std_canonical, rtol=1e-6)
print(f"Recalculo float64 estricto == canonico del equipo (rtol=1e-6): {fresh_matches_canonical}")
print(f"Scaler REALMENTE usado por el VAE == canonico (rtol=1e-6):     {used_matches_canonical}")

if fresh_matches_canonical and not used_matches_canonical:
    print("\n-> El origen de datos y la formula son correctos (float64 estricto SI coincide")
    print("   con el canonico). La diferencia esta en COMO se calculo el scaler realmente")
    print("   usado -- consistente con conversion a float32 ANTES de calcular media/std,")
    print("   no despues. El VAE SI entreno con estadisticas ligeramente distintas a las")
    print("   canonicas, no es solo un problema de metadata/export.")
elif not fresh_matches_canonical:
    print("\n-> AVISO: ni siquiera el recalculo float64 estricto coincide con el canonico.")
    print("   La causa NO es (solo) precision de float32 -- hay algo mas que investigar")
    print("   antes de concluir nada (posible diferencia de datos de origen, formula, o ventanas).")
