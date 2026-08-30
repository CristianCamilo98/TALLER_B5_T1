import numpy as np
from generadores.marco.scaling import apply_scaler

scaler_npz = np.load("scaler_donors.npz")
scaler = {"mean": scaler_npz["mean"], "std": scaler_npz["std"]}

donor_train_raw = np.load("cache_donor_train_shared.npz")["values"]
donor_train_scaled = apply_scaler(donor_train_raw, scaler)
synthetic = np.load("synthetic_scaled_raw.npz")["values"]

canales = ["log_return", "log_high_low_range", "log1p_volume"]
std_train = donor_train_scaled.std(axis=(0, 1))
std_synth = synthetic.std(axis=(0, 1))

print(f"{'canal':20s} {'std_train (~1)':>15s} {'std_synth':>10s} {'varianza retenida (%)':>22s}")
for i, nombre in enumerate(canales):
    ratio_var = (std_synth[i] ** 2 / std_train[i] ** 2) * 100
    print(f"{nombre:20s} {std_train[i]:15.4f} {std_synth[i]:10.4f} {ratio_var:22.1f}")
