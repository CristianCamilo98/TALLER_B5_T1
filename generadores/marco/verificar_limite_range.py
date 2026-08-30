import numpy as np

scaler_npz = np.load("scaler_donors.npz")
mean_range, std_range = scaler_npz["mean"][1], scaler_npz["std"][1]  # canal 1 = log_high_low_range

raw_zero_scaled = (0.0 - mean_range) / std_range
print(f"media(log_high_low_range) en donor_train: {mean_range:.4f}")
print(f"std(log_high_low_range) en donor_train:   {std_range:.4f}")
print(f"log(High/Low)=0 (caso High==Low), tras escalar, corresponde a z = {raw_zero_scaled:.4f}")
print("Ningún punto real debería aparecer más a la izquierda de ese valor -- es un límite matemático, no ruido.")