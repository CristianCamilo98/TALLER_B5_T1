import json
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("generadores/marco/outputs")

scaler = np.load("scaler_donors.npz")

manifest = {
    "generator": "marco_vae",
    "seed": 42,
    "n_samples": 5000,
    "window_length": 65,
    "channels": ["log_return", "log_high_low_range", "log1p_volume"],
    "space": "global_channel_normalized",
    "mean": scaler["mean"].tolist(),
    "std": scaler["std"].tolist(),
    "fit_split": "donor_train",
    "fit_axes": [0, 1],
    "ddof": 0,
    "fit_dtype": "float64",
    "note": "Regenerado tras corregir scaling.py (float64 estricto en fit, antes convertia a float32 antes de calcular mean/std)",
}

with open(OUTPUT_DIR / "donor_synthetic_normalized_seed42_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Manifest regenerado con stats del scaler corregido")
print(f"mean: {manifest['mean']}")
print(f"std: {manifest['std']}")
