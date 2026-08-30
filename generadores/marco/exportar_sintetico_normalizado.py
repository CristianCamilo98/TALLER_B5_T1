import json
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("generadores/marco/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

synthetic = np.load("synthetic_scaled_raw.npz")["values"]
scaler_npz = np.load("scaler_donors.npz")
n_samples = synthetic.shape[0]

df = pd.DataFrame({
    "sample_id": np.arange(n_samples),
    "features_flat": [row.reshape(-1).tolist() for row in synthetic],
})
output_path = OUTPUT_DIR / "donor_synthetic_normalized_seed42.parquet"
df.to_parquet(output_path)

manifest = {
    "generator": "marco_vae",
    "seed": 42,
    "n_samples": int(n_samples),
    "window_length": 65,
    "channels": ["log_return", "log_high_low_range", "log1p_volume"],
    "space": "z-score sobre donor_train (ddof=0), SIN calibrar a NVDA",
    "scaler_mean": scaler_npz["mean"].tolist(),
    "scaler_std": scaler_npz["std"].tolist(),
    "purpose": "comparacion entre generadores (VAE/GAN/Diffusion) contra donor_train/donor_validation, antes de calibracion a NVDA",
}
with open(OUTPUT_DIR / "donor_synthetic_normalized_seed42_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Guardado: {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")
