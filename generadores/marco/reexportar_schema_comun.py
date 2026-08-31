import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("generadores/marco/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

synthetic = np.load("synthetic_scaled_raw.npz")["values"]  # [5000, 65, 3], espacio z-score global, ya en float32
n_samples = synthetic.shape[0]

records = []
for i in range(n_samples):
    records.append({
        "synthetic_id": i,
        "source_model": "marco_vae",
        "training_seed": 42,
        "space": "global_channel_normalized",
        "window_length": 65,
        "n_channels": 3,
        "channel_order": ["log_return", "log_high_low_range", "log1p_volume"],
        "features_flat": synthetic[i].astype(np.float32).reshape(-1).tolist(),
    })

df = pd.DataFrame.from_records(records)

# Orden exacto de columnas que pide el contrato, sin nada extra
df = df[["synthetic_id", "source_model", "training_seed", "space",
         "window_length", "n_channels", "channel_order", "features_flat"]]

output_path = OUTPUT_DIR / "donor_synthetic_normalized_seed42.parquet"
df.to_parquet(output_path, index=False)

print(f"Guardado: {output_path}")
print(f"Filas: {len(df)} (esperado: 5000)")
print(f"Columnas: {df.columns.tolist()}")
print(f"synthetic_id unicos: {df['synthetic_id'].nunique()} (esperado: 5000, sin duplicados)")
print(f"Tamano: {output_path.stat().st_size / 1e6:.2f} MB")
