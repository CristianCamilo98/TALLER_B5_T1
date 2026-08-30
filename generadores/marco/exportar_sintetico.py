import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("generadores/marco/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mismo formato que usa el resto del equipo en data/features/windows/*.parquet
# (una fila por ventana, "features_flat" con los 195 valores en orden row-major
# t=0..64 x canal) -- así cualquiera puede leerlo con el mismo patrón que ya
# usáis para el resto de datos del proyecto.
synthetic = np.load("nvda_sintetico_calibrado.npz")["values"]  # [25000, 65, 3]
n_samples = synthetic.shape[0]

df = pd.DataFrame({
    "sample_id": np.arange(n_samples),
    "features_flat": [row.reshape(-1).tolist() for row in synthetic],
})

output_path = OUTPUT_DIR / "nvda_synthetic_windows.parquet"
df.to_parquet(output_path)

print(f"Guardado: {output_path}")
print(f"{n_samples} ventanas x 195 floats (65 dias x 3 canales, row-major)")
print(f"Tamano del fichero: {output_path.stat().st_size / 1e6:.1f} MB")