import numpy as np
import pandas as pd

WINDOWS_DIR = "data/features/windows"


def load_window_file(path):
    df = pd.read_parquet(path)
    values = np.stack([np.asarray(row, dtype="float64").reshape(65, 3) for row in df["features_flat"]])
    return values, df


print("Cargando donor_train...")
donor_train, _ = load_window_file(f"{WINDOWS_DIR}/donor_train.parquet")
np.savez_compressed("cache_donor_train_shared.npz", values=donor_train)
print("  shape:", donor_train.shape)

print("Cargando donor_validation...")
donor_validation, _ = load_window_file(f"{WINDOWS_DIR}/donor_validation.parquet")
np.savez_compressed("cache_donor_validation_shared.npz", values=donor_validation)
print("  shape:", donor_validation.shape)

print("Cargando nvda_visible...")
nvda_visible, df_visible = load_window_file(f"{WINDOWS_DIR}/nvda_visible.parquet")
np.savez_compressed("cache_nvda_visible_shared.npz", values=nvda_visible)
print("  shape:", nvda_visible.shape)
print("  rango real:", df_visible["window_start_date"].min(), "->", df_visible["window_end_date"].max())

print("Cargando nvda_full_history (Oracle, ya viene acotado por Cristian, sin filtrado manual)...")
nvda_oracle, df_oracle = load_window_file(f"{WINDOWS_DIR}/nvda_full_history.parquet")
np.savez_compressed("cache_oracle.npz", values=nvda_oracle)
print("  shape:", nvda_oracle.shape)
print("  rango real:", df_oracle["window_start_date"].min(), "->", df_oracle["window_end_date"].max())

print("Cargando nvda_test (desde test_index.parquet, ya trae contexto+target explícitos)...")
test_index = pd.read_parquet("data/features/test_index.parquet")
nvda_test = np.stack([np.asarray(row, dtype="float64").reshape(65, 3) for row in test_index["features_flat"]])
np.savez_compressed("cache_nvda_test.npz", values=nvda_test)
print("  shape:", nvda_test.shape)
print("  rango de targets:", test_index["target_start_date"].min(), "->", test_index["target_end_date"].max())

print("\n=== Comprobación contra window_counts.csv oficial ===")
esperado = {"donor_train": 4910, "donor_validation": 380, "nvda_visible": 62, "nvda_full_history": 2703, "nvda_test": 150}
obtenido = {"donor_train": len(donor_train), "donor_validation": len(donor_validation),
            "nvda_visible": len(nvda_visible), "nvda_full_history": len(nvda_oracle), "nvda_test": len(nvda_test)}
for k in esperado:
    print(f"{k:20s} esperado={esperado[k]:5d}  obtenido={obtenido[k]:5d}  {'OK' if esperado[k] == obtenido[k] else 'MISMATCH'}")
