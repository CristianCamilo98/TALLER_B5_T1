import pandas as pd

paths = {
    "cristian_raw": "generadores/cristian/outputs/synthetic_seed42_n5000.parquet",
    "cristian_norm": "generadores/cristian/outputs/synthetic_seed42_n5000_normalized.parquet",
    "daniel_norm": "generadores/daniel/outputs/diffusion_seed42_normalized.parquet",
    "marco_norm": "generadores/marco/outputs/donor_synthetic_normalized_seed42.parquet",
}

for name, path in paths.items():
    df = pd.read_parquet(path)
    print(f"=== {name} ===")
    print("columnas:", df.columns.tolist())
    print("filas:", len(df))
    print(df.head(1))
    print()
