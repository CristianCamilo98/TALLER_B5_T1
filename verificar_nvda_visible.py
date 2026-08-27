import pandas as pd

windows = pd.read_parquet("data/features/windows_65_stride1.parquet").reset_index(drop=True)
windows["window_row"] = windows.index

splits = pd.read_parquet("data/splits/window_splits_stride1.parquet")
merged = windows.merge(splits[["window_row", "split"]], on="window_row", how="inner")

nvda_visible = merged[merged["split"] == "nvda_visible"].sort_values("window_end_date")
print("nº ventanas nvda_visible:", len(nvda_visible))
print("\nPrimeras 5 (las que más cerca están del corte del 1 jul 2022):")
print(nvda_visible[["window_start_date", "window_end_date"]].head())

before_cutoff = (nvda_visible["window_start_date"] < "2022-07-01").sum()
print(f"\nVentanas cuyo start_date cae ANTES del 1 jul 2022 (fuera del periodo 'visible'): {before_cutoff} de {len(nvda_visible)}")