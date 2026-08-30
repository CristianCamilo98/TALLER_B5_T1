import numpy as np
import pandas as pd
import importlib

io_synthetic = importlib.import_module("common_pipeline.03_utility.io_synthetic")
discover_synthetic_pools = io_synthetic.discover_synthetic_pools
CHANNEL_ORDER = io_synthetic.CHANNEL_ORDER

calib = pd.read_csv("common_pipeline/03_utility/results/tables/nvda_calibration.csv").set_index("channel").loc[list(CHANNEL_ORDER)]
mu, sigma = calib["mean"].to_numpy(), calib["std"].to_numpy()
threshold_z = -mu / sigma

pools = discover_synthetic_pools()

print(f"\nUmbral de invalidez en Z, por canal: {dict(zip(CHANNEL_ORDER, threshold_z.round(2)))}\n")
for method, pool in pools.items():
    print(f"--- {method} ---")
    for i, channel in enumerate(CHANNEL_ORDER):
        z_min, z_max = pool[:, :, i].min(), pool[:, :, i].max()
        margen = z_min - threshold_z[i]
        print(f"  {channel:20s}: Z min={z_min:8.3f}  Z max={z_max:8.3f}  margen sobre umbral={margen:6.3f}")
