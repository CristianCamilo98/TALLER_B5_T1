import json
import numpy as np

scaler = np.load("scaler_donors.npz")
stats = {
    "fit_split": "donor_train",
    "fit_axes": [0, 1],
    "ddof": 0,
    "fit_dtype": "float64",
    "channel_order": ["log_return", "log_high_low_range", "log1p_volume"],
    "mean": scaler["mean"].tolist(),
    "std": scaler["std"].tolist(),
    "zero_variance_guard": "std == 0 -> 1e-8 (implementacion Marco; el contrato comun usa sigma < 1e-8 -> 1.0)",
    "zero_variance_guard_triggered": bool((scaler["std"] < 1e-8).any()),
}
Path = __import__("pathlib").Path
Path("generadores/marco/evidence").mkdir(parents=True, exist_ok=True)
with open("generadores/marco/evidence/scaler_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
