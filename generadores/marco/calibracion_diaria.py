import numpy as np
import pandas as pd


def unique_daily_values(window_parquet_path: str) -> np.ndarray:
    """
    Reconstruye los días únicos subyacentes a partir de ventanas con stride=1
    (cada ventana añade exactamente 1 día nuevo respecto a la anterior).
    Evita contar los días centrales hasta 62 veces frente a los de borde,
    que solo aparecen 1 vez -- el sesgo que señaló el compañero.
    """
    df = pd.read_parquet(window_parquet_path).sort_values("window_start_date").reset_index(drop=True)
    windows = np.stack([np.asarray(row, dtype="float32").reshape(65, 3) for row in df["features_flat"]])

    primera_ventana_completa = windows[0]              # los 65 días de la primera ventana
    ultimo_dia_de_cada_resto = windows[1:, -1, :]       # el día "nuevo" de cada ventana siguiente
    return np.concatenate([primera_ventana_completa, ultimo_dia_de_cada_resto], axis=0)


nvda_daily = unique_daily_values("data/features/windows/nvda_visible.parquet")
print("Días únicos reconstruidos:", nvda_daily.shape)  # esperado: (126, 3)

mean_nvda = nvda_daily.mean(axis=0)
std_nvda = nvda_daily.std(axis=0, ddof=0)   # ddof=0 ya es el default de NumPy, coincide con lo que pide tu compañero