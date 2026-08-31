import numpy as np


def fit_scaler(train_values: np.ndarray) -> dict:
    """
    Ajusta media/desviacion por canal, en float64 estricto (ddof=0),
    axes=(0,1), tal y como exige el contrato comun del equipo.
    Fuerza el calculo en float64 aunque el array de entrada ya lo sea --
    proteccion extra para que nunca vuelva a colarse una perdida de
    precision silenciosa si algun futuro caller pasa datos en float32.
    """
    data64 = train_values.astype(np.float64)
    mean = data64.mean(axis=(0, 1), dtype=np.float64)
    std = data64.std(axis=(0, 1), ddof=0, dtype=np.float64)
    std = np.where(std == 0, 1e-8, std)
    return {"mean": mean, "std": std}


def apply_scaler(data: np.ndarray, scaler: dict) -> np.ndarray:
    """(X - mean) / std. La conversion a float32 se hace DESPUES de esto,
    en el punto donde se construyen los tensores del modelo -- no aqui."""
    return (data.astype(np.float64) - scaler["mean"]) / scaler["std"]


def inverse_scaler(data_scaled: np.ndarray, scaler: dict) -> np.ndarray:
    return (data_scaled * scaler["std"]) + scaler["mean"]
