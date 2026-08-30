import numpy as np

def fit_scaler(data):
    """
    Calcula la media y desviación estándar para cada canal.
    data shape esperado: (N_ventanas, 65_dias, 3_canales)
    """
    # Calculamos la estadística colapsando las ventanas y los días (ejes 0 y 1),
    # dejando un array de tamaño (3,) correspondiente a cada canal.
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    
    # Prevenir divisiones por cero en canales sin varianza
    std[std == 0] = 1e-8
    
    return {"mean": mean, "std": std}

def apply_scaler(data, scaler):
    """
    Aplica Z-score standardization: (X - mu) / sigma
    El broadcasting de NumPy alinea automáticamente el array (3,) con el último eje de data.
    """
    return (data - scaler["mean"]) / scaler["std"]

def inverse_scaler(data_scaled, scaler):
    """
    Revierte la escala: X = (Z * sigma) + mu
    Vital para la fase de calibración y generación de muestras sintéticas.
    """
    return (data_scaled * scaler["std"]) + scaler["mean"]