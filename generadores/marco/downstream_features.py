import numpy as np

ANNUALIZATION = 252  # sesiones bursátiles/año -- factor estándar para anualizar volatilidad


def _realized_vol(returns: np.ndarray) -> np.ndarray:
    """RV = sqrt(252/n * sum(r_i^2)) -- volatilidad realizada anualizada.
    returns: [..., n] -> devuelve [...] (colapsa el último eje)."""
    n = returns.shape[-1]
    return np.sqrt(ANNUALIZATION / n * np.sum(returns ** 2, axis=-1))


def build_supervised_pairs(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    windows: [N, 65, 3], canales [log_return, log_high_low_range, log_volume].
    Contexto = días 0-59 (60 días); Futuro = días 60-64 (5 días).
    Devuelve X [N,8] (features), y [N] (target: rv_5 futura).
    """
    log_return, log_range, log_volume = windows[:, :, 0], windows[:, :, 1], windows[:, :, 2]

    context_return = log_return[:, :60]
    context_range = log_range[:, :60]
    context_volume = log_volume[:, :60]
    future_return = log_return[:, 60:65]

    X = np.stack([
        _realized_vol(context_return[:, -5:]),                    # rv_5
        _realized_vol(context_return[:, -20:]),                   # rv_20
        _realized_vol(context_return[:, -60:]),                   # rv_60
        np.mean(np.abs(context_return[:, -20:]), axis=-1),        # mean_abs_return_20
        np.sum(context_return[:, -20:], axis=-1),                 # momentum_20
        np.mean(context_range[:, -20:], axis=-1),                 # mean_range_20
        np.mean(context_volume[:, -20:], axis=-1),                # mean_volume_20
        np.std(context_volume[:, -20:], axis=-1),                 # std_volume_20
    ], axis=-1).astype("float32")

    y = _realized_vol(future_return).astype("float32")
    return X, y