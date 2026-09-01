"""Small NumPy RealNVP normalizing flow for flattened OHLCV windows.

The project environment does not currently ship with PyTorch or TensorFlow, so
this module keeps the neural model explicit: affine coupling layers, trainable
MLPs, exact inverse, log likelihood, log-det-Jacobian, and Adam updates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))
WINDOW_LENGTH = 65
N_CHANNELS = 3
FEATURE_DIM = WINDOW_LENGTH * N_CHANNELS


@dataclass(frozen=True)
class FlowConfig:
    input_dim: int = FEATURE_DIM
    hidden_dims: tuple[int, ...] = (96, 96)
    n_coupling_layers: int = 6
    scale_clip: float = 1.5
    init_scale: float = 0.04
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FlowConfig":
        values = dict(payload)
        values["hidden_dims"] = tuple(int(value) for value in values["hidden_dims"])
        return cls(**values)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10000
    batch_size: int = 256
    learning_rate: float = 8.0e-4
    weight_decay: float = 1.0e-5
    validation_fraction: float = 0.10
    gradient_clip_norm: float = 25.0
    patience: int = 200
    min_delta: float = 1.0e-5
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (WINDOW_LENGTH, N_CHANNELS):
        raise ValueError("windows must have shape (N, 65, 3)")
    return values.reshape(values.shape[0], FEATURE_DIM)


def unflatten_windows(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64)
    if flat.ndim != 2 or flat.shape[1] != FEATURE_DIM:
        raise ValueError("values must have shape (N, 195)")
    return flat.reshape(flat.shape[0], WINDOW_LENGTH, N_CHANNELS).astype(np.float32)


def _mask(input_dim: int, layer_index: int) -> np.ndarray:
    return ((np.arange(input_dim) + layer_index) % 2).astype(np.float64)


class DenseTanhNetwork:
    """Tanh MLP used inside each affine coupling transform."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        rng: np.random.Generator,
        init_scale: float,
    ) -> None:
        dims = (input_dim, *hidden_dims, output_dim)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = init_scale * np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

    def forward(self, values: np.ndarray, *, cache: bool = False) -> Any:
        activations = [values]
        pre_activations: list[np.ndarray] = []
        current = values
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            pre = current @ weight + bias
            current = np.tanh(pre)
            pre_activations.append(pre)
            activations.append(current)
        output = current @ self.weights[-1] + self.biases[-1]
        if cache:
            return output, (activations, pre_activations)
        return output

    def backward(
        self,
        grad_output: np.ndarray,
        cache: Any,
        prefix: str,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        activations, pre_activations = cache
        grads: dict[str, np.ndarray] = {}
        grad = grad_output

        last = len(self.weights) - 1
        grads[f"{prefix}.W{last}"] = activations[-1].T @ grad
        grads[f"{prefix}.b{last}"] = grad.sum(axis=0)
        grad = grad @ self.weights[last].T

        for idx in range(last - 1, -1, -1):
            grad = grad * (1.0 - np.tanh(pre_activations[idx]) ** 2)
            grads[f"{prefix}.W{idx}"] = activations[idx].T @ grad
            grads[f"{prefix}.b{idx}"] = grad.sum(axis=0)
            grad = grad @ self.weights[idx].T
        return grad, grads

    def parameters(self, prefix: str) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {}
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            params[f"{prefix}.W{idx}"] = weight
            params[f"{prefix}.b{idx}"] = bias
        return params

    def load_parameters(self, prefix: str, state: dict[str, np.ndarray]) -> None:
        for idx in range(len(self.weights)):
            self.weights[idx][...] = state[f"{prefix}.W{idx}"]
            self.biases[idx][...] = state[f"{prefix}.b{idx}"]


class AffineCoupling:
    def __init__(self, *, mask: np.ndarray, config: FlowConfig, rng: np.random.Generator) -> None:
        self.mask = mask.astype(np.float64)
        self.inverse_mask = 1.0 - self.mask
        self.scale_clip = float(config.scale_clip)
        self.network = DenseTanhNetwork(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=2 * config.input_dim,
            rng=rng,
            init_scale=config.init_scale,
        )

    def forward(self, values: np.ndarray, *, cache: bool = False) -> Any:
        masked = values * self.mask
        network_out, network_cache = self.network.forward(masked, cache=True)
        raw_scale, shift_raw = np.split(network_out, 2, axis=1)
        scale = self.scale_clip * np.tanh(raw_scale) * self.inverse_mask
        shift = shift_raw * self.inverse_mask
        exp_scale = np.exp(scale)
        transformed = masked + self.inverse_mask * (values * exp_scale + shift)
        log_det = scale.sum(axis=1)
        if not cache:
            return transformed, log_det
        return transformed, log_det, {
            "input": values,
            "raw_scale": raw_scale,
            "scale": scale,
            "exp_scale": exp_scale,
            "network_cache": network_cache,
        }

    def inverse(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        masked = values * self.mask
        network_out = self.network.forward(masked)
        raw_scale, shift_raw = np.split(network_out, 2, axis=1)
        scale = self.scale_clip * np.tanh(raw_scale) * self.inverse_mask
        shift = shift_raw * self.inverse_mask
        recovered = masked + self.inverse_mask * ((values - shift) * np.exp(-scale))
        return recovered, -scale.sum(axis=1)

    def backward(
        self,
        grad_output: np.ndarray,
        grad_log_det: np.ndarray,
        cache: dict[str, Any],
        prefix: str,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        values = cache["input"]
        raw_scale = cache["raw_scale"]
        exp_scale = cache["exp_scale"]
        batch_grad_log_det = grad_log_det.reshape(-1, 1)

        grad_input = grad_output * (self.mask + self.inverse_mask * exp_scale)
        grad_scale = (
            grad_output * self.inverse_mask * values * exp_scale
            + batch_grad_log_det * self.inverse_mask
        )
        grad_shift = grad_output * self.inverse_mask
        grad_raw_scale = grad_scale * self.scale_clip * (1.0 - np.tanh(raw_scale) ** 2)
        grad_network_out = np.concatenate([grad_raw_scale, grad_shift], axis=1)
        grad_masked, grads = self.network.backward(
            grad_network_out,
            cache["network_cache"],
            f"{prefix}.net",
        )
        grad_input += grad_masked * self.mask
        return grad_input, grads

    def parameters(self, prefix: str) -> dict[str, np.ndarray]:
        return self.network.parameters(f"{prefix}.net")

    def load_parameters(self, prefix: str, state: dict[str, np.ndarray]) -> None:
        self.network.load_parameters(f"{prefix}.net", state)


class RealNVP:
    def __init__(self, config: FlowConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.layers = [
            AffineCoupling(mask=_mask(config.input_dim, idx), config=config, rng=rng)
            for idx in range(config.n_coupling_layers)
        ]

    def forward(self, values: np.ndarray, *, cache: bool = False) -> Any:
        current = np.asarray(values, dtype=np.float64)
        log_det = np.zeros(current.shape[0], dtype=np.float64)
        caches = []
        for idx, layer in enumerate(self.layers):
            if cache:
                current, layer_log_det, layer_cache = layer.forward(current, cache=True)
                caches.append((idx, layer_cache))
            else:
                current, layer_log_det = layer.forward(current)
            log_det += layer_log_det
        if cache:
            return current, log_det, caches
        return current, log_det

    def inverse(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(values, dtype=np.float64)
        log_det = np.zeros(current.shape[0], dtype=np.float64)
        for layer in reversed(self.layers):
            current, layer_log_det = layer.inverse(current)
            log_det += layer_log_det
        return current, log_det

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        latent, log_det = self.forward(values)
        base = -0.5 * np.sum(latent * latent + LOG_2PI, axis=1)
        return base + log_det

    def negative_log_likelihood(self, values: np.ndarray) -> float:
        return float(-np.mean(self.log_prob(values)))

    def loss_and_gradients(self, values: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        batch = np.asarray(values, dtype=np.float64)
        latent, log_det, caches = self.forward(batch, cache=True)
        per_row = 0.5 * np.sum(latent * latent + LOG_2PI, axis=1) - log_det
        loss = float(per_row.mean())

        grad = latent / batch.shape[0]
        grad_log_det = np.full(batch.shape[0], -1.0 / batch.shape[0], dtype=np.float64)
        grads: dict[str, np.ndarray] = {}
        for idx, layer_cache in reversed(caches):
            grad, layer_grads = self.layers[idx].backward(
                grad,
                grad_log_det,
                layer_cache,
                f"layers.{idx}",
            )
            grads.update(layer_grads)
        return loss, grads

    def parameters(self) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {}
        for idx, layer in enumerate(self.layers):
            params.update(layer.parameters(f"layers.{idx}"))
        return params

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.parameters().items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for idx, layer in enumerate(self.layers):
            layer.load_parameters(f"layers.{idx}", state)


class AdamOptimizer:
    def __init__(self, *, learning_rate: float, weight_decay: float) -> None:
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1.0e-8
        self.step_index = 0
        self.moments_1: dict[str, np.ndarray] = {}
        self.moments_2: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.step_index += 1
        for name, param in params.items():
            grad = grads[name]
            if self.weight_decay and ".W" in name:
                grad = grad + self.weight_decay * param
            m = self.moments_1.setdefault(name, np.zeros_like(param))
            v = self.moments_2.setdefault(name, np.zeros_like(param))
            m[...] = self.beta1 * m + (1.0 - self.beta1) * grad
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            m_hat = m / (1.0 - self.beta1**self.step_index)
            v_hat = v / (1.0 - self.beta2**self.step_index)
            param[...] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


def _clip_gradients(grads: dict[str, np.ndarray], max_norm: float) -> float:
    squared = float(sum(np.sum(grad * grad) for grad in grads.values()))
    norm = float(np.sqrt(squared))
    if norm > max_norm > 0:
        scale = max_norm / (norm + 1.0e-12)
        for grad in grads.values():
            grad *= scale
    return norm


def _nll_in_batches(model: RealNVP, values: np.ndarray, batch_size: int) -> float:
    losses = []
    weights = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        losses.append(model.negative_log_likelihood(batch))
        weights.append(len(batch))
    return float(np.average(losses, weights=weights))


def train_real_nvp(
    windows: np.ndarray,
    *,
    flow_config: FlowConfig,
    training_config: TrainingConfig,
) -> tuple[RealNVP, list[dict[str, float]]]:
    values = flatten_windows(windows)
    rng = np.random.default_rng(training_config.seed)
    indices = rng.permutation(values.shape[0])
    n_validation = max(1, int(round(values.shape[0] * training_config.validation_fraction)))
    validation_idx = indices[:n_validation]
    train_idx = indices[n_validation:]
    if len(train_idx) == 0:
        raise ValueError("validation_fraction leaves no training rows")

    train_values = values[train_idx]
    validation_values = values[validation_idx]
    model = RealNVP(flow_config)
    optimizer = AdamOptimizer(
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    best_state = model.state_dict()
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.epochs + 1):
        order = rng.permutation(len(train_values))
        batch_losses: list[float] = []
        grad_norms: list[float] = []
        for start in range(0, len(order), training_config.batch_size):
            batch = train_values[order[start : start + training_config.batch_size]]
            loss, grads = model.loss_and_gradients(batch)
            grad_norms.append(_clip_gradients(grads, training_config.gradient_clip_norm))
            optimizer.step(model.parameters(), grads)
            batch_losses.append(loss)

        train_nll = _nll_in_batches(model, train_values, training_config.batch_size)
        validation_nll = _nll_in_batches(model, validation_values, training_config.batch_size)
        row = {
            "epoch": float(epoch),
            "batch_nll": float(np.mean(batch_losses)),
            "train_nll": train_nll,
            "validation_nll": validation_nll,
            "grad_norm": float(np.mean(grad_norms)),
        }
        history.append(row)
        if validation_nll + training_config.min_delta < best_validation:
            best_validation = validation_nll
            best_state = model.state_dict()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    model.load_state_dict(best_state)
    return model, history


def sample_windows(
    model: RealNVP,
    *,
    n_windows: int,
    seed: int,
    temperature: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, temperature, size=(n_windows, model.config.input_dim))
    generated, _ = model.inverse(latent)
    if not np.isfinite(generated).all():
        raise RuntimeError("Normalizing Flow generated non-finite values")
    return unflatten_windows(generated)


def save_checkpoint(
    path: Path,
    model: RealNVP,
    *,
    metadata: dict[str, Any],
    history: list[dict[str, float]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = model.state_dict()
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    payload["history_json"] = np.asarray(json.dumps(history, sort_keys=True))
    np.savez_compressed(path, **payload)
    return path


def load_checkpoint(path: Path) -> tuple[RealNVP, dict[str, Any], list[dict[str, float]]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        history = json.loads(str(payload["history_json"].item()))
        config = FlowConfig.from_dict(metadata["architecture"]["flow_config"])
        model = RealNVP(config)
        state = {
            key: np.asarray(payload[key], dtype=np.float64)
            for key in payload.files
            if key not in {"metadata_json", "history_json"}
        }
    model.load_state_dict(state)
    return model, metadata, history
