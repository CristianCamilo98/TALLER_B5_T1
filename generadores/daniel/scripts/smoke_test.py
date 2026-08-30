"""Fast end-to-end mechanical check; this is not model training."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.data_adapter import load_canonical_donor_tensors  # noqa: E402
from generadores.daniel.src.diffusion import GaussianDiffusion  # noqa: E402
from generadores.daniel.src.network import TemporalDenoiser  # noqa: E402
from generadores.daniel.src.reproducibility import set_seed  # noqa: E402
from generadores.daniel.src.sampler import DDPMSampler  # noqa: E402
from generadores.daniel.src.temporary_normalizer import (  # noqa: E402
    TemporaryTickerChannelNormalizer,
)
from generadores.daniel.src.validation import validate_window_tensor  # noqa: E402


def _load_config() -> dict:
    path = REPOSITORY_ROOT / "generadores/daniel/config/diffusion.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    config = _load_config()
    seed = int(config["reproducibility"]["seed"])
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    environment = set_seed(seed)

    train, validation = load_canonical_donor_tensors(REPOSITORY_ROOT)
    normalizer = TemporaryTickerChannelNormalizer().fit(train.tensor, train.tickers)
    normalized_train = normalizer.transform(train.tensor, train.tickers)
    normalized_validation = normalizer.transform(validation.tensor, validation.tickers)
    validate_window_tensor(normalized_train, expected_count=4910, name="normalized_train")
    validate_window_tensor(
        normalized_validation, expected_count=380, name="normalized_validation"
    )

    model_config = config["model"]
    diffusion_config = config["diffusion"]
    model = TemporalDenoiser(
        input_length=model_config["input_length"],
        input_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
        time_embedding_dim=model_config["time_embedding_dim"],
    )
    diffusion = GaussianDiffusion(
        model,
        steps=diffusion_config["steps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    batch = normalized_train[:8]
    timesteps = torch.arange(batch.shape[0], dtype=torch.long) % diffusion.steps
    forward = model(batch, timesteps)
    optimizer.zero_grad(set_to_none=True)
    loss = diffusion.training_loss(batch)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not gradients_finite:
        raise RuntimeError("Smoke test produced non-finite gradients")
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["training"]["gradient_clip_norm"]
    )
    optimizer.step()

    sampler = DDPMSampler(diffusion)
    samples = sampler.sample(4, seed=seed)
    repeat = sampler.sample(4, seed=seed)
    other = sampler.sample(4, seed=seed + 1)
    report = {
        "environment": environment,
        "train_shape": list(train.tensor.shape),
        "validation_shape": list(validation.tensor.shape),
        "forward_shape": list(forward.shape),
        "loss": float(loss.detach()),
        "backward": "PASS",
        "gradients_finite": gradients_finite,
        "optimizer_step": "PASS",
        "sample_shape": list(samples.shape),
        "samples_finite": bool(torch.isfinite(samples).all()),
        "same_seed_equal": bool(torch.equal(samples, repeat)),
        "different_seed_different": bool(not torch.equal(samples, other)),
        "trainable_parameters": model.trainable_parameters,
    }
    if report["forward_shape"] != [8, 65, 3]:
        raise RuntimeError("Unexpected forward shape")
    if report["sample_shape"] != [4, 65, 3] or not report["samples_finite"]:
        raise RuntimeError("Mechanical sampling failed")
    if not report["same_seed_equal"] or not report["different_seed_different"]:
        raise RuntimeError("Sampling seed contract failed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
