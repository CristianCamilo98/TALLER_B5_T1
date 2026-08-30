"""Tiny fixed-noise overfit diagnostic; never a full donor training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.data_adapter import load_donor_windows  # noqa: E402
from generadores.daniel.src.diffusion import GaussianDiffusion  # noqa: E402
from generadores.daniel.src.network import TemporalDenoiser  # noqa: E402
from generadores.daniel.src.temporary_normalizer import (  # noqa: E402
    GlobalChannelNormalizer,
)
from generadores.daniel.src.trainer import tiny_overfit_diagnostic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    config_path = REPOSITORY_ROOT / "generadores/daniel/config/diffusion.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train = load_donor_windows(
        "donor_train", REPOSITORY_ROOT, dtype=torch.float64
    )
    normalizer = GlobalChannelNormalizer().fit(train.tensor)
    normalized = normalizer.transform(train.tensor)

    model_config = config["model"]
    model = TemporalDenoiser(
        input_length=model_config["input_length"],
        input_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
        time_embedding_dim=model_config["time_embedding_dim"],
    )
    diffusion_config = config["diffusion"]
    diffusion = GaussianDiffusion(
        model,
        steps=diffusion_config["steps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )
    result = tiny_overfit_diagnostic(
        diffusion,
        normalized[:32],
        steps=args.steps,
        learning_rate=0.001,
        seed=config["reproducibility"]["seed"],
        gradient_clip_norm=config["training"]["gradient_clip_norm"],
    )
    result["subset_shape"] = [32, 65, 3]
    result["uses_validation"] = False
    result["clear_reduction"] = result["reduction_percent"] >= 20.0
    print(json.dumps(result, indent=2))
    if not result["finite"] or not result["gradients_finite"]:
        raise SystemExit("Tiny-overfit numerical failure")
    if not result["clear_reduction"]:
        raise SystemExit("Tiny-overfit loss did not decrease clearly")


if __name__ == "__main__":
    main()
