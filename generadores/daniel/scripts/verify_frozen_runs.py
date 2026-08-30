"""Verify checkpoints and sanity sampling for global-normalized runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.diffusion import GaussianDiffusion  # noqa: E402
from generadores.daniel.src.frozen_runs import (  # noqa: E402
    load_global_channel_manifests,
    sha256_file,
    validate_frozen_manifests,
)
from generadores.daniel.src.network import TemporalDenoiser  # noqa: E402
from generadores.daniel.src.run_artifacts import (  # noqa: E402
    FROZEN_TRAINING_SEEDS,
    GLOBAL_CHANNEL_RUN_IDS,
)
from generadores.daniel.src.sampler import DDPMSampler  # noqa: E402


def _load_sampler(checkpoint_path: Path, effective_config: dict, device: str):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = effective_config["model"]
    model = TemporalDenoiser(
        input_length=model_config["input_length"],
        input_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
        time_embedding_dim=model_config["time_embedding_dim"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    diffusion_config = effective_config["diffusion"]
    diffusion = GaussianDiffusion(
        model,
        steps=diffusion_config["steps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    ).to(device)
    return payload, DDPMSampler(diffusion)


def main() -> None:
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    manifests = load_global_channel_manifests(artifact_root)
    validate_frozen_manifests(manifests, run_ids=GLOBAL_CHANNEL_RUN_IDS)
    verification: dict = {"device": device, "runs": {}}

    for manifest in manifests:
        seed = int(manifest["training_seed"])
        best_path = REPOSITORY_ROOT / manifest["best_checkpoint_path"]
        last_path = REPOSITORY_ROOT / manifest["last_checkpoint_path"]
        if sha256_file(best_path) != manifest["best_checkpoint_sha256"]:
            raise RuntimeError(f"Best checkpoint hash mismatch for seed {seed}")
        if sha256_file(last_path) != manifest["last_checkpoint_sha256"]:
            raise RuntimeError(f"Last checkpoint hash mismatch for seed {seed}")
        last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        best_payload, sampler = _load_sampler(
            best_path, manifest["effective_config"], device
        )
        first = sampler.sample(16, seed=seed)
        repeated = sampler.sample(16, seed=seed)
        if tuple(first.shape) != (16, 65, 3):
            raise RuntimeError(f"Invalid sanity sample shape for seed {seed}")
        if not torch.isfinite(first).all().item() or not torch.equal(first, repeated):
            raise RuntimeError(f"Sanity sampling failed for seed {seed}")
        verification["runs"][str(seed)] = {
            "best_checkpoint_load": True,
            "last_checkpoint_load": bool("model_state_dict" in last_payload),
            "sample_shape": list(first.shape),
            "sample_finite": True,
            "sample_reproducible": True,
        }
    output = artifact_root / "manifests/diffusion_global_channel_seeds_verification.json"
    output.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    print(f"verification={output}")


if __name__ == "__main__":
    main()
