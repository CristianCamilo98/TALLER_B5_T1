"""Verify checkpoints, sanity sampling, and seed-42 baseline reproduction."""

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
    histories_numerically_equal,
    load_frozen_manifests,
    model_states_equal,
    sha256_file,
    validate_frozen_manifests,
)
from generadores.daniel.src.network import TemporalDenoiser  # noqa: E402
from generadores.daniel.src.run_artifacts import FROZEN_TRAINING_SEEDS  # noqa: E402
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
    manifests = load_frozen_manifests(artifact_root)
    validate_frozen_manifests(manifests)
    verification: dict = {"device": device, "runs": {}}

    frozen_seed42_payload = None
    frozen_seed42_samples = None
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
        if seed == 42:
            frozen_seed42_payload = best_payload
            frozen_seed42_samples = first

    baseline_checkpoint = artifact_root / "checkpoints/diffusion_seed42_baseline/best_model.pt"
    baseline_history = artifact_root / "histories/diffusion_seed42_baseline.csv"
    frozen_history = artifact_root / "histories/diffusion_seed42_frozen.csv"
    baseline_payload, baseline_sampler = _load_sampler(
        baseline_checkpoint, manifests[0]["effective_config"], device
    )
    states_equal, state_max_abs = model_states_equal(
        baseline_payload["model_state_dict"], frozen_seed42_payload["model_state_dict"]
    )
    histories_equal, history_max_abs = histories_numerically_equal(
        baseline_history, frozen_history
    )
    baseline_samples = baseline_sampler.sample(16, seed=42)
    samples_equal = bool(torch.equal(baseline_samples, frozen_seed42_samples))
    reproduction = {
        "baseline_best_epoch": int(baseline_payload["epoch"]),
        "frozen_best_epoch": int(frozen_seed42_payload["epoch"]),
        "baseline_best_validation_loss": float(baseline_payload["validation_loss"]),
        "frozen_best_validation_loss": float(frozen_seed42_payload["validation_loss"]),
        "state_dict_exactly_equal": states_equal,
        "state_dict_max_abs_difference": state_max_abs,
        "history_metrics_exactly_equal": histories_equal,
        "history_metrics_max_abs_difference": history_max_abs,
        "sanity_samples_exactly_equal": samples_equal,
    }
    if not (states_equal and histories_equal and samples_equal):
        raise RuntimeError(f"Seed-42 frozen run did not reproduce baseline: {reproduction}")
    verification["seed42_reproduction"] = reproduction
    output = artifact_root / "manifests/diffusion_frozen_seeds_verification.json"
    output.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    print(f"verification={output}")


if __name__ == "__main__":
    main()
