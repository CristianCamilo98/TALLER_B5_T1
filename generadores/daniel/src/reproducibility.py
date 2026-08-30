"""Local reproducibility controls and environment reporting."""

from __future__ import annotations

import platform
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_algorithms: bool = False) -> dict:
    """Seed Python, NumPy, PyTorch CPU, and all available CUDA devices.

    Deterministic algorithms are opt-in and use warn-only mode because some
    accelerator operations may lack deterministic implementations. Sampling
    additionally uses an explicit local `torch.Generator`.
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms
    return environment_report(seed, deterministic_algorithms=deterministic_algorithms)


def environment_report(seed: int, *, deterministic_algorithms: bool) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "seed": int(seed),
        "device": device,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": bool(deterministic_algorithms),
        "sampling_uses_explicit_generator": True,
    }
