"""Run Daniel-only normalized real-vs-synthetic visual diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.data_adapter import load_donor_windows  # noqa: E402
from generadores.daniel.src.final_pools import (  # noqa: E402
    load_final_pool,
    write_json_artifact,
)
from generadores.daniel.src.frozen_runs import sha256_file  # noqa: E402
from generadores.daniel.src.temporary_normalizer import (  # noqa: E402
    GlobalChannelNormalizer,
)
from generadores.daniel.src.validation import CHANNEL_ORDER  # noqa: E402
from generadores.daniel.src.visual_diagnostics_v2 import (  # noqa: E402
    BALANCED_COUNT,
    DIAGNOSTIC_SEED,
    deterministic_balanced_samples,
    logistic_c2st,
    plot_logistic_c2st,
    plot_marginals,
    plot_tsne,
    tsne_projection,
)


REAL_SPLIT = "donor_validation"
SYNTHETIC_SPACE = "normalized"
SYNTHETIC_POOL_RELATIVE = Path(
    "generadores/daniel/artifacts/samples/"
    "diffusion_seed42_global_channel_normalized_5000.npz"
)
FINAL_POOL_MANIFEST_RELATIVE = Path(
    "generadores/daniel/artifacts/manifests/"
    "diffusion_seed42_global_channel_final_pool.json"
)
NORMALIZER_RELATIVE = Path(
    "generadores/daniel/artifacts/manifests/"
    "diffusion_seed42_global_channel_normalizer.json"
)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def diagnostic_output_paths(artifact_root: Path) -> dict[str, Path]:
    return {
        "marginals": artifact_root / "figures/diffusion_global_channel_real_vs_synthetic_marginals.png",
        "logistic": artifact_root / "figures/diffusion_global_channel_real_vs_synthetic_logistic.png",
        "tsne": artifact_root / "figures/diffusion_global_channel_real_vs_synthetic_tsne.png",
        "manifest": artifact_root
        / "manifests/diffusion_global_channel_real_vs_synthetic_diagnostics.json",
    }


def main() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("Visual diagnostics require a clean versioned worktree")
    started = perf_counter()
    generation_commit = _git("rev-parse", "HEAD")
    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    outputs = diagnostic_output_paths(artifact_root)

    # The persisted train-only normalizer is reused verbatim. Validation is
    # transform-only: its observations cannot influence scaler parameters.
    normalizer_path = REPOSITORY_ROOT / NORMALIZER_RELATIVE
    pool_manifest = json.loads(
        (REPOSITORY_ROOT / FINAL_POOL_MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )
    normalizer_sha256 = pool_manifest["normalizer_sha256"]
    if sha256_file(normalizer_path) != normalizer_sha256:
        raise RuntimeError("Frozen train-only normalizer hash mismatch")
    normalizer = GlobalChannelNormalizer.load_json(normalizer_path)
    validation = load_donor_windows(REAL_SPLIT, REPOSITORY_ROOT, dtype=torch.float64)
    real_normalized = normalizer.transform(validation.tensor).numpy()

    synthetic_path = REPOSITORY_ROOT / SYNTHETIC_POOL_RELATIVE
    if sha256_file(synthetic_path) != pool_manifest["normalized_pool_sha256"]:
        raise RuntimeError("Final normalized seed-42 pool hash mismatch")
    stored_pool = load_final_pool(synthetic_path)
    if stored_pool["space"] != SYNTHETIC_SPACE or stored_pool["training_seed"] != 42:
        raise RuntimeError("Synthetic diagnostic input is not seed-42 normalized space")
    synthetic_pool = stored_pool["samples"].numpy()
    real_balanced, synthetic_balanced, subset_indices = deterministic_balanced_samples(
        real_normalized, synthetic_pool, count=BALANCED_COUNT, seed=DIAGNOSTIC_SEED
    )

    # Marginals use the complete synthetic pool for a stable view of centre,
    # spread, and tails. C2ST/t-SNE below use balanced 380-vs-380 observations.
    marginal_rows = plot_marginals(real_normalized, synthetic_pool, outputs["marginals"])
    classifier = logistic_c2st(real_balanced, synthetic_balanced)
    plot_logistic_c2st(classifier, outputs["logistic"])
    tsne = tsne_projection(real_balanced, synthetic_balanced)
    plot_tsne(tsne, outputs["tsne"])

    manifest = {
        "run_id": "diffusion_global_channel_real_vs_synthetic_diagnostics",
        "diagnostic_scope": "generator_local_normalized_space_not_common_fidelity",
        "real_source": "data/features/windows/donor_validation.parquet",
        "real_source_sha256": validation.input_sha256,
        "real_count": int(len(real_balanced)),
        "synthetic_source": SYNTHETIC_POOL_RELATIVE.as_posix(),
        "synthetic_source_sha256": pool_manifest["normalized_pool_sha256"],
        "synthetic_pool_count": int(len(synthetic_pool)),
        "synthetic_count": int(len(synthetic_balanced)),
        "synthetic_subset_indices": subset_indices.tolist(),
        "seed": DIAGNOSTIC_SEED,
        "window_shape": [65, 3],
        "flattened_features": 195,
        "channels": list(CHANNEL_ORDER),
        "normalizer_path": NORMALIZER_RELATIVE.as_posix(),
        "normalizer_sha256": normalizer_sha256,
        "marginal_statistics": marginal_rows,
        "logistic": {
            "pipeline": ["StandardScaler", "LogisticRegression"],
            "labels": {"real": 0, "synthetic": 1},
            "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "evaluation": classifier["evaluation"],
            "roc_auc": classifier["roc_auc"],
            "accuracy": classifier["accuracy"],
        },
        "pca": tsne["pca"],
        "tsne": tsne["tsne"],
        "tsne_runtime_seconds": tsne["runtime_seconds"],
        "figures": {key: path.relative_to(REPOSITORY_ROOT).as_posix() for key, path in outputs.items() if key != "manifest"},
        "git_commit": generation_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": perf_counter() - started,
        "target_data_used": False,
        "downstream_used": False,
    }
    write_json_artifact(manifest, outputs["manifest"])
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
