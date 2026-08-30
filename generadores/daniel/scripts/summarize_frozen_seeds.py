"""Create the local table and validation-loss plot for frozen seed runs."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.frozen_runs import (  # noqa: E402
    build_frozen_summary,
    load_frozen_manifests,
    write_frozen_summary,
)
from generadores.daniel.src.run_artifacts import (  # noqa: E402
    FROZEN_RUN_IDS,
    FROZEN_TRAINING_SEEDS,
    read_history,
)


def main() -> None:
    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    manifests = load_frozen_manifests(artifact_root)
    summary = build_frozen_summary(manifests)
    summary_path = artifact_root / "histories/diffusion_frozen_seeds_summary.csv"
    write_frozen_summary(summary, summary_path)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for seed in FROZEN_TRAINING_SEEDS:
        run_id = FROZEN_RUN_IDS[seed]
        history = read_history(artifact_root / "histories" / f"{run_id}.csv")
        axis.plot(
            history["epoch"],
            history["validation_loss"],
            label=f"seed {seed}",
            linewidth=1.4,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Deterministic validation epsilon-prediction MSE")
    axis.set_title("Frozen DDPM validation loss across training seeds")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure_path = artifact_root / "figures/diffusion_frozen_seeds_validation_loss.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    losses = summary["best_validation_loss"].to_numpy(dtype=np.float64)
    print(summary.to_string(index=False))
    print(f"mean_best_validation_loss={losses.mean():.12f}")
    print(f"std_best_validation_loss={losses.std(ddof=0):.12f}")
    print(f"min_best_validation_loss={losses.min():.12f}")
    print(f"max_best_validation_loss={losses.max():.12f}")
    print(f"summary={summary_path}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
