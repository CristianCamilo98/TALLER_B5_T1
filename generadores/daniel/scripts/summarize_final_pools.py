"""Build the local three-seed final-pool summary and sanity plots."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.final_pool_plots import generate_final_pool_figures  # noqa: E402
from generadores.daniel.src.final_pools import (  # noqa: E402
    build_final_pool_summary,
    load_final_pool,
)
from generadores.daniel.src.temporary_nvda_calibration import (  # noqa: E402
    load_nvda_visible_daily,
)


def main() -> None:
    artifact_root = REPOSITORY_ROOT / "generadores/daniel/artifacts"
    seeds = (42, 123, 2026)
    manifests = [
        json.loads(
            (artifact_root / f"manifests/diffusion_seed{seed}_final_pool.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in seeds
    ]
    summary = build_final_pool_summary(manifests)
    summary_path = artifact_root / "manifests/diffusion_final_pools_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    pools = {
        seed: load_final_pool(
            artifact_root / f"samples/diffusion_seed{seed}_nvda_like_5000.npz"
        )["samples"].numpy()
        for seed in seeds
    }
    visible = load_nvda_visible_daily(REPOSITORY_ROOT)
    figures = generate_final_pool_figures(
        summary, pools, visible, artifact_root / "figures"
    )
    print(summary.to_string(index=False))
    print(f"summary={summary_path}")
    print(json.dumps({key: str(value) for key, value in figures.items()}, indent=2))


if __name__ == "__main__":
    main()
