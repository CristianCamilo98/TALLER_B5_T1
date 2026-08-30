from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_fidelity import plot_marginals


def test_marginal_plot_outputs_use_common_channel_names(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    real = rng.normal(size=(8, 65, 3)).astype(np.float32)
    synthetic = {"method": rng.normal(size=(8, 65, 3)).astype(np.float32)}
    paths = plot_marginals(real, synthetic, tmp_path)
    assert [path.name for path in paths] == [
        "marginals_log_return.png",
        "marginals_log_high_low_range.png",
        "marginals_log1p_volume.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
