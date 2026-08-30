from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generadores.daniel.scripts.evaluate_visual_diagnostics import (
    diagnostic_output_paths,
)
from generadores.daniel.src.final_pool_plots import rejection_annotations
from generadores.daniel.src.run_artifacts import (
    FROZEN_BASELINE,
    LONG_TRAINING_MAX_EPOCHS,
    LONG_TRAINING_PATIENCE,
    long_training_diagnostic_config,
    validate_long_training_config,
)
from generadores.daniel.src.training_diagnostics import (
    best_history_point,
    dynamic_zoom_bounds,
    plot_training_views,
)
from generadores.daniel.src.visual_diagnostics_v2 import (
    deterministic_balanced_samples,
    flatten_windows,
    logistic_c2st,
    shared_histogram_bins,
    tsne_projection,
)


ROOT = Path(__file__).resolve().parents[3]


def _windows(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, 65, 3)).astype(np.float32)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch": np.arange(1, 51),
            "train_loss": np.linspace(1.0, 0.4, 50),
            "validation_loss": np.r_[np.linspace(0.9, 0.6, 20), np.linspace(0.61, 0.8, 30)],
            "learning_rate": 0.0002,
            "epoch_seconds": 1.0,
        }
    )


def test_balanced_subset_is_380_per_class_and_deterministic() -> None:
    real = _windows(380, 1)
    synthetic = _windows(5000, 2)
    first = deterministic_balanced_samples(real, synthetic)
    second = deterministic_balanced_samples(real, synthetic)
    assert first[0].shape == first[1].shape == (380, 65, 3)
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert len(np.unique(first[2])) == 380


def test_window_flattening_preserves_195_feature_contract() -> None:
    values = np.arange(2 * 65 * 3, dtype=np.float32).reshape(2, 65, 3)
    flattened = flatten_windows(values)
    assert flattened.shape == (2, 195)
    assert np.array_equal(flattened[0], values[0].reshape(195))


def test_marginal_histogram_uses_shared_edges() -> None:
    real = np.array([-2.0, -1.0, 0.0])
    synthetic = np.array([1.0, 2.0, 3.0])
    bins = shared_histogram_bins(real, synthetic, n_bins=10)
    assert len(bins) == 11
    assert bins[0] == -2.0 and bins[-1] == 3.0


def test_logistic_c2st_is_balanced_and_out_of_fold() -> None:
    real = _windows(380, 3)
    synthetic = _windows(380, 4) + 0.1
    result = logistic_c2st(real, synthetic)
    assert result["evaluation"] == "out_of_fold"
    assert result["n_splits"] == 5
    assert result["labels"].tolist().count(0) == 380
    assert result["labels"].tolist().count(1) == 380
    assert set(result["fold_ids"]) == set(range(5))
    assert np.isfinite([result["roc_auc"], result["accuracy"]]).all()


def test_tsne_uses_joint_balanced_labels_and_documented_config() -> None:
    result = tsne_projection(_windows(40, 5), _windows(40, 6))
    assert result["embedding"].shape == (80, 2)
    assert np.bincount(result["labels"]).tolist() == [40, 40]
    assert result["pca"] == {"n_components": 30, "random_state": 42}
    assert result["tsne"]["perplexity"] == 30
    assert result["tsne"]["init"] == "pca"
    assert result["tsne"]["learning_rate"] == "auto"


def test_visual_diagnostic_paths_are_local_runtime_artifacts(tmp_path) -> None:
    paths = diagnostic_output_paths(tmp_path)
    assert {path.suffix for path in paths.values()} == {".png", ".json"}
    assert all(path.is_relative_to(tmp_path) for path in paths.values())


def test_visual_entrypoint_has_no_target_data_access() -> None:
    script = (
        ROOT / "generadores/daniel/scripts/evaluate_visual_diagnostics.py"
    ).read_text(encoding="utf-8").lower()
    for forbidden in ("nvda_visible", "nvda_hidden", "nvda_full_history", "test_index"):
        assert forbidden not in script
    assert 'real_split = "donor_validation"' in script
    assert 'synthetic_space = "normalized"' in script


def test_long_config_changes_only_epochs_and_patience() -> None:
    source = deepcopy(FROZEN_BASELINE)
    effective = long_training_diagnostic_config(source)
    validate_long_training_config(effective)
    differences = []
    for section, values in effective.items():
        for key, value in values.items():
            if value != source[section][key]:
                differences.append(f"{section}.{key}")
    assert differences == [
        "training.max_epochs",
        "training.early_stopping_patience",
    ]
    assert effective["training"]["max_epochs"] == LONG_TRAINING_MAX_EPOCHS
    assert effective["training"]["early_stopping_patience"] == LONG_TRAINING_PATIENCE
    assert source == FROZEN_BASELINE
    drifted = deepcopy(effective)
    drifted["training"]["learning_rate"] = 1.0
    with pytest.raises(ValueError, match="two approved"):
        validate_long_training_config(drifted)


def test_best_checkpoint_and_zoom_are_validation_driven_and_dynamic(tmp_path) -> None:
    history = _history()
    assert best_history_point(history) == (20, 0.6)
    assert dynamic_zoom_bounds(history) == (10, 50)
    history_path = tmp_path / "history.csv"
    history.to_csv(history_path, index=False)
    result = plot_training_views(
        history_path,
        tmp_path / "full.png",
        tmp_path / "zoom.png",
        run_id="diagnostic",
    )
    assert result == {
        "best_epoch": 20,
        "best_validation_loss": 0.6,
        "stopping_epoch": 50,
        "zoom_start": 10,
        "zoom_end": 50,
    }
    assert (tmp_path / "full.png").is_file()
    assert (tmp_path / "zoom.png").is_file()


def test_zero_rejection_annotations_report_exact_counts() -> None:
    summary = pd.DataFrame(
        {
            "seed": [42, 123, 2026],
            "n_accepted": [5000, 5000, 5000],
            "n_rejected": [0, 0, 0],
            "rejection_rate": [0.0, 0.0, 0.0],
        }
    )
    assert rejection_annotations(summary) == [
        "5000 accepted / 0 rejected\n0.0%",
        "5000 accepted / 0 rejected\n0.0%",
        "5000 accepted / 0 rejected\n0.0%",
    ]
