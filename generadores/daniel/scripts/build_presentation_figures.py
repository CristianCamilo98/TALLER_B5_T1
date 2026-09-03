"""Build DDPM-only presentation diagnostics from frozen project inputs.

The donor-validation curve is the official STRICT_FINAL C2ST. The donor-train
and NVDA-hidden curves are additional post-hoc diagnostics only: they are not
written to scientific tables and are not used for model selection or tuning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from build_readme_figures import (
    DDPM_OUTPUT,
    NORMALIZER,
    OUTPUT_DIR,
    REPO_ROOT,
    _configure_style,
    _frozen_c2st,
    _load_fidelity_windows,
    _save,
    _stack_windows,
)


COMMON_FIDELITY_CODE = REPO_ROOT / "common_pipeline" / "02_fidelity"
DONOR_TRAIN = REPO_ROOT / "data" / "features" / "windows" / "donor_train.parquet"
NVDA_FULL_HISTORY = (
    REPO_ROOT / "data" / "features" / "windows" / "nvda_full_history.parquet"
)
NVDA_HIDDEN_END = pd.Timestamp("2022-06-30")
SUBSAMPLE_SEED = 42

EXPECTED_SHA256 = {
    DONOR_TRAIN: "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f",
    NVDA_FULL_HISTORY: "19f651da100e6a304dda77831448d50a015e6eff5b9e019f6b3b2ffc6e908617",
    DDPM_OUTPUT: "bb9b5ad6b412fd785f73344cd765c56447b92de2b5827e40b3dc77d06e40a6c2",
}

NAVY = "#17324D"
TEAL = "#2A9D8F"
BLUE = "#3A6EA5"
MAGENTA = "#B35C9E"
GRAY = "#697681"
GRID = "#D9E1E8"


def _verify_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256[path]:
        raise RuntimeError(f"Frozen input SHA256 mismatch: {path}")


def _normalizer_statistics() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(NORMALIZER.read_text(encoding="utf-8"))
    if manifest.get("fit_split") != "donor_train":
        raise RuntimeError("Expected the canonical donor_train-only normalizer")
    mean = np.asarray(manifest["mean"], dtype=np.float64)
    std = np.asarray(manifest["std"], dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0.0):
        raise RuntimeError("Invalid canonical normalizer statistics")
    return mean, std


def _normalize_frame(frame: pd.DataFrame, source: Path) -> np.ndarray:
    mean, std = _normalizer_statistics()
    windows = _stack_windows(frame, source=source).astype(np.float64)
    return ((windows - mean) / std).astype(np.float32)


def _load_synthetic_pool() -> np.ndarray:
    frame = pd.read_parquet(
        DDPM_OUTPUT,
        columns=["source_model", "training_seed", "features_flat"],
    )
    if set(frame["source_model"]) != {"diffusion_ddpm"}:
        raise RuntimeError("Expected source_model=diffusion_ddpm")
    if set(frame["training_seed"]) != {42}:
        raise RuntimeError("Expected the official training seed 42")
    pool = _stack_windows(frame, source=DDPM_OUTPUT).astype(np.float32, copy=False)
    if pool.shape != (5000, 65, 3):
        raise RuntimeError("Expected the frozen 5,000-window DDPM pool")
    return pool


def _balanced_pair(real: np.ndarray, synthetic_pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(real), len(synthetic_pool))
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    if len(real) == count:
        balanced_real = real
    else:
        real_indices = rng.choice(len(real), size=count, replace=False)
        balanced_real = real[real_indices]
    synthetic_indices = rng.choice(len(synthetic_pool), size=count, replace=False)
    return balanced_real, synthetic_pool[synthetic_indices]


def _common_c2st(real: np.ndarray, synthetic: np.ndarray):
    sys.path.insert(0, str(COMMON_FIDELITY_CODE))
    try:
        from fidelity_core import c2st_out_of_fold
    finally:
        sys.path.pop(0)
    return c2st_out_of_fold(real, synthetic, random_state=42)


def _load_donor_train(synthetic_pool: np.ndarray):
    frame = pd.read_parquet(DONOR_TRAIN)
    real = _normalize_frame(frame, DONOR_TRAIN)
    balanced_real, synthetic = _balanced_pair(real, synthetic_pool)
    return frame, balanced_real, synthetic, _common_c2st(balanced_real, synthetic)


def _load_nvda_hidden(synthetic_pool: np.ndarray):
    frame = pd.read_parquet(NVDA_FULL_HISTORY)
    frame["window_start_date"] = pd.to_datetime(frame["window_start_date"])
    frame["window_end_date"] = pd.to_datetime(frame["window_end_date"])
    hidden = frame.loc[frame["window_end_date"].le(NVDA_HIDDEN_END)].copy()
    if hidden.empty or hidden["window_end_date"].max() != NVDA_HIDDEN_END:
        raise RuntimeError("Could not reconstruct the canonical NVDA hidden boundary")
    if hidden["ticker"].nunique() != 1 or hidden["ticker"].iloc[0] != "NVDA":
        raise RuntimeError("NVDA hidden must contain only NVDA windows")
    real = _normalize_frame(hidden, NVDA_FULL_HISTORY)
    balanced_real, synthetic = _balanced_pair(real, synthetic_pool)
    return hidden, balanced_real, synthetic, _common_c2st(balanced_real, synthetic)


def _plot_curve(axis, result, *, color: str, label: str) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        result.labels,
        result.probabilities,
        pos_label=1,
    )
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color=color,
        linewidth=2.4,
        label=f"{label} (AUC = {result.roc_auc:.3f})",
    )


def build_multi_reference_roc() -> None:
    for path in EXPECTED_SHA256:
        _verify_sha256(path)

    synthetic_pool = _load_synthetic_pool()
    donor_train_frame, donor_train, donor_train_synthetic, donor_train_result = (
        _load_donor_train(synthetic_pool)
    )
    donor_validation, donor_validation_synthetic = _load_fidelity_windows()
    donor_validation_result, strict_final_auc = _frozen_c2st(
        donor_validation,
        donor_validation_synthetic,
    )
    hidden_frame, nvda_hidden, nvda_hidden_synthetic, nvda_hidden_result = (
        _load_nvda_hidden(synthetic_pool)
    )

    fig, axis = plt.subplots(figsize=(10.6, 6.0))
    _plot_curve(
        axis,
        nvda_hidden_result,
        color=MAGENTA,
        label="vs NVDA hidden real",
    )
    _plot_curve(
        axis,
        donor_validation_result,
        color=TEAL,
        label="vs donor validation",
    )
    _plot_curve(
        axis,
        donor_train_result,
        color=BLUE,
        label="vs donor train",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=GRAY,
        linewidth=1.5,
        linestyle="--",
        label="Random (AUC = 0.500)",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    fig.suptitle(
        "DDPM — Real vs Synthetic ROC Diagnostics",
        fontsize=17,
        color=NAVY,
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "5-fold OOF classifier · synthetic = positive class",
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.025,
        "Lower AUC toward 0.5 = harder to distinguish from synthetic.",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.10, right=0.97)
    _save(fig, "ddpm_real_vs_synthetic_roc")

    validation_difference = abs(donor_validation_result.roc_auc - strict_final_auc)
    print(f"Wrote presentation ROC to {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print(
        "donor_train: "
        f"real={len(donor_train)}, synthetic={len(donor_train_synthetic)}, "
        f"auc={donor_train_result.roc_auc:.15f}, "
        f"window_end={donor_train_frame['window_end_date'].min()}.."
        f"{donor_train_frame['window_end_date'].max()}"
    )
    print(
        "donor_validation: "
        f"real={len(donor_validation)}, synthetic={len(donor_validation_synthetic)}, "
        f"auc={donor_validation_result.roc_auc:.15f}, "
        f"strict_final={strict_final_auc:.15f}, difference={validation_difference:.18g}"
    )
    print(
        "nvda_hidden: "
        f"source={NVDA_FULL_HISTORY.relative_to(REPO_ROOT)}, "
        f"real={len(nvda_hidden)}, synthetic={len(nvda_hidden_synthetic)}, "
        f"auc={nvda_hidden_result.roc_auc:.15f}, "
        f"window_start={hidden_frame['window_start_date'].min()}.."
        f"{hidden_frame['window_start_date'].max()}, "
        f"window_end={hidden_frame['window_end_date'].min()}.."
        f"{hidden_frame['window_end_date'].max()}"
    )
    print("nvda_future_test_used=NO")


def main() -> None:
    _configure_style()
    build_multi_reference_roc()


if __name__ == "__main__":
    main()
