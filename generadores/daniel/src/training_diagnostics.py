"""Training convergence plots and frozen-vs-long comparison helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .run_artifacts import read_history


def best_history_point(history: pd.DataFrame) -> tuple[int, float]:
    index = history["validation_loss"].idxmin()
    return int(history.loc[index, "epoch"]), float(history.loc[index, "validation_loss"])


def dynamic_zoom_bounds(
    history: pd.DataFrame, *, epochs_before: int = 10, epochs_after: int = 30
) -> tuple[int, int]:
    best_epoch, _ = best_history_point(history)
    first_epoch = int(history["epoch"].min())
    last_epoch = int(history["epoch"].max())
    return max(first_epoch, best_epoch - epochs_before), min(
        last_epoch, best_epoch + epochs_after
    )


def _draw_training_axis(
    axis,
    history: pd.DataFrame,
    *,
    run_id: str,
    best_epoch: int | None = None,
    best_loss: float | None = None,
    stopping_epoch: int | None = None,
) -> dict:
    observed_best_epoch, observed_best_loss = best_history_point(history)
    best_epoch = observed_best_epoch if best_epoch is None else int(best_epoch)
    best_loss = observed_best_loss if best_loss is None else float(best_loss)
    stopping_epoch = (
        int(history["epoch"].max()) if stopping_epoch is None else int(stopping_epoch)
    )
    axis.plot(history["epoch"], history["train_loss"], label="Train loss", linewidth=1.5)
    axis.plot(
        history["epoch"], history["validation_loss"], label="Validation loss", linewidth=1.5
    )
    axis.scatter([best_epoch], [best_loss], color="black", zorder=4, label="Best validation")
    axis.axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    axis.axvline(
        stopping_epoch,
        color="tab:red",
        linestyle=":",
        linewidth=1.3,
        label=f"Stopping epoch {stopping_epoch}",
    )
    axis.annotate(
        f"best={best_loss:.6f}\nepoch={best_epoch}",
        xy=(best_epoch, best_loss),
        xytext=(8, -38),
        textcoords="offset points",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "grey"},
    )
    # Training loss may continue down while validation loss rises. That pattern
    # is compatible with overfitting: early stopping therefore preserves the
    # best validation checkpoint, not the weights from the final epoch.
    axis.set_title(run_id)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Epsilon-prediction MSE")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)
    return {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stopping_epoch": stopping_epoch,
    }


def plot_training_views(
    history_path: Path | str,
    full_output: Path | str,
    zoom_output: Path | str,
    *,
    run_id: str,
) -> dict:
    history = read_history(history_path)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    result = _draw_training_axis(axis, history, run_id=f"{run_id} — full training")
    figure.tight_layout()
    full = Path(full_output)
    full.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(full, dpi=160)
    plt.close(figure)

    start, end = dynamic_zoom_bounds(history)
    zoomed = history.loc[history["epoch"].between(start, end)]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    _draw_training_axis(
        axis,
        zoomed,
        run_id=f"{run_id} — convergence zoom",
        best_epoch=result["best_epoch"],
        best_loss=result["best_validation_loss"],
        stopping_epoch=result["stopping_epoch"],
    )
    axis.set_xlim(start, end)
    figure.tight_layout()
    zoom = Path(zoom_output)
    zoom.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(zoom, dpi=160)
    plt.close(figure)
    return {**result, "zoom_start": start, "zoom_end": end}


def plot_frozen_vs_long(
    frozen_history_path: Path | str,
    long_history_path: Path | str,
    output: Path | str,
) -> dict:
    frozen = read_history(frozen_history_path)
    long = read_history(long_history_path)
    frozen_epoch, frozen_loss = best_history_point(frozen)
    long_epoch, long_loss = best_history_point(long)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(frozen["epoch"], frozen["validation_loss"], label="Frozen validation")
    axis.plot(long["epoch"], long["validation_loss"], label="Long diagnostic validation")
    axis.scatter([frozen_epoch], [frozen_loss], color="tab:blue", zorder=4)
    axis.scatter([long_epoch], [long_loss], color="tab:orange", zorder=4)
    axis.annotate(
        f"frozen min {frozen_loss:.6f} @ {frozen_epoch}",
        (frozen_epoch, frozen_loss),
        xytext=(8, -30),
        textcoords="offset points",
        fontsize=8,
    )
    axis.annotate(
        f"long min {long_loss:.6f} @ {long_epoch}",
        (long_epoch, long_loss),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Deterministic validation MSE")
    axis.set_title("Seed 42 frozen vs long-training diagnostic")
    axis.grid(alpha=0.22)
    axis.legend()
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return {
        "frozen_best_epoch": frozen_epoch,
        "frozen_best_validation_loss": frozen_loss,
        "long_best_epoch": long_epoch,
        "long_best_validation_loss": long_loss,
        "delta_best_validation_loss": long_loss - frozen_loss,
    }
