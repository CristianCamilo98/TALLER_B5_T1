"""Render unsmoothed train/validation loss curves from a run history CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generadores.daniel.src.run_artifacts import read_history  # noqa: E402

RUN_ID = "diffusion_seed42_baseline"


def plot_training(history_path: Path, output_path: Path) -> dict:
    history = read_history(history_path)
    best_index = history["validation_loss"].idxmin()
    best_epoch = int(history.loc[best_index, "epoch"])
    best_loss = float(history.loc[best_index, "validation_loss"])

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(history["epoch"], history["train_loss"], label="Train loss", linewidth=1.5)
    axis.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation loss",
        linewidth=1.5,
    )
    axis.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Best epoch: {best_epoch}",
    )
    axis.scatter([best_epoch], [best_loss], color="black", s=30, zorder=3)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Epsilon-prediction MSE")
    axis.set_title("DDPM seed 42 baseline — training convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return {"best_epoch": best_epoch, "best_validation_loss": best_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        default=REPOSITORY_ROOT / f"generadores/daniel/artifacts/histories/{RUN_ID}.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / f"generadores/daniel/artifacts/figures/{RUN_ID}_loss.png",
    )
    args = parser.parse_args()
    result = plot_training(args.history.resolve(), args.output.resolve())
    print(f"figure={args.output.resolve()}")
    print(f"best_epoch={result['best_epoch']}")
    print(f"best_validation_loss={result['best_validation_loss']:.10f}")


if __name__ == "__main__":
    main()
