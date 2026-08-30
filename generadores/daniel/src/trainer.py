"""Data-agnostic DDPM training loop and tiny-overfit diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .diffusion import GaussianDiffusion
from .reproducibility import set_seed


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    max_epochs: int = 200
    early_stopping_patience: int = 20
    seed: int = 42
    validation_seed: int = 424242
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict]
    best_epoch: int
    best_validation_loss: float
    stopping_reason: str
    total_seconds: float


def _extract_tensor(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]
    raise TypeError("DataLoader batches must be tensors or begin with a tensor")


def _gradients_are_finite(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )


class DiffusionTrainer:
    """Minimal trainer with AdamW, clipping, validation, and checkpoints."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        config: TrainerConfig,
        *,
        checkpoint_directory: Path | str | None = None,
        checkpoint_metadata: dict | None = None,
    ) -> None:
        self.diffusion = diffusion.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.diffusion.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.checkpoint_directory = (
            Path(checkpoint_directory) if checkpoint_directory is not None else None
        )
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        set_seed(config.seed)

    def _run_loader(self, loader, *, training: bool) -> float:
        self.diffusion.model.train(training)
        losses: list[float] = []
        validation_generator = None
        if not training:
            validation_generator = torch.Generator(device=self.config.device).manual_seed(
                self.config.validation_seed
            )
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for raw_batch in loader:
                batch = _extract_tensor(raw_batch).to(self.config.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                loss = self.diffusion.training_loss(
                    batch,
                    generator=validation_generator,
                )
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite diffusion loss")
                if training:
                    loss.backward()
                    if not _gradients_are_finite(self.diffusion.model):
                        raise RuntimeError("Non-finite gradients")
                    clip_grad_norm_(
                        self.diffusion.model.parameters(), self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                    if not all(
                        torch.isfinite(parameter).all().item()
                        for parameter in self.diffusion.model.parameters()
                    ):
                        raise RuntimeError("Non-finite model parameters after optimizer step")
                losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("DataLoader produced no batches")
        return float(np.mean(losses))

    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        history: list[dict],
        *,
        train_loss: float,
        validation_loss: float,
    ) -> None:
        if self.checkpoint_directory is None:
            return
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.diffusion.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "trainer_config": asdict(self.config),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "history": history,
                **self.checkpoint_metadata,
            },
            self.checkpoint_directory / name,
        )

    def fit(self, train_loader, validation_loader, *, epoch_callback=None) -> TrainingResult:
        history: list[dict] = []
        best_validation = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        stopping_reason = "max_epochs"
        training_start = perf_counter()
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = perf_counter()
            train_loss = self._run_loader(train_loader, training=True)
            validation_loss = self._run_loader(validation_loader, training=False)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "epoch_seconds": perf_counter() - epoch_start,
            }
            history.append(row)
            self._save_checkpoint(
                "last_model.pt",
                epoch,
                history,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(
                    "best_model.pt",
                    epoch,
                    history,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                )
            else:
                epochs_without_improvement += 1
            if epoch_callback is not None:
                epoch_callback(row, history)
            if epochs_without_improvement >= self.config.early_stopping_patience:
                stopping_reason = "early_stopping_patience"
                break
        return TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation,
            stopping_reason=stopping_reason,
            total_seconds=perf_counter() - training_start,
        )


def tiny_overfit_diagnostic(
    diffusion: GaussianDiffusion,
    batch: torch.Tensor,
    *,
    steps: int = 150,
    learning_rate: float = 0.001,
    seed: int = 42,
    gradient_clip_norm: float = 1.0,
    moving_average_window: int = 20,
) -> dict:
    """Overfit one fixed `(t, epsilon)` realization for implementation diagnosis."""

    if not 100 <= steps <= 300:
        raise ValueError("Tiny-overfit steps must be between 100 and 300")
    if batch.shape[0] != 32:
        raise ValueError("Tiny-overfit diagnostic requires exactly 32 windows")
    set_seed(seed)
    device = next(diffusion.model.parameters()).device
    batch = batch.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    fixed_timesteps = torch.randint(
        0, diffusion.steps, (batch.shape[0],), device=device, generator=generator
    )
    fixed_noise = torch.randn(
        batch.shape, dtype=batch.dtype, device=device, generator=generator
    )
    optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=learning_rate)
    losses: list[float] = []
    diffusion.model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.training_loss(
            batch, timesteps=fixed_timesteps, noise=fixed_noise
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError("Tiny-overfit loss became non-finite")
        loss.backward()
        if not _gradients_are_finite(diffusion.model):
            raise RuntimeError("Tiny-overfit gradients became non-finite")
        clip_grad_norm_(diffusion.model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    window = min(moving_average_window, steps // 2)
    initial = float(np.mean(losses[:window]))
    final = float(np.mean(losses[-window:]))
    reduction_percent = 100.0 * (initial - final) / initial
    return {
        "steps": steps,
        "moving_average_window": window,
        "initial_moving_average_loss": initial,
        "final_moving_average_loss": final,
        "reduction_percent": reduction_percent,
        "finite": bool(np.isfinite(losses).all()),
        "gradients_finite": _gradients_are_finite(diffusion.model),
    }
