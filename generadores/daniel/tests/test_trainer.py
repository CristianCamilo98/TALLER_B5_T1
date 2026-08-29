import torch
from torch.utils.data import DataLoader

from generadores.daniel.src.diffusion import GaussianDiffusion
from generadores.daniel.src.network import TemporalDenoiser
from generadores.daniel.src.trainer import DiffusionTrainer, TrainerConfig


def test_trainer_runs_validation_and_writes_best_and_last(tmp_path) -> None:
    torch.manual_seed(3)
    model = TemporalDenoiser(base_channels=8, time_embedding_dim=16)
    diffusion = GaussianDiffusion(model, steps=4)
    config = TrainerConfig(
        learning_rate=0.001,
        max_epochs=2,
        early_stopping_patience=2,
        seed=42,
        validation_seed=424242,
        device="cpu",
    )
    trainer = DiffusionTrainer(
        diffusion,
        config,
        checkpoint_directory=tmp_path,
        checkpoint_metadata={
            "effective_config": {"test": True},
            "seed": 42,
            "channel_order": ["log_return", "log_high_low_range", "log1p_volume"],
            "window_shape": [65, 3],
        },
    )
    train_loader = DataLoader(torch.randn((8, 65, 3)), batch_size=4, shuffle=False)
    validation_loader = DataLoader(torch.randn((4, 65, 3)), batch_size=4, shuffle=False)
    result = trainer.fit(train_loader, validation_loader)
    assert len(result.history) == 2
    assert all(torch.isfinite(torch.tensor(row["train_loss"])) for row in result.history)
    assert all(
        torch.isfinite(torch.tensor(row["validation_loss"])) for row in result.history
    )
    best_path = tmp_path / "best_model.pt"
    last_path = tmp_path / "last_model.pt"
    assert best_path.is_file() and last_path.is_file()
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    for payload in (best, last):
        assert "model_state_dict" in payload
        assert "optimizer_state_dict" in payload
        assert "epoch" in payload
        assert "validation_loss" in payload
        assert payload["seed"] == 42
        assert payload["window_shape"] == [65, 3]


def test_validation_loss_is_repeatable_and_cannot_update_model() -> None:
    torch.manual_seed(9)
    model = TemporalDenoiser(base_channels=8, time_embedding_dim=16)
    diffusion = GaussianDiffusion(model, steps=10)
    trainer = DiffusionTrainer(
        diffusion,
        TrainerConfig(seed=42, validation_seed=424242, device="cpu"),
    )
    validation_loader = DataLoader(
        torch.randn((12, 65, 3)), batch_size=4, shuffle=False
    )
    before = {key: value.clone() for key, value in model.state_dict().items()}
    first = trainer._run_loader(validation_loader, training=False)
    second = trainer._run_loader(validation_loader, training=False)
    after = model.state_dict()
    assert first == second
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert all(parameter.grad is None for parameter in model.parameters())
