from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_MODULE = REPO_ROOT / "common_pipeline" / "01_contract"
SCRIPT_PATH = REPO_ROOT / "generadores" / "david" / "scripts" / "generate_normalized.py"
EXPERIMENT_SCRIPT_PATH = REPO_ROOT / "generadores" / "david" / "scripts" / "experiment_normalized.py"
PLOT_SCRIPT_PATH = REPO_ROOT / "generadores" / "david" / "scripts" / "plot_experiment_diagnostics.py"
FLOW_MODULE_PATH = REPO_ROOT / "generadores" / "david" / "src" / "normalizing_flow.py"
if str(CONTRACT_MODULE) not in sys.path:
    sys.path.insert(0, str(CONTRACT_MODULE))

import baseline  # noqa: E402
from constants import (  # noqa: E402
    BASELINE_NOISE_SCALE,
    BASELINE_SEED,
    BASELINE_SOURCE_MODEL,
    CHANNEL_ORDER,
    EXPECTED_ROWS,
    GLOBAL_NORMALIZED_SPACE,
    N_CHANNELS,
    WINDOW_LENGTH,
)
from discovery import DiscoveredOutput  # noqa: E402
from io_utils import sha256_file, stack_features  # noqa: E402
from validation import validate_output  # noqa: E402


def load_generate_module():
    spec = importlib.util.spec_from_file_location("david_generate_normalized", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_experiment_module():
    spec = importlib.util.spec_from_file_location("david_experiment_normalized", EXPERIMENT_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_plot_module():
    spec = importlib.util.spec_from_file_location("david_plot_experiment_diagnostics", PLOT_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_flow_module():
    spec = importlib.util.spec_from_file_location("david_normalizing_flow", FLOW_MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_output_path_is_david_official_location():
    module = load_generate_module()
    assert module.DEFAULT_OUTPUT_PATH == (
        REPO_ROOT
        / "generadores"
        / "david"
        / "outputs"
        / "normalizing_flow_seed42_normalized.parquet"
    )


def test_default_generator_is_normalizing_flow():
    module = load_generate_module()

    assert module.OFFICIAL_SOURCE_MODEL == "normalizing_flow"
    assert module.DEFAULT_CHECKPOINT.name == "normalizing_flow_seed42.npz"


def test_real_nvp_inverse_and_log_likelihood_are_well_defined():
    flow = load_flow_module()
    rng = np.random.default_rng(123)
    model = flow.RealNVP(
        flow.FlowConfig(hidden_dims=(8,), n_coupling_layers=2, init_scale=0.02, seed=123)
    )
    values = rng.normal(size=(4, WINDOW_LENGTH, N_CHANNELS))
    flat = flow.flatten_windows(values)

    latent, forward_log_det = model.forward(flat)
    recovered, inverse_log_det = model.inverse(latent)

    np.testing.assert_allclose(recovered, flat, atol=1e-8)
    np.testing.assert_allclose(forward_log_det + inverse_log_det, 0.0, atol=1e-8)
    assert np.isfinite(model.log_prob(flat)).all()


def test_real_nvp_uses_actnorm_and_fixed_permutations():
    flow = load_flow_module()
    config = flow.FlowConfig(input_dim=6, hidden_dims=(7,), n_coupling_layers=3, seed=99)
    model = flow.RealNVP(config)

    assert config.use_actnorm
    assert config.use_permutations
    assert len(model.actnorms) == 3
    assert len(model.permutations) == 2
    assert "actnorms.0.log_scale" in model.parameters()
    assert not np.array_equal(model.permutations[0], np.arange(config.input_dim))


def test_real_nvp_manual_gradient_matches_finite_difference():
    flow = load_flow_module()
    rng = np.random.default_rng(456)
    model = flow.RealNVP(
        flow.FlowConfig(
            input_dim=4,
            hidden_dims=(5,),
            n_coupling_layers=2,
            init_scale=0.03,
            seed=456,
        )
    )
    values = rng.normal(size=(3, 4))
    _, grads = model.loss_and_gradients(values)
    params = model.parameters()
    name = "layers.0.net.W0"
    index = (0, 0)
    original = float(params[name][index])
    eps = 1.0e-5

    params[name][index] = original + eps
    plus = model.negative_log_likelihood(values)
    params[name][index] = original - eps
    minus = model.negative_log_likelihood(values)
    params[name][index] = original

    numeric = (plus - minus) / (2.0 * eps)
    assert grads[name][index] == pytest.approx(numeric, abs=1.0e-5)


def test_real_nvp_loads_legacy_checkpoints_without_new_transforms():
    flow = load_flow_module()
    payload = {
        "input_dim": 4,
        "hidden_dims": [5],
        "n_coupling_layers": 2,
        "scale_bound": 1.5,
        "init_scale": 0.03,
        "seed": 456,
    }
    config = flow.FlowConfig.from_dict(payload)

    assert not config.use_actnorm
    assert not config.use_permutations


def test_real_nvp_training_checkpoint_and_contract(tmp_path: Path):
    flow = load_flow_module()
    generate = load_generate_module()
    rng = np.random.default_rng(321)
    windows = rng.normal(size=(16, WINDOW_LENGTH, N_CHANNELS)).astype(np.float32)
    flow_config = flow.FlowConfig(hidden_dims=(8,), n_coupling_layers=2, init_scale=0.02, seed=321)
    training_config = flow.TrainingConfig(
        epochs=2,
        batch_size=4,
        learning_rate=1.0e-3,
        validation_fraction=0.25,
        patience=2,
        seed=321,
    )

    model, history = flow.train_real_nvp(
        windows,
        flow_config=flow_config,
        training_config=training_config,
    )
    assert len(history) == 2
    assert np.isfinite(history[-1]["validation_nll"])

    checkpoint = tmp_path / "normalizing_flow_seed42.npz"
    flow.save_checkpoint(
        checkpoint,
        model,
        metadata={
            "source_model": "normalizing_flow",
            "training_seed": 42,
            "architecture": {
                "name": "RealNVP",
                "uses_log_det_jacobian": True,
                "flow_config": flow_config.to_dict(),
            },
            "training": {
                "objective": "negative_log_likelihood",
                "optimizer": "Adam",
                "training_config": training_config.to_dict(),
            },
        },
        history=history,
    )
    loaded, metadata, loaded_history = flow.load_checkpoint(checkpoint)
    sampled = flow.sample_windows(loaded, n_windows=4, seed=42)
    frame = generate.make_canonical_frame(sampled)
    output_path = tmp_path / "david.parquet"
    frame.to_parquet(output_path, index=False)

    assert metadata["source_model"] == "normalizing_flow"
    assert loaded_history == history
    assert frame["source_model"].eq("normalizing_flow").all()
    discovered = DiscoveredOutput(
        generator_id="david",
        path=output_path,
        filename=output_path.name,
        rows=len(frame),
        sha256=sha256_file(output_path),
        columns=tuple(frame.columns),
    )
    report = validate_output(discovered, frame, expected_rows=4)
    assert report.contract_status == "PASS"


def test_david_baseline_is_reproducible_with_seed42(tmp_path: Path):
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"

    first = baseline.build_baseline(output_path=first_path, seed=BASELINE_SEED)
    second = baseline.build_baseline(output_path=second_path, seed=BASELINE_SEED)

    assert first.sha256 == second.sha256


def test_david_baseline_changes_when_seed_changes(tmp_path: Path):
    seed42_path = tmp_path / "seed42.parquet"
    seed123_path = tmp_path / "seed123.parquet"

    seed42 = baseline.build_baseline(output_path=seed42_path, seed=42)
    seed123 = baseline.build_baseline(output_path=seed123_path, seed=123)

    assert seed42.sha256 != seed123.sha256


def test_david_baseline_contract(tmp_path: Path):
    output_path = tmp_path / "david.parquet"
    result = baseline.build_baseline(
        output_path=output_path,
        seed=BASELINE_SEED,
        noise_scale=BASELINE_NOISE_SCALE,
        n_windows=EXPECTED_ROWS,
    )
    frame = pd.read_parquet(output_path)

    assert result.shape == (EXPECTED_ROWS, WINDOW_LENGTH, N_CHANNELS)
    assert list(frame.columns) == [
        "synthetic_id",
        "source_model",
        "training_seed",
        "space",
        "window_length",
        "n_channels",
        "channel_order",
        "features_flat",
    ]
    assert frame["source_model"].eq(BASELINE_SOURCE_MODEL).all()
    assert frame["training_seed"].eq(BASELINE_SEED).all()
    assert frame["space"].eq(GLOBAL_NORMALIZED_SPACE).all()
    assert frame["window_length"].eq(WINDOW_LENGTH).all()
    assert frame["n_channels"].eq(N_CHANNELS).all()
    assert frame["synthetic_id"].tolist() == list(range(EXPECTED_ROWS))
    assert all(tuple(value) == CHANNEL_ORDER for value in frame["channel_order"])

    tensor = stack_features(frame)
    assert tensor.shape == (EXPECTED_ROWS, WINDOW_LENGTH, N_CHANNELS)
    assert np.isfinite(tensor).all()

    discovered = DiscoveredOutput(
        generator_id="david",
        path=output_path,
        filename=output_path.name,
        rows=len(frame),
        sha256=sha256_file(output_path),
        columns=tuple(frame.columns),
    )
    report = validate_output(discovered, frame)
    assert report.contract_status == "PASS"


def test_experimental_candidates_stay_outside_official_outputs():
    module = load_experiment_module()
    official_outputs_dir = REPO_ROOT / "generadores" / "david" / "outputs"

    assert official_outputs_dir not in module.EXPERIMENT_OUTPUTS_DIR.parents
    assert module.OFFICIAL_OUTPUT.parent == official_outputs_dir


def test_experimental_candidate_frame_uses_canonical_contract():
    module = load_experiment_module()
    windows = np.zeros((2, WINDOW_LENGTH, N_CHANNELS), dtype=np.float32)

    frame = module.make_canonical_frame(windows, source_model="unit_test_candidate")

    assert list(frame.columns) == [
        "synthetic_id",
        "source_model",
        "training_seed",
        "space",
        "window_length",
        "n_channels",
        "channel_order",
        "features_flat",
    ]
    assert frame["source_model"].eq("unit_test_candidate").all()
    assert frame["training_seed"].eq(BASELINE_SEED).all()
    assert frame["space"].eq(GLOBAL_NORMALIZED_SPACE).all()
    assert all(tuple(value) == CHANNEL_ORDER for value in frame["channel_order"])


def test_diagnostic_plot_parser_extracts_noise_and_rho():
    module = load_plot_module()

    assert module.parse_noise_and_rho("temporal_jitter_0p40_rho0p85") == (0.40, 0.85)
    assert module.parse_noise_and_rho("bootstrap_jitter_0p15") == (None, None)
