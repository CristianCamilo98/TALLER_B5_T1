from __future__ import annotations

import pytest

import registry


def _entry(method_id: str, source_model: str, family: str = "neural_generator") -> dict:
    return {
        "method_id": method_id,
        "method_family": family,
        "source_model": source_model,
        "source_directory": f"generadores/{method_id}",
        "path": f"generadores/{method_id}/outputs/synthetic.parquet",
        "sha256": "a" * 64,
        "rows": 5000,
        "logical_shape": [5000, 65, 3],
        "training_seed": 42,
        "space": "global_channel_normalized",
        "window_length": 65,
        "n_channels": 3,
        "channel_order": ["log_return", "log_high_low_range", "log1p_volume"],
        "features_flat_length": 195,
        "normalization_status": "NORMALIZATION_NUMERICALLY_MATCHES",
        "contract_status": "PASS",
        "donor_lineage_status": "NOT_VERIFIABLE",
    }


def _payload(*, david_model: str | None = "normalizing_flow", extra: dict | None = None) -> dict:
    methods = [
        _entry("cristian", "wgan_gp"),
        _entry("daniel", "diffusion_ddpm"),
        _entry("marco", "marco_vae"),
    ]
    if david_model is not None:
        methods.append(_entry("david", david_model))
    methods.append(_entry("bootstrap_jitter", "bootstrap_jitter", "simple_baseline"))
    if extra is not None:
        methods.append(extra)
    return {
        "registry_version": registry.REGISTRY_VERSION,
        "expected_neural_methods": 4,
        "expected_simple_baselines": 1,
        "methods": methods,
    }


def test_strict_complete_official_roles_pass() -> None:
    selected, summary = registry.select_experiment_methods(_payload())
    assert set(method["method_id"] for method in selected["methods"]) == {
        "cristian", "daniel", "marco", "david", "bootstrap_jitter"
    }
    assert summary["strict_complete"] is True
    assert summary["run_mode"] == "STRICT_FINAL"


def test_strict_wrong_david_role_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="david:normalizing_flow"):
        registry.select_experiment_methods(_payload(david_model="temporal_jitter"))


def test_partial_wrong_david_selects_only_satisfied_roles_and_baseline() -> None:
    selected, summary = registry.select_experiment_methods(
        _payload(david_model="temporal_jitter"), allow_partial=True
    )
    assert set(method["method_id"] for method in selected["methods"]) == {
        "cristian", "daniel", "marco", "bootstrap_jitter"
    }
    assert summary["excluded_certified_methods"] == ["david"]
    assert summary["is_final_run"] is False


def test_partial_missing_david_still_selects_three_roles_and_baseline() -> None:
    selected, summary = registry.select_experiment_methods(
        _payload(david_model=None), allow_partial=True
    )
    assert len(selected["methods"]) == 4
    assert summary["official_generators_present"] == 3
    assert summary["missing_official_roles"] == ["david:normalizing_flow"]


def test_random_fifth_model_cannot_replace_normalizing_flow() -> None:
    random_model = _entry("new_generator_xyz", "normalizing_flow")
    with pytest.raises(RuntimeError, match="official_generators=3/4"):
        registry.select_experiment_methods(
            _payload(david_model="temporal_jitter", extra=random_model)
        )
