from __future__ import annotations

from common_protocol import load_experiment_config


def test_frozen_protocol_contract() -> None:
    config, _ = load_experiment_config()
    assert config["universe"]["target"] == "NVDA"
    assert config["universe"]["donors"] == [
        "AMD", "INTC", "QCOM", "AVGO", "MU", "TXN", "ADI", "MCHP", "MRVL", "NXPI"
    ]
    assert config["features"]["channels"] == [
        "log_return", "log_high_low_range", "log1p_volume"
    ]
    assert config["windows"] == {
        "length": 65,
        "context": 60,
        "horizon": 5,
        "strides": {
            "donor_train": 5,
            "donor_validation": 5,
            "target_visible": 1,
            "target_full_history": 1,
            "target_test": 5,
        },
    }
    assert config["synthetic_experiment"]["seeds"] == [42, 123, 2026]
    assert config["synthetic_experiment"]["ratios"] == [0.25, 0.5, 0.75]
    assert config["downstream"]["model"] == "ridge"
    assert config["downstream"]["alpha"] == 1.0
    assert config["downstream"]["status"] == "contract_only_not_implemented"

