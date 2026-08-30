from __future__ import annotations

from evaluate_fidelity import build_parser, parse_synthetic_arguments


def test_cli_expects_four_neural_methods_and_no_baseline_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.expected_neural_count == 4
    assert args.baseline_path is None
    assert args.evaluation_subset_seed == 42


def test_explicit_synthetic_paths_are_method_keyed() -> None:
    parsed = parse_synthetic_arguments(
        ["vae=generadores/marco/outputs/vae.parquet", "ddpm=outputs/ddpm.parquet"]
    )
    assert list(parsed) == ["vae", "ddpm"]
    assert parsed["vae"].as_posix() == "generadores/marco/outputs/vae.parquet"
