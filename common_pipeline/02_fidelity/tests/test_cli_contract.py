from __future__ import annotations

from evaluate_fidelity import build_parser


def test_cli_uses_certified_registry_and_is_strict_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.registry_path.name == "certified_outputs.json"
    assert args.allow_partial is False
    assert args.evaluation_subset_seed == 42
