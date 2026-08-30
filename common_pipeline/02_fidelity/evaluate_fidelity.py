"""CLI for the unified synthetic-fidelity evaluation.

Example from the repository root::

    python common_pipeline/02_fidelity/evaluate_fidelity.py

The default contract expects four neural generators. During parallel
development, ``--expected-neural-count`` can reflect the number already merged;
the evaluation mathematics does not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from fidelity_core import (  # noqa: E402
    CHANNEL_ORDER,
    DONOR_TRAIN_COUNT,
    DONOR_VALIDATION_COUNT,
    EVALUATION_SUBSET_SEED,
    EVALUATION_SUBSET_SIZE,
    SYNTHETIC_POOL_COUNT,
    acf_table,
    apply_common_subset,
    c2st_table,
    common_subset_indices,
    correlation_tables,
    discover_neural_outputs,
    joint_tsne_coordinates,
    load_real_reference,
    load_synthetic_pool,
    marginal_statistics,
    nearest_neighbor_table,
    wasserstein_table,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def parse_synthetic_arguments(values: list[str] | None) -> dict[str, Path]:
    """Parse repeatable ``METHOD=PATH`` arguments without personal paths."""

    parsed: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--synthetic values must use METHOD=PATH")
        method, raw_path = value.split("=", 1)
        method = method.strip()
        if not method or method in parsed:
            raise ValueError(f"Invalid or duplicate synthetic method {method!r}")
        parsed[method] = Path(raw_path)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=MODULE_DIR.parents[1],
        help="Repository root (defaults to the root containing common_pipeline)",
    )
    parser.add_argument(
        "--synthetic",
        action="append",
        metavar="METHOD=PATH",
        help="Explicit normalized neural output; repeat once per method",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        help="Optional bootstrap-jitter normalized Parquet from 01_contract",
    )
    parser.add_argument(
        "--expected-neural-count",
        type=int,
        default=4,
        help="Required neural method count (four in the final protocol)",
    )
    parser.add_argument(
        "--evaluation-subset-seed",
        type=int,
        default=EVALUATION_SUBSET_SEED,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=MODULE_DIR / "results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = args.repo_root.resolve()
    results_dir = args.results_dir.resolve()
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    explicit = parse_synthetic_arguments(args.synthetic)
    neural_paths = explicit or discover_neural_outputs(repository_root)
    if len(neural_paths) != args.expected_neural_count:
        raise RuntimeError(
            f"Expected {args.expected_neural_count} neural normalized outputs, "
            f"found {len(neural_paths)}: {sorted(neural_paths)}"
        )

    donor_train_path = repository_root / "data/features/windows/donor_train.parquet"
    donor_validation_path = (
        repository_root / "data/features/windows/donor_validation.parquet"
    )
    donor_train, real_validation, statistics = load_real_reference(
        donor_train_path, donor_validation_path
    )
    if donor_train.shape[0] != DONOR_TRAIN_COUNT:
        raise RuntimeError("Unexpected donor_train count")
    if real_validation.shape[0] != DONOR_VALIDATION_COUNT:
        raise RuntimeError("Unexpected donor_validation count")

    pools = {
        method: load_synthetic_pool(
            path, method=method, expected_normalizer=statistics
        )
        for method, path in sorted(neural_paths.items())
    }
    baseline_method = None
    if args.baseline_path is not None:
        baseline_path = args.baseline_path
        if not baseline_path.is_absolute():
            baseline_path = repository_root / baseline_path
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Optional baseline does not exist: {baseline_path}")
        baseline_method = "bootstrap_jitter"
        pools[baseline_method] = load_synthetic_pool(
            baseline_path,
            method=baseline_method,
            expected_normalizer=statistics,
        )

    if any(pool.windows.shape[0] != SYNTHETIC_POOL_COUNT for pool in pools.values()):
        raise RuntimeError("Every common synthetic pool must contain exactly 5000 windows")

    # One RNG call creates one ordered list of row positions. Reusing this list
    # for every method removes subset-selection noise from model comparisons.
    indices = common_subset_indices(
        evaluation_subset_seed=args.evaluation_subset_seed
    )
    synthetic_subset = apply_common_subset(
        {method: pool.windows for method, pool in pools.items()}, indices
    )
    datasets = {"real": real_validation, **synthetic_subset}

    tables = {
        "marginal_statistics.csv": marginal_statistics(datasets),
        "wasserstein.csv": wasserstein_table(real_validation, synthetic_subset),
        "return_acf.csv": acf_table(
            real_validation, synthetic_subset, absolute=False
        ),
        "abs_return_acf.csv": acf_table(
            real_validation, synthetic_subset, absolute=True
        ),
        "nearest_neighbor.csv": nearest_neighbor_table(
            donor_train, real_validation, synthetic_subset
        ),
        "c2st_results.csv": c2st_table(real_validation, synthetic_subset),
    }
    correlation_values, correlation_errors = correlation_tables(
        real_validation, synthetic_subset
    )
    tables["channel_correlations.csv"] = correlation_values
    tables["correlation_errors.csv"] = correlation_errors
    tables["joint_tsne_coordinates.csv"] = joint_tsne_coordinates(
        real_validation, synthetic_subset
    )
    for filename, table in tables.items():
        table.to_csv(tables_dir / filename, index=False, float_format="%.12g")

    np.savetxt(
        tables_dir / "evaluation_subset_indices.csv",
        indices,
        fmt="%d",
        delimiter=",",
        header="row_position",
        comments="",
    )
    subset_digest = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
    manifest = {
        "protocol": "common_normalized_fidelity_v1",
        "real_reference": repository_relative(donor_validation_path, repository_root),
        "nearest_neighbor_reference": repository_relative(
            donor_train_path, repository_root
        ),
        "real_count": int(real_validation.shape[0]),
        "real_shape": list(real_validation.shape),
        "synthetic_pool_count": SYNTHETIC_POOL_COUNT,
        "evaluation_subset_size": EVALUATION_SUBSET_SIZE,
        "evaluation_subset_seed": args.evaluation_subset_seed,
        "evaluation_subset_indices_sha256": subset_digest,
        "channel_order": list(CHANNEL_ORDER),
        "space": "global_channel_normalized",
        "normalization": {
            "fit_split": "donor_train",
            "fit_axes": [0, 1],
            "fit_dtype": "float64",
            "output_dtype": "float32",
            "ddof": 0,
            "mean": statistics.mean.tolist(),
            "std": statistics.std.tolist(),
        },
        "synthetic_methods": {
            method: {
                "path": repository_relative(pool.path, repository_root),
                "sha256": sha256_file(pool.path),
                "training_seed": pool.training_seed,
                "space": pool.space,
                "channel_order": list(pool.channel_order),
                "metadata_evidence": pool.metadata_evidence,
            }
            for method, pool in pools.items()
        },
        "baseline_included": baseline_method is not None,
        "physical_sign_validation_applied": False,
        "synthetic_repair_applied": False,
        "tables": sorted(tables),
    }
    (results_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Evaluated {len(pools)} methods against {len(real_validation)} real windows; "
        f"tables written to {tables_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
