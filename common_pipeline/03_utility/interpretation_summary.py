"""
Interpretation table (Part 13) -- descriptiva, POST-evaluacion.
No se reutiliza para re-tunear el protocolo con el "mejor ratio".
"""
from pathlib import Path
import argparse
import importlib
import pandas as pd

utility_run = importlib.import_module("common_pipeline.03_utility.utility_run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=utility_run.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--results-root", type=Path, default=utility_run.RESULTS_ROOT)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    run_dir = utility_run.existing_run_dir(
        results_root=args.results_root,
        registry_path=args.registry_path,
        run_id=args.run_id,
    )
    tables_dir = run_dir / "tables"
    summary = pd.read_csv(tables_dir / "downstream_results_summary.csv")
    validity = pd.read_csv(tables_dir / "physical_validity.csv").set_index("method")

    rows = []
    for method in sorted(summary["method"].unique()):
        if method == "REAL_ONLY":
            continue
        sub = summary[(summary["method"] == method) & (summary["ratio"] > 0)]
        best = sub.loc[sub["mean_rmse"].idxmin()]
        rows.append({
            "method": method,
            "best_observed_ratio": best["ratio"],
            "best_mean_rmse": best["mean_rmse"],
            "rmse_delta_vs_real": best["delta_rmse_pct_vs_real_only"],
            "best_mean_mae": best["mean_mae"],
            "mae_delta_vs_real": best["delta_mae_pct_vs_real_only"],
            "invalid_rate": validity.loc[method, "invalid_rate"],
        })
    result = pd.DataFrame(rows)
    out_path = tables_dir / "interpretation_summary.csv"
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
