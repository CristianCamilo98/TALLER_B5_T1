"""
Interpretation table (Part 13) -- descriptiva, POST-evaluacion.
No se reutiliza para re-tunear el protocolo con el "mejor ratio".
"""
from pathlib import Path
import pandas as pd

TABLES_DIR = Path("common_pipeline/03_utility/results/tables")


def main():
    summary = pd.read_csv(TABLES_DIR / "downstream_results_summary.csv")
    validity = pd.read_csv(TABLES_DIR / "physical_validity.csv").set_index("method")

    rows = []
    for method in sorted(summary["method"].unique()):
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
    out_path = TABLES_DIR / "interpretation_summary.csv"
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
