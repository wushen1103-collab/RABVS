from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "AUBC",
    "Hit@100",
    "Hit@500",
    "Hit@1000",
    "Recall@1000",
    "EF_0.1pct",
    "EF_1pct",
    "BEDROC_alpha20",
    "ECE",
    "Brier",
    "neural_inferences",
    "inference_saved_ratio",
    "hits_per_1000_inferences",
    "unique_scaffold_count@1000",
    "mean_pairwise_cosine_distance_at1000",
    "mean_pairwise_tanimoto_at1000",
    "PAINS_Brenk_free_ratio_at1000",
    "PAINS_free_ratio_at1000",
    "Brenk_free_ratio_at1000",
    "unseen_initial_scaffold_hit_rate_at1000",
    "high_score_high_unc_hit_rate",
    "high_score_low_unc_hit_rate",
    "false_confidence_gap",
]


def mean_std(series: pd.Series) -> str:
    return f"{series.mean():.4f} +/- {series.std(ddof=1):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--table-dir", default="tables")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(run_dir / "summary.csv")

    rows = []
    for method, group in summary.groupby("method"):
        row = {"method": method}
        for metric in [m for m in METRICS if m in group.columns]:
            row[metric] = mean_std(group[metric])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("method")
    table.to_csv(table_dir / "main_table.csv", index=False)
    table.to_csv(table_dir / "synthetic_smoke_main_table.csv", index=False)

    available_metrics = [m for m in METRICS if m in summary.columns]
    per_target = (
        summary.groupby(["target", "method"], as_index=False)[available_metrics]
        .mean(numeric_only=True)
        .sort_values(["target", "method"])
    )
    per_target.to_csv(table_dir / "per_target_table.csv", index=False)
    per_target.to_csv(table_dir / "synthetic_smoke_per_target.csv", index=False)

    print(table_dir / "main_table.csv")
    print(table_dir / "per_target_table.csv")


if __name__ == "__main__":
    main()
