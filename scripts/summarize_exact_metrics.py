from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = [
    "AUBC",
    "Hit@100",
    "Hit@500",
    "Hit@1000",
    "Recall@1000",
    "EF_0.1pct",
    "EF_1pct",
    "BEDROC_alpha20",
    "PAINS_free_ratio_at1000",
    "Brenk_free_ratio_at1000",
    "PAINS_Brenk_free_ratio_at1000",
    "unique_scaffold_count@1000",
    "mean_pairwise_tanimoto_at1000",
]


def pvalue_nonzero(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-12)]
    if len(values) == 0:
        return math.nan
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def paired_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    comparison: str,
) -> list[dict]:
    available = [metric for metric in METRICS if metric in left and metric in right]
    merged = left[["seed", "target", *available]].merge(
        right[["seed", "target", *available]],
        on=["seed", "target"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    rows = []
    for metric in available:
        delta = merged[f"{metric}_left"] - merged[f"{metric}_right"]
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "n_pairs": len(delta),
                "left_mean": merged[f"{metric}_left"].mean(),
                "right_mean": merged[f"{metric}_right"].mean(),
                "mean_delta": delta.mean(),
                "std_delta": delta.std(ddof=1),
                "median_delta": delta.median(),
                "min_delta": delta.min(),
                "max_delta": delta.max(),
                "negative_count": int((delta < -1e-12).sum()),
                "zero_count": int((np.abs(delta) <= 1e-12).sum()),
                "positive_count": int((delta > 1e-12).sum()),
                "wilcoxon_p_nonzero": pvalue_nonzero(delta.to_numpy()),
            }
        )
    return rows


def method_row(dataset: str, label: str, frame: pd.DataFrame) -> dict:
    row = {"dataset": dataset, "variant": label, "n_target_seed": len(frame)}
    for metric in [metric for metric in METRICS if metric in frame]:
        row[f"{metric}_mean"] = frame[metric].mean()
        row[f"{metric}_std"] = frame[metric].std(ddof=1)
    return row


def main() -> None:
    dude_fair = pd.read_csv("runs/exact_metrics_dude_all102_fair/summary.csv")
    dude_original = pd.read_csv(
        "runs/exact_metrics_dude_all102_original_boba/summary.csv"
    )
    lit = pd.read_csv("runs/exact_metrics_lit_all15/summary.csv")

    dude_main = dude_fair[dude_fair["method"] == "rabvs_budgeted_no_ucb"].copy()
    dude_fair_boba = dude_fair[dude_fair["method"] == "boba_ucb"].copy()
    dude_original_boba = dude_original[dude_original["method"] == "boba_ucb"].copy()
    lit_main = lit[lit["method"] == "rabvs_budgeted_reliable"].copy()
    lit_boba = lit[lit["method"] == "boba_ucb"].copy()

    method_rows = [
        method_row("DUD-E all102", "rabvs_budgeted_no_ucb", dude_main),
        method_row("DUD-E all102", "boba_final_retrieval", dude_fair_boba),
        method_row("DUD-E all102", "boba_original", dude_original_boba),
        method_row("LIT-PCBA all15", "rabvs_budgeted_reliable", lit_main),
        method_row("LIT-PCBA all15", "boba_ucb", lit_boba),
    ]
    paired = []
    paired.extend(
        paired_rows(dude_main, dude_fair_boba, "dude_main_minus_fair_boba")
    )
    paired.extend(
        paired_rows(dude_main, dude_original_boba, "dude_main_minus_original_boba")
    )
    paired.extend(paired_rows(lit_main, lit_boba, "lit_main_minus_boba"))

    Path("tables").mkdir(exist_ok=True)
    pd.DataFrame(method_rows).to_csv(
        "tables/exact_metrics_method_summary.csv", index=False
    )
    pd.DataFrame(paired).to_csv(
        "tables/exact_metrics_paired_summary.csv", index=False
    )
    print("tables/exact_metrics_method_summary.csv")
    print("tables/exact_metrics_paired_summary.csv")


if __name__ == "__main__":
    main()
