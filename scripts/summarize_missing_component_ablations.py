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
    "ECE",
    "Brier",
    "NLL",
    "calib_ECE_test",
    "calib_Brier_test",
    "calib_NLL_test",
    "conformal_coverage90",
    "risk_coverage_auc",
    "PAINS_Brenk_free_ratio_at1000",
    "unique_scaffold_count@1000",
    "mean_pairwise_cosine_distance_at1000",
    "inference_saved_ratio",
    "hits_per_1000_inferences",
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


def main() -> None:
    ensemble5 = pd.read_csv("runs/dude_missing_ablation_fixed20_ensemble5/summary.csv")
    ensemble1 = pd.read_csv("runs/dude_missing_ablation_fixed20_ensemble1/summary.csv")
    main5 = ensemble5[ensemble5["method"] == "rabvs_budgeted_no_ucb"].copy()
    no_partition = ensemble5[
        ensemble5["method"] == "rabvs_no_partition_reliable"
    ].copy()
    main1 = ensemble1[ensemble1["method"] == "rabvs_budgeted_no_ucb"].copy()

    frames = []
    for label, frame in [
        ("main_ensemble5", main5),
        ("no_partition_ensemble5", no_partition),
        ("main_single_model", main1),
    ]:
        available = [metric for metric in METRICS if metric in frame]
        row = {
            "variant": label,
            "n_target_seed": len(frame),
        }
        for metric in available:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1)
        frames.append(row)

    paired = []
    paired.extend(
        paired_rows(no_partition, main5, "no_partition_minus_main_ensemble5")
    )
    paired.extend(paired_rows(main1, main5, "single_model_minus_ensemble5"))

    Path("tables").mkdir(exist_ok=True)
    pd.DataFrame(frames).to_csv(
        "tables/missing_component_ablation_method_summary.csv", index=False
    )
    pd.DataFrame(paired).to_csv(
        "tables/missing_component_ablation_paired_summary.csv", index=False
    )
    print("tables/missing_component_ablation_method_summary.csv")
    print("tables/missing_component_ablation_paired_summary.csv")


if __name__ == "__main__":
    main()
