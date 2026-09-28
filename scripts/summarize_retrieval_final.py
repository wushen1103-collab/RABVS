from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - scipy exists in the experiment env
    wilcoxon = None

METRICS = [
    "AUBC",
    "Hit@100",
    "Hit@500",
    "Hit@1000",
    "Recall@1000",
    "EF_0.1pct",
    "EF_1pct",
    "PAINS_Brenk_free_ratio_at1000",
    "unique_scaffold_count@1000",
    "hits_per_1000_inferences",
]

DECISION_METRICS = [
    "AUBC",
    "Hit@1000",
    "PAINS_Brenk_free_ratio_at1000",
    "unique_scaffold_count@1000",
    "hits_per_1000_inferences",
]

EXPERIMENTS = [
    {
        "experiment": "fixed20_standard_finalretrieval0p5",
        "status": "promoted_multiseed",
        "budget": "standard",
        "dataset": "DUD-E fixed20",
        "runs": [
            "runs/dude_method_search_fixed20_finalretrieval0p5_std_seed0/summary.csv",
            "runs/dude_method_search_fixed20_finalretrieval0p5_std_seeds12/summary.csv",
        ],
        "original_baseline": "runs/dude_fixed20_revised_core/summary.csv",
        "adopted_method": "rabvs_budgeted_no_ucb",
        "all_methods": ["rabvs_budgeted_reliable", "rabvs_budgeted_no_ucb"],
        "baseline_method": "boba_ucb",
    },
    {
        "experiment": "fixed20_low_finalretrieval0p5",
        "status": "supplement_multiseed",
        "budget": "low",
        "dataset": "DUD-E fixed20",
        "runs": [
            "runs/dude_method_search_fixed20_finalretrieval0p5_low_seed0/summary.csv",
            "runs/dude_method_search_fixed20_finalretrieval0p5_low_seeds12/summary.csv",
        ],
        "original_baseline": "runs/dude_fixed20_low_budget_revised/summary.csv",
        "adopted_method": "rabvs_budgeted_reliable",
        "all_methods": ["rabvs_budgeted_reliable", "rabvs_budgeted_no_ucb"],
        "baseline_method": "boba_ucb",
    },
    {
        "experiment": "all102_standard_finalretrieval0p5",
        "status": "promoted_multiseed",
        "budget": "standard",
        "dataset": "DUD-E all102",
        "runs": [
            "runs/dude_method_search_all102_finalretrieval0p5_std_seed0/summary.csv",
            "runs/dude_method_search_all102_finalretrieval0p5_std_seeds12/summary.csv",
        ],
        "original_baseline": "runs/dude_all102_revised_core/summary.csv",
        "adopted_method": "rabvs_budgeted_no_ucb",
        "all_methods": ["rabvs_budgeted_reliable", "rabvs_budgeted_no_ucb"],
        "baseline_method": "boba_ucb",
    },
    {
        "experiment": "all102_low_finalretrieval0p5_seed0_stop",
        "status": "stopped_after_smoke",
        "budget": "low",
        "dataset": "DUD-E all102",
        "runs": ["runs/dude_method_search_all102_finalretrieval0p5_low_seed0/summary.csv"],
        "original_baseline": "runs/dude_all102_low_budget_revised/summary.csv",
        "adopted_method": "rabvs_budgeted_reliable",
        "all_methods": ["rabvs_budgeted_reliable", "rabvs_budgeted_no_ucb"],
        "baseline_method": "boba_ucb",
    },
    {
        "experiment": "all102_standard_finalretrieval0p25_seed0_sensitivity",
        "status": "sensitivity_seed0",
        "budget": "standard",
        "dataset": "DUD-E all102",
        "runs": ["runs/dude_method_search_all102_finalretrieval0p25_std_seed0/summary.csv"],
        "original_baseline": "runs/dude_all102_revised_core/summary.csv",
        "adopted_method": "rabvs_budgeted_no_ucb",
        "all_methods": ["rabvs_budgeted_reliable", "rabvs_budgeted_no_ucb"],
        "baseline_method": "boba_ucb",
    },
]


def read_existing(paths: list[str]) -> pd.DataFrame:
    frames = []
    missing = []
    for path in paths:
        p = Path(path)
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError("Missing result files: " + ", ".join(missing))
    if not frames:
        raise ValueError("No result frames loaded")
    return pd.concat(frames, ignore_index=True)


def wilcoxon_nonzero(diff: pd.Series) -> float:
    values = diff.astype(float).to_numpy()
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > 1e-12]
    if len(values) == 0 or wilcoxon is None:
        return math.nan
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def paired_summary(
    *,
    experiment: dict,
    new_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    method: str,
    baseline_method: str,
    comparison: str,
) -> tuple[list[dict], pd.DataFrame]:
    key_cols = ["seed", "target"]
    available = [m for m in METRICS if m in new_df.columns and m in baseline_df.columns]
    left = new_df[new_df["method"] == method][key_cols + available].copy()
    right = baseline_df[baseline_df["method"] == baseline_method][key_cols + available].copy()
    merged = left.merge(right, on=key_cols, suffixes=("_method", "_baseline"))
    rows = []
    for metric in available:
        diff = merged[f"{metric}_method"] - merged[f"{metric}_baseline"]
        rows.append(
            {
                "experiment": experiment["experiment"],
                "status": experiment["status"],
                "dataset": experiment["dataset"],
                "budget": experiment["budget"],
                "comparison": comparison,
                "method": method,
                "baseline": baseline_method,
                "metric": metric,
                "n_pairs": len(diff),
                "method_mean": merged[f"{metric}_method"].mean(),
                "baseline_mean": merged[f"{metric}_baseline"].mean(),
                "mean_delta": diff.mean(),
                "std_delta": diff.std(ddof=1),
                "median_delta": diff.median(),
                "min_delta": diff.min(),
                "max_delta": diff.max(),
                "negative_count": int((diff < -1e-12).sum()),
                "zero_count": int((np.abs(diff) <= 1e-12).sum()),
                "positive_count": int((diff > 1e-12).sum()),
                "wilcoxon_p_nonzero": wilcoxon_nonzero(diff),
            }
        )
    return rows, merged


def add_deltas(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    for metric in [m for m in DECISION_METRICS if f"{m}_method" in out and f"{m}_baseline" in out]:
        out[f"delta_{metric}"] = out[f"{metric}_method"] - out[f"{metric}_baseline"]
    return out


def make_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp in EXPERIMENTS:
        for comparison in ["vs_original_boba", "vs_finalretrieval_boba"]:
            subset = summary[
                (summary["experiment"] == exp["experiment"])
                & (summary["comparison"] == comparison)
                & (summary["method"] == exp["adopted_method"])
            ]
            if subset.empty:
                continue
            row = {
                "experiment": exp["experiment"],
                "status": exp["status"],
                "dataset": exp["dataset"],
                "budget": exp["budget"],
                "comparison": comparison,
                "adopted_method": exp["adopted_method"],
            }
            for metric in DECISION_METRICS:
                metric_row = subset[subset["metric"] == metric]
                if metric_row.empty:
                    continue
                r = metric_row.iloc[0]
                prefix = metric.replace("@", "_at_").replace("/", "_").replace(".", "p")
                row[f"{prefix}_mean_delta"] = r["mean_delta"]
                row[f"{prefix}_median_delta"] = r["median_delta"]
                row[f"{prefix}_min_delta"] = r["min_delta"]
                row[f"{prefix}_negative_count"] = int(r["negative_count"])
                row[f"{prefix}_p"] = r["wilcoxon_p_nonzero"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    Path("tables").mkdir(exist_ok=True)
    all_rows = []
    weak_tables = []
    for exp in EXPERIMENTS:
        new_df = read_existing(exp["runs"])
        original_df = pd.read_csv(exp["original_baseline"])
        baseline_method = exp["baseline_method"]
        methods = [m for m in exp["all_methods"] if m in set(new_df["method"])]
        for method in methods:
            rows, merged = paired_summary(
                experiment=exp,
                new_df=new_df,
                baseline_df=original_df,
                method=method,
                baseline_method=baseline_method,
                comparison="vs_original_boba",
            )
            all_rows.extend(rows)
            if method == exp["adopted_method"]:
                weak = add_deltas(merged).sort_values(
                    ["delta_Hit@1000", "delta_AUBC"], ascending=[True, True]
                ).head(20)
                weak.insert(0, "comparison", "vs_original_boba")
                weak.insert(0, "experiment", exp["experiment"])
                weak_tables.append(weak)

            rows, merged = paired_summary(
                experiment=exp,
                new_df=new_df,
                baseline_df=new_df,
                method=method,
                baseline_method=baseline_method,
                comparison="vs_finalretrieval_boba",
            )
            all_rows.extend(rows)
            if method == exp["adopted_method"]:
                weak = add_deltas(merged).sort_values(
                    ["delta_Hit@1000", "delta_AUBC"], ascending=[True, True]
                ).head(20)
                weak.insert(0, "comparison", "vs_finalretrieval_boba")
                weak.insert(0, "experiment", exp["experiment"])
                weak_tables.append(weak)

        exp_summary = pd.DataFrame([r for r in all_rows if r["experiment"] == exp["experiment"]])
        exp_summary.to_csv(f"tables/retrieval_final_{exp['experiment']}_summary.csv", index=False)

    summary = pd.DataFrame(all_rows)
    summary.to_csv("tables/retrieval_final_comparison_summary.csv", index=False)
    make_decision_table(summary).to_csv("tables/retrieval_final_decision_table.csv", index=False)

    if weak_tables:
        weak_all = pd.concat(weak_tables, ignore_index=True)
        keep_cols = [
            "experiment",
            "comparison",
            "seed",
            "target",
            "delta_AUBC",
            "delta_Hit@1000",
            "delta_PAINS_Brenk_free_ratio_at1000",
            "delta_unique_scaffold_count@1000",
            "delta_hits_per_1000_inferences",
        ]
        weak_all[[c for c in keep_cols if c in weak_all.columns]].to_csv(
            "tables/retrieval_final_weakcases.csv", index=False
        )

    # Back-compatible names used during the first interactive analysis.
    all102 = summary[summary["experiment"] == "all102_standard_finalretrieval0p5"]
    all102.to_csv("tables/retrieval_final_all102_multiseed_summary.csv", index=False)
    fixed20 = summary[summary["experiment"].str.startswith("fixed20_")]
    fixed20.to_csv("tables/retrieval_final_fixed20_multiseed_summary.csv", index=False)
    print("tables/retrieval_final_comparison_summary.csv")
    print("tables/retrieval_final_decision_table.csv")
    print("tables/retrieval_final_weakcases.csv")


if __name__ == "__main__":
    main()
