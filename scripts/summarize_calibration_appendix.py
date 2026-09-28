from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

METRICS = [
    "ECE",
    "Brier",
    "NLL",
    "calib_ECE_test",
    "calib_Brier_test",
    "calib_NLL_test",
    "conformal_coverage90",
    "conformal_residual_q90",
    "risk_coverage_auc",
    "false_confidence_gap",
]

LOWER_BETTER = {
    "ECE",
    "Brier",
    "NLL",
    "calib_ECE_test",
    "calib_Brier_test",
    "calib_NLL_test",
    "conformal_residual_q90",
    "risk_coverage_auc",
}


def wilcoxon_nonzero(values: pd.Series) -> float:
    arr = values.astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    arr = arr[np.abs(arr) > 1e-12]
    if len(arr) == 0 or wilcoxon is None:
        return float("nan")
    try:
        return float(wilcoxon(arr, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def mean_std(series: pd.Series) -> str:
    return f"{series.mean():.4f} +/- {series.std(ddof=1):.4f}"


def summarize_run(summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    available = [metric for metric in METRICS if metric in summary.columns]
    for method, group in summary.groupby("method"):
        row = {"dataset": dataset, "method": method, "n": len(group)}
        for metric in available:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
            row[f"{metric}_mean_std"] = mean_std(group[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method"])


def paired_deltas(summary: pd.DataFrame, dataset: str, baseline: str = "boba_ucb") -> pd.DataFrame:
    rows = []
    available = [metric for metric in METRICS if metric in summary.columns]
    base = summary[summary.method == baseline][["seed", "target"] + available]
    for method in sorted(m for m in summary.method.unique() if m != baseline):
        cur = summary[summary.method == method][["seed", "target"] + available]
        merged = cur.merge(base, on=["seed", "target"], suffixes=("_method", "_baseline"))
        for metric in available:
            raw_delta = merged[f"{metric}_method"] - merged[f"{metric}_baseline"]
            direction = -1.0 if metric in LOWER_BETTER else 1.0
            favorable_delta = direction * raw_delta
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "n_pairs": len(raw_delta),
                    "method_mean": merged[f"{metric}_method"].mean(),
                    "baseline_mean": merged[f"{metric}_baseline"].mean(),
                    "raw_mean_delta": raw_delta.mean(),
                    "favorable_mean_delta": favorable_delta.mean(),
                    "raw_median_delta": raw_delta.median(),
                    "raw_min_delta": raw_delta.min(),
                    "raw_max_delta": raw_delta.max(),
                    "favorable_positive_count": int((favorable_delta > 1e-12).sum()),
                    "favorable_negative_count": int((favorable_delta < -1e-12).sum()),
                    "wilcoxon_p_nonzero": wilcoxon_nonzero(raw_delta),
                    "lower_better": metric in LOWER_BETTER,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    inputs = [
        ("DUD-E fixed20 final retrieval", ["runs/dude_calibration_fixed20/summary.csv"]),
        ("DUD-E all102 final retrieval seed0", ["runs/dude_calibration_all102_finalretrieval0p5_seed0/summary.csv"]),
        ("DUD-E all102 final retrieval multiseed", [
            "runs/dude_calibration_all102_finalretrieval0p5_seed0/summary.csv",
            "runs/dude_calibration_all102_finalretrieval0p5_seeds12/summary.csv",
        ]),
        ("LIT-PCBA all15 exact metrics", ["runs/exact_metrics_lit_all15/summary.csv"]),
    ]
    summaries = []
    deltas = []
    for dataset, paths in inputs:
        frames = []
        for path in paths:
            path_obj = Path(path)
            if path_obj.exists():
                frames.append(pd.read_csv(path_obj))
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        summaries.append(summarize_run(frame, dataset))
        deltas.append(paired_deltas(frame, dataset))
    if not summaries:
        raise FileNotFoundError("No calibration summaries are available yet")
    Path("tables").mkdir(exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv("tables/calibration_appendix_summary.csv", index=False)
    pd.concat(deltas, ignore_index=True).to_csv("tables/calibration_appendix_paired_deltas.csv", index=False)
    print("tables/calibration_appendix_summary.csv")
    print("tables/calibration_appendix_paired_deltas.csv")


if __name__ == "__main__":
    main()
