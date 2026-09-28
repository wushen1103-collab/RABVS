from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def paired_permutation_pvalue(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return float("nan")
    obs = abs(diff.mean())
    if len(diff) <= 20:
        means = []
        for signs in product([-1.0, 1.0], repeat=len(diff)):
            means.append(abs((diff * np.asarray(signs)).mean()))
        return float((np.asarray(means) >= obs - 1e-12).mean())
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(100000, len(diff)))
    means = np.abs((signs * diff).mean(axis=1))
    return float((means >= obs - 1e-12).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baselines", default="greedy_surrogate,boba_ucb")
    parser.add_argument(
        "--metrics",
        default=(
            "AUBC,Hit@1000,hits_per_1000_inferences,"
            "PAINS_Brenk_free_ratio_at1000,unique_scaffold_count@1000,"
            "unseen_initial_scaffold_hit_rate_at1000,false_confidence_gap"
        ),
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    requested_metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    metrics = [metric for metric in requested_metrics if metric in summary.columns]
    methods = [m for m in sorted(summary["method"].unique()) if m not in baselines]
    key_cols = ["target", "seed"]

    rows = []
    for baseline in baselines:
        base = summary[summary["method"] == baseline][key_cols + metrics]
        for method in methods:
            cur = summary[summary["method"] == method][key_cols + metrics]
            merged = cur.merge(base, on=key_cols, suffixes=("_method", "_baseline"))
            for metric in metrics:
                diff = merged[f"{metric}_method"] - merged[f"{metric}_baseline"]
                rows.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "n_pairs": len(diff),
                        "method_mean": merged[f"{metric}_method"].mean(),
                        "baseline_mean": merged[f"{metric}_baseline"].mean(),
                        "mean_diff": diff.mean(),
                        "paired_permutation_p": paired_permutation_pvalue(diff.to_numpy()),
                    }
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
