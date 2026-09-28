from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def parse_comparisons(value: str) -> list[tuple[str, str]]:
    pairs = []
    for item in value.split(","):
        method, baseline = item.strip().split(":", 1)
        pairs.append((method, baseline))
    return pairs


def bootstrap_mean_delta(
    values: np.ndarray, baseline: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    value_samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    baseline_samples = rng.choice(baseline, size=(n_boot, len(baseline)), replace=True)
    deltas = value_samples.mean(axis=1) - baseline_samples.mean(axis=1)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docked", required=True)
    parser.add_argument(
        "--comparisons",
        default=(
            "rabvs_budgeted_reliable:random_decoy_control,"
            "random_active_control:random_decoy_control,"
            "rabvs_budgeted_reliable:random_active_control"
        ),
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docked = pd.read_csv(args.docked)
    docked = docked[docked["dock_status"] == "ok"].copy()
    rng = np.random.default_rng(20260707)
    rows = []
    for target, group in docked.groupby("target", sort=True):
        for method, baseline in parse_comparisons(args.comparisons):
            values = group.loc[group["method"] == method, "best_vina_score"].to_numpy(float)
            base = group.loc[group["method"] == baseline, "best_vina_score"].to_numpy(float)
            if len(values) == 0 or len(base) == 0:
                raise ValueError(f"Missing {method} or {baseline} for {target}")
            test = mannwhitneyu(values, base, alternative="less", method="auto")
            ci_low, ci_high = bootstrap_mean_delta(values, base, rng, args.n_boot)
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "baseline": baseline,
                    "n_method": int(len(values)),
                    "n_baseline": int(len(base)),
                    "mean_method": float(values.mean()),
                    "mean_baseline": float(base.mean()),
                    "mean_delta": float(values.mean() - base.mean()),
                    "mean_delta_ci95_low": ci_low,
                    "mean_delta_ci95_high": ci_high,
                    "median_delta": float(np.median(values) - np.median(base)),
                    "probability_method_scores_lower": float(
                        1.0 - test.statistic / (len(values) * len(base))
                    ),
                    "mannwhitney_p_one_sided_lower": float(test.pvalue),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
