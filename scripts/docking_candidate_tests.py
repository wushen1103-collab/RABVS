from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def wilcoxon_nonzero(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values != 0]
    if len(values) == 0:
        return float("nan")
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", default="boba_ucb")
    args = parser.parse_args()

    delta = pd.read_csv(args.delta)
    rows = []
    metrics = [
        ("delta_risk_alert_rate", "lower"),
        ("delta_mean_vina_score", "lower"),
        ("delta_best_vina_score", "lower"),
    ]
    for method, group in delta[delta["method"] != args.baseline].groupby("method", sort=True):
        for metric, direction in metrics:
            values = group[metric].to_numpy(dtype=float)
            if direction == "lower":
                improved_or_nonworse = int((values <= 0).sum())
                strictly_improved = int((values < 0).sum())
            else:
                improved_or_nonworse = int((values >= 0).sum())
                strictly_improved = int((values > 0).sum())
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_targets": int(len(values)),
                    "mean_delta": float(values.mean()),
                    "median_delta": float(np.median(values)),
                    "strictly_improved_targets": strictly_improved,
                    "improved_or_nonworse_targets": improved_or_nonworse,
                    "wilcoxon_p_nonzero": wilcoxon_nonzero(values),
                    "sign_test_p": float(binomtest(improved_or_nonworse, len(values), 0.5).pvalue),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
