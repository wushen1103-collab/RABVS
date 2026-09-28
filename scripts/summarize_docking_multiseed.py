from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


SUMMARY_METRICS = [
    "active_rate",
    "risk_alert_rate",
    "mean_mu",
    "mean_vina_score",
    "median_vina_score",
    "best_vina_score",
]


def load_inputs(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        seed_text, path_text = spec.split(":", 1)
        frame = pd.read_csv(path_text)
        frame.insert(0, "seed", int(seed_text))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["seed", "target", "method"]
    if combined.duplicated(keys).any():
        duplicates = combined.loc[combined.duplicated(keys, keep=False), keys]
        raise ValueError(f"Duplicate seed-target-method rows:\n{duplicates}")
    return combined.sort_values(keys).reset_index(drop=True)


def load_docked_inputs(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        seed_text, path_text = spec.split(":", 1)
        frame = pd.read_csv(path_text)
        frame["seed"] = int(seed_text)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def method_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in raw.groupby("method", sort=True):
        row = {
            "method": method,
            "n_pairs": int(len(group)),
            "n_targets": int(group["target"].nunique()),
            "n_seeds": int(group["seed"].nunique()),
            "n_docked_ok_sum": int(group["n_docked_ok"].sum()),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def target_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, method), group in raw.groupby(["target", "method"], sort=True):
        row = {
            "target": target,
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def delta_table(raw: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for (seed, target), group in raw.groupby(["seed", "target"], sort=True):
        base_rows = group[group["method"] == baseline]
        if len(base_rows) != 1:
            raise ValueError(f"Expected one {baseline} row for seed={seed}, target={target}")
        base = base_rows.iloc[0]
        for row in group.itertuples(index=False):
            result = {
                "seed": int(seed),
                "target": target,
                "method": row.method,
                "baseline": baseline,
                "n_candidates": int(row.n_candidates),
                "n_docked_ok": int(row.n_docked_ok),
            }
            for metric in SUMMARY_METRICS:
                result[f"delta_{metric}"] = float(getattr(row, metric) - base[metric])
            rows.append(result)
    return pd.DataFrame(rows)


def delta_summary_table(delta: pd.DataFrame, baseline: str) -> pd.DataFrame:
    metrics = [f"delta_{metric}" for metric in SUMMARY_METRICS]
    rows = []
    for method, group in delta[delta["method"] != baseline].groupby("method", sort=True):
        row = {
            "method": method,
            "baseline": baseline,
            "n_pairs": int(len(group)),
            "n_targets": int(group["target"].nunique()),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.nanmean(values))
            row[f"std_{metric}"] = float(np.nanstd(values, ddof=1))
            row[f"median_{metric}"] = float(np.nanmedian(values))
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_nonzero(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    values = values[values != 0]
    if len(values) == 0:
        return float("nan")
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def test_table(delta: pd.DataFrame, baseline: str) -> pd.DataFrame:
    metrics = {
        "delta_active_rate": "higher",
        "delta_risk_alert_rate": "lower",
        "delta_mean_vina_score": "lower",
        "delta_best_vina_score": "lower",
    }
    rows = []
    for method, group in delta[delta["method"] != baseline].groupby("method", sort=True):
        for metric, direction in metrics.items():
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            oriented = values if direction == "higher" else -values
            improved = int((oriented > 0).sum())
            tied = int((oriented == 0).sum())
            worsened = int((oriented < 0).sum())
            non_tied = improved + worsened
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "preferred_direction": direction,
                    "n_pairs": int(len(values)),
                    "mean_delta": float(values.mean()),
                    "std_delta": float(values.std(ddof=1)),
                    "median_delta": float(np.median(values)),
                    "strictly_improved_pairs": improved,
                    "tied_pairs": tied,
                    "worsened_pairs": worsened,
                    "wilcoxon_p_nonzero": wilcoxon_nonzero(values),
                    "exact_sign_p_non_tied": (
                        float(binomtest(improved, non_tied, 0.5).pvalue)
                        if non_tied
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def pairwise_overlap_tables(docked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (seed, target), group in docked.groupby(["seed", "target"], sort=True):
        methods = sorted(group["method"].unique())
        sets = {
            method: set(group.loc[group["method"] == method, "smiles"].astype(str))
            for method in methods
        }
        for method_a, method_b in combinations(methods, 2):
            set_a, set_b = sets[method_a], sets[method_b]
            overlap = len(set_a & set_b)
            union = len(set_a | set_b)
            rows.append(
                {
                    "seed": int(seed),
                    "target": target,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_a": len(set_a),
                    "n_b": len(set_b),
                    "n_overlap": overlap,
                    "overlap_rate_min_set": float(overlap / max(1, min(len(set_a), len(set_b)))),
                    "jaccard": float(overlap / max(1, union)),
                    "identical_sets": bool(set_a == set_b),
                }
            )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for (method_a, method_b), group in raw.groupby(["method_a", "method_b"], sort=True):
        summary_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_pairs": int(len(group)),
                "mean_overlap_rate_min_set": float(group["overlap_rate_min_set"].mean()),
                "std_overlap_rate_min_set": float(group["overlap_rate_min_set"].std(ddof=1)),
                "mean_jaccard": float(group["jaccard"].mean()),
                "std_jaccard": float(group["jaccard"].std(ddof=1)),
                "identical_set_pairs": int(group["identical_sets"].sum()),
            }
        )
    return raw, pd.DataFrame(summary_rows)


def write_csv(frame: pd.DataFrame, path_text: str) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="SEED:summary.csv")
    parser.add_argument("--baseline", default="boba_ucb")
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--method-out", required=True)
    parser.add_argument("--target-out", required=True)
    parser.add_argument("--delta-out", required=True)
    parser.add_argument("--delta-summary-out", required=True)
    parser.add_argument("--tests-out", required=True)
    parser.add_argument("--docked-input", action="append", help="SEED:docked.csv")
    parser.add_argument("--pairwise-out")
    parser.add_argument("--pairwise-summary-out")
    args = parser.parse_args()

    raw = load_inputs(args.input)
    delta = delta_table(raw, args.baseline)
    write_csv(raw, args.raw_out)
    write_csv(method_table(raw), args.method_out)
    write_csv(target_table(raw), args.target_out)
    write_csv(delta, args.delta_out)
    write_csv(delta_summary_table(delta, args.baseline), args.delta_summary_out)
    write_csv(test_table(delta, args.baseline), args.tests_out)
    if args.docked_input:
        if not args.pairwise_out or not args.pairwise_summary_out:
            parser.error("--docked-input requires both pairwise output paths")
        pairwise, pairwise_summary = pairwise_overlap_tables(load_docked_inputs(args.docked_input))
        write_csv(pairwise, args.pairwise_out)
        write_csv(pairwise_summary, args.pairwise_summary_out)


if __name__ == "__main__":
    main()
