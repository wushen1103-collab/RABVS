from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def delta_table(summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for target, group in summary.groupby("target", sort=True):
        base_rows = group[group["method"] == baseline]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "target": target,
                    "method": row.method,
                    "baseline": baseline,
                    "n_candidates": int(row.n_candidates),
                    "n_docked_ok": int(row.n_docked_ok),
                    "delta_active_rate": float(row.active_rate - base.active_rate),
                    "delta_risk_alert_rate": float(row.risk_alert_rate - base.risk_alert_rate),
                    "delta_mean_vina_score": float(row.mean_vina_score - base.mean_vina_score),
                    "delta_median_vina_score": float(row.median_vina_score - base.median_vina_score),
                    "delta_best_vina_score": float(row.best_vina_score - base.best_vina_score),
                }
            )
    return pd.DataFrame(rows)


def overlap_table(docked: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for target, group in docked.groupby("target", sort=True):
        base = set(group.loc[group["method"] == baseline, "smiles"].astype(str))
        for method, method_group in group.groupby("method", sort=True):
            current = set(method_group["smiles"].astype(str))
            overlap = len(current & base)
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "baseline": baseline,
                    "n_unique_candidates": len(current),
                    "overlap_with_baseline": overlap,
                    "overlap_rate": float(overlap / max(1, len(current))),
                }
            )
    return pd.DataFrame(rows)


def aggregate_delta(delta: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for method, group in delta[delta["method"] != baseline].groupby("method", sort=True):
        rows.append(
            {
                "method": method,
                "n_targets": int(group["target"].nunique()),
                "mean_delta_active_rate": float(group["delta_active_rate"].mean()),
                "mean_delta_risk_alert_rate": float(group["delta_risk_alert_rate"].mean()),
                "mean_delta_mean_vina_score": float(group["delta_mean_vina_score"].mean()),
                "median_delta_mean_vina_score": float(group["delta_mean_vina_score"].median()),
                "mean_delta_best_vina_score": float(group["delta_best_vina_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docked", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--baseline", default="boba_ucb")
    parser.add_argument("--delta-out", required=True)
    parser.add_argument("--overlap-out", required=True)
    parser.add_argument("--aggregate-out")
    args = parser.parse_args()

    docked = pd.read_csv(args.docked)
    summary = pd.read_csv(args.summary)

    delta_out = Path(args.delta_out)
    delta_out.parent.mkdir(parents=True, exist_ok=True)
    delta = delta_table(summary, args.baseline)
    delta.to_csv(delta_out, index=False)

    overlap_out = Path(args.overlap_out)
    overlap_out.parent.mkdir(parents=True, exist_ok=True)
    overlap_table(docked, args.baseline).to_csv(overlap_out, index=False)
    if args.aggregate_out:
        aggregate_out = Path(args.aggregate_out)
        aggregate_out.parent.mkdir(parents=True, exist_ok=True)
        aggregate_delta(delta, args.baseline).to_csv(aggregate_out, index=False)
    print(delta_out)
    print(overlap_out)


if __name__ == "__main__":
    main()
