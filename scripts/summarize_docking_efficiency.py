from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_docked(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        seed_text, path_text = spec.split(":", 1)
        frame = pd.read_csv(path_text)
        frame["seed"] = int(seed_text)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_efficiency(
    docked: pd.DataFrame,
    pool_summary: pd.DataFrame,
    pool_method: str,
    topk_budget: int,
) -> pd.DataFrame:
    pool_sizes = pool_summary[pool_summary["method"] == pool_method][
        ["target", "seed", "pool_size"]
    ].drop_duplicates()
    rows = []
    for (seed, target, method), group in docked.groupby(["seed", "target", "method"], sort=True):
        match = pool_sizes[(pool_sizes["seed"] == seed) & (pool_sizes["target"] == target)]
        if len(match) != 1:
            raise ValueError(f"Expected one pool-size row for seed={seed}, target={target}")
        pool_size = int(match["pool_size"].iloc[0])
        ok = group[group["dock_status"] == "ok"]
        n_docked = int(len(ok))
        active_hits = int(group["label"].sum())
        rows.append(
            {
                "seed": int(seed),
                "target": target,
                "method": method,
                "pool_size": pool_size,
                "topk_protocol": int(group["rank"].max()),
                "n_candidates": int(len(group)),
                "n_docked_ok": n_docked,
                "active_hits_in_docked_top10": active_hits,
                "active_rate_in_docked_top10": float(group["label"].mean()),
                "hits_per_1000_dockings": float(active_hits / max(1, n_docked) * 1000.0),
                "docking_saved_ratio_vs_top1000": float(1.0 - n_docked / float(topk_budget)),
                "docking_saved_ratio_vs_full_pool": float(1.0 - n_docked / max(1, pool_size)),
                "gpu_hours_per_hit": "NA_CPU_Vina",
                "compute_device": "CPU_Vina",
            }
        )
    return pd.DataFrame(rows)


def method_summary(efficiency: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "active_hits_in_docked_top10",
        "active_rate_in_docked_top10",
        "hits_per_1000_dockings",
        "docking_saved_ratio_vs_top1000",
        "docking_saved_ratio_vs_full_pool",
    ]
    rows = []
    for method, group in efficiency.groupby("method", sort=True):
        row = {
            "method": method,
            "n_pairs": int(len(group)),
            "n_targets": int(group["target"].nunique()),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def delta_table(efficiency: pd.DataFrame, baseline: str) -> pd.DataFrame:
    metrics = [
        "active_hits_in_docked_top10",
        "active_rate_in_docked_top10",
        "hits_per_1000_dockings",
        "docking_saved_ratio_vs_top1000",
        "docking_saved_ratio_vs_full_pool",
    ]
    rows = []
    for (seed, target), group in efficiency.groupby(["seed", "target"], sort=True):
        base_rows = group[group["method"] == baseline]
        if len(base_rows) != 1:
            raise ValueError(f"Expected one baseline row for seed={seed}, target={target}")
        base = base_rows.iloc[0]
        for row in group.itertuples(index=False):
            if row.method == baseline:
                continue
            item = {
                "seed": int(seed),
                "target": target,
                "method": row.method,
                "baseline": baseline,
            }
            for metric in metrics:
                item[f"delta_{metric}"] = float(getattr(row, metric) - base[metric])
            rows.append(item)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path_text: str) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docked-input", action="append", required=True, help="SEED:docked.csv")
    parser.add_argument("--pool-summary", required=True)
    parser.add_argument("--pool-method", default="boba_ucb")
    parser.add_argument("--baseline", default="boba_ucb")
    parser.add_argument("--topk-budget", type=int, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method-out", required=True)
    parser.add_argument("--delta-out", required=True)
    args = parser.parse_args()

    docked = load_docked(args.docked_input)
    pool_summary = pd.read_csv(args.pool_summary)
    efficiency = build_efficiency(docked, pool_summary, args.pool_method, args.topk_budget)
    write_csv(efficiency, args.out)
    write_csv(method_summary(efficiency), args.method_out)
    write_csv(delta_table(efficiency, args.baseline), args.delta_out)


if __name__ == "__main__":
    main()
