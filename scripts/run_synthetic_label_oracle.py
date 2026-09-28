from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rabvs.bandit import final_ranking, mean_pairwise_distance, predict_with_uncertainty, select_batch
from rabvs.metrics import (
    aubc,
    brier_score,
    enrichment_factor,
    expected_calibration_error,
    hit_at,
    recall_at,
    unique_scaffolds,
)
from rabvs.synthetic import make_synthetic_pool
from rabvs.utils import cpu_worker_cap, ensure_dir


def run_one(cfg: dict, seed: int, target_idx: int, method: str) -> tuple[list[dict], dict]:
    pool = make_synthetic_pool(
        pool_size=cfg["pool_size"],
        n_targets=cfg["n_targets"],
        n_partitions=cfg["n_partitions"],
        n_features=cfg["n_features"],
        n_scaffolds=cfg["n_scaffolds"],
        seed=seed,
    )
    labels = pool.labels[:, target_idx]
    rng = np.random.default_rng(seed + 1009 * target_idx)

    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    init_pos = rng.choice(positives, size=min(len(positives), max(8, cfg["initial_labeled"] // 16)), replace=False)
    init_neg = rng.choice(negatives, size=cfg["initial_labeled"] - len(init_pos), replace=False)
    labeled = np.unique(np.concatenate([init_pos, init_neg]))

    sampled_per_partition = np.bincount(pool.partitions[labeled], minlength=cfg["n_partitions"]).astype(float)
    round_rows: list[dict] = []
    round_hits: list[int] = []
    last_pred = None

    for round_no in range(cfg["rounds"]):
        unlabeled = np.setdiff1d(np.arange(cfg["pool_size"]), labeled, assume_unique=False)
        pred = predict_with_uncertainty(pool.features, labeled, labels, seed=seed + round_no)
        last_pred = pred
        batch = select_batch(
            method=method,
            pred=pred,
            labels_seen=labels[labeled],
            unlabeled_idx=unlabeled,
            partitions=pool.partitions,
            scaffolds=pool.scaffolds,
            risk_alert=pool.risk_alert,
            sampled_per_partition=sampled_per_partition,
            query_size=cfg["query_per_round"],
            round_no=round_no,
            seed=seed,
        )
        labeled = np.unique(np.concatenate([labeled, batch]))
        sampled_per_partition += np.bincount(pool.partitions[batch], minlength=cfg["n_partitions"])
        cumulative_hits = int(labels[labeled].sum())
        round_hits.append(cumulative_hits)
        round_rows.append(
            {
                "seed": seed,
                "target": pool.target_ids[target_idx],
                "method": method,
                "round": round_no + 1,
                "queried_total": int(len(labeled)),
                "new_hits": int(labels[batch].sum()),
                "cumulative_hits": cumulative_hits,
                "active_ratio_queried": float(labels[labeled].mean()),
                "unique_scaffolds_queried": int(len(np.unique(pool.scaffolds[labeled]))),
                "risk_alert_rate_queried": float(pool.risk_alert[labeled].mean()),
            }
        )

    if last_pred is None:
        raise RuntimeError("No prediction bundle produced.")
    ranking = final_ranking(method, last_pred, pool.scaffolds, pool.risk_alert)
    topk_metrics = {}
    for k in cfg["final_k"]:
        topk_metrics[f"Hit@{k}"] = hit_at(labels, ranking, k)
        topk_metrics[f"Recall@{k}"] = recall_at(labels, ranking, k)
        topk_metrics[f"unique_scaffold_count@{k}"] = unique_scaffolds(pool.scaffolds, ranking, k)

    full_inference_budget = cfg["pool_size"] * max(1, cfg["rounds"])
    if method in {"greedy_surrogate", "uncertainty_sampling"}:
        neural_inferences = full_inference_budget
    elif method == "random":
        neural_inferences = len(labeled)
    else:
        avg_partition_size = cfg["pool_size"] / cfg["n_partitions"]
        reservoir_per_partition = max(1, int(round(0.05 * avg_partition_size)))
        reservoir_budget = cfg["n_partitions"] * min(256, reservoir_per_partition)
        neural_inferences = cfg["initial_labeled"] + reservoir_budget + cfg["rounds"] * cfg["query_per_round"]

    summary = {
        "seed": seed,
        "target": pool.target_ids[target_idx],
        "method": method,
        "active_ratio_pool": float(labels.mean()),
        "queried_total": int(len(labeled)),
        "queried_hits": int(labels[labeled].sum()),
        "AUBC": aubc(round_hits, cfg["query_per_round"]),
        "ECE": expected_calibration_error(labels, last_pred.mu),
        "Brier": brier_score(labels, last_pred.mu),
        "neural_inferences": int(neural_inferences),
        "inference_saved_ratio": float(1.0 - neural_inferences / full_inference_budget),
        "hits_per_1000_inferences": float(topk_metrics["Hit@1000"] / max(1, neural_inferences) * 1000.0),
        "EF_0.1pct": enrichment_factor(labels, ranking, 0.001),
        "EF_1pct": enrichment_factor(labels, ranking, 0.01),
        "mean_pairwise_cosine_distance_at1000": mean_pairwise_distance(pool.features, ranking, 1000),
        "PAINS_Brenk_free_ratio_at1000": float(1.0 - pool.risk_alert[ranking[:1000]].mean()),
        **topk_metrics,
    }
    return round_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ensure_dir(cfg["output_dir"])

    jobs = [
        (seed, target_idx, method)
        for seed in cfg["seeds"]
        for target_idx in range(cfg["n_targets"])
        for method in cfg["methods"]
    ]
    n_jobs = cpu_worker_cap(int(cfg.get("n_jobs", 1)))
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(run_one)(cfg, seed, target_idx, method)
        for seed, target_idx, method in tqdm(jobs, desc="synthetic label-oracle")
    )

    round_rows = [row for rows, _ in results for row in rows]
    summaries = [summary for _, summary in results]
    pd.DataFrame(round_rows).to_csv(out_dir / "round_logs.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
