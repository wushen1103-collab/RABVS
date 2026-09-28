from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rabvs.bandit import final_ranking, predict_with_uncertainty, select_batch
from run_dude_pilot import prepare_pool, stable_target_offset


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def replay_ranking(
    cfg: dict,
    target: str,
    seed: int,
    method: str,
    pool,
    features: np.ndarray,
    partitions: np.ndarray,
    n_partitions: int,
) -> tuple[np.ndarray, object, np.ndarray]:
    labels = pool.labels
    rng = np.random.default_rng(seed + stable_target_offset(target))

    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    init_pos_n = min(len(positives), max(1, cfg["initial_labeled"] // 32))
    init_neg_n = min(len(negatives), cfg["initial_labeled"] - init_pos_n)
    init_pos = rng.choice(positives, size=init_pos_n, replace=False)
    init_neg = rng.choice(negatives, size=init_neg_n, replace=False)
    labeled = np.unique(np.concatenate([init_pos, init_neg]))
    sampled_per_partition = np.bincount(partitions[labeled], minlength=n_partitions).astype(float)

    last_pred = None
    for round_no in range(cfg["rounds"]):
        unlabeled = np.setdiff1d(np.arange(len(labels)), labeled, assume_unique=False)
        pred = predict_with_uncertainty(features, labeled, labels, seed=seed + round_no)
        last_pred = pred
        batch = select_batch(
            method=method,
            pred=pred,
            labels_seen=labels[labeled],
            unlabeled_idx=unlabeled,
            partitions=partitions,
            scaffolds=pool.scaffolds,
            risk_alert=pool.risk_alert,
            sampled_per_partition=sampled_per_partition,
            query_size=cfg["query_per_round"],
            round_no=round_no,
            seed=seed,
        )
        labeled = np.unique(np.concatenate([labeled, batch]))
        sampled_per_partition += np.bincount(partitions[batch], minlength=n_partitions)

    if last_pred is None:
        raise RuntimeError("No predictions produced")
    return final_ranking(method, last_pred, pool.scaffolds, pool.risk_alert), last_pred, labeled


def export_candidates(
    cfg: dict,
    targets: list[str],
    seeds: list[int],
    methods: list[str],
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for target in targets:
        for seed in seeds:
            pool, features, partitions, n_partitions = prepare_pool(cfg, target, seed)
            for method in methods:
                ranking, pred, labeled = replay_ranking(
                    cfg, target, seed, method, pool, features, partitions, n_partitions
                )
                labeled_set = set(int(idx) for idx in labeled.tolist())
                for rank, idx in enumerate(ranking[:top_n], start=1):
                    rows.append(
                        {
                            "target": target,
                            "seed": seed,
                            "method": method,
                            "rank": rank,
                            "pool_index": int(idx),
                            "mol_id": str(pool.mol_ids[idx]),
                            "smiles": str(pool.smiles[idx]),
                            "label": int(pool.labels[idx]),
                            "scaffold": int(pool.scaffolds[idx]),
                            "risk_alert": int(pool.risk_alert[idx]),
                            "mu": float(pred.mu[idx]),
                            "u_epi": float(pred.u_epi[idx]),
                            "u_conf": float(pred.u_conf[idx]),
                            "density_penalty": float(pred.density_penalty[idx]),
                            "was_queried": int(int(idx) in labeled_set),
                            "queried_total": int(len(labeled)),
                            "queried_hits": int(pool.labels[labeled].sum()),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dude_all102_final_ablation.yaml")
    parser.add_argument("--targets", default="aa2ar,adrb2")
    parser.add_argument(
        "--methods",
        default="boba_ucb,rabvs_budgeted_reliable,rabvs_budgeted_reliable_q90,rabvs_budgeted_reliable_no_density",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--out", default="runs/docking_candidates/candidates.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    targets = parse_csv_arg(args.targets)
    methods = parse_csv_arg(args.methods)
    seeds = [int(seed) for seed in parse_csv_arg(args.seeds)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = export_candidates(cfg, targets, seeds, methods, args.top_n)
    frame.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
