from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rabvs.bandit import predict_with_uncertainty, select_batch
from rabvs.metrics import bedroc_score
from rabvs.dude import load_dude_target
from rabvs.lit_pcba import load_lit_pcba_target
from run_dude_pilot import prepare_pool as prepare_dude_pool
from run_dude_pilot import stable_target_offset
from run_lit_pcba_pilot import prepare_pool as prepare_lit_pool


DUDE_TARGETS = [
    line.strip()
    for line in (ROOT / "configs" / "dude_all102_targets.txt").read_text().splitlines()
    if line.strip()
]
LIT_TARGETS = [
    "ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1",
    "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def rrf_ranking(mu: np.ndarray, risk: np.ndarray, scaffolds: np.ndarray, wc: float, wn: float) -> np.ndarray:
    counts = np.bincount(scaffolds)
    novelty = 1.0 / np.maximum(counts[scaffolds], 1)
    score = (
        1.0 / (60.0 + rank_desc(mu))
        + wc / (60.0 + rank_asc(risk))
        + wn / (60.0 + rank_desc(novelty))
    )
    return np.argsort(-score, kind="stable")


def make_cfg(dataset: str, targets: list[str], jobs: int) -> dict:
    common = {
        "targets": targets,
        "n_partitions": 256,
        "n_components": 96,
        "initial_labeled": 512,
        "rounds": 8,
        "query_per_round": 256,
        "ensemble_size": 5,
        "retrieval_round_blend": 0.0,
        "retrieval_final_blend": 0.5,
        "n_jobs": jobs,
    }
    if dataset == "dude":
        return {
            **common,
            "data_root": "data/raw/dude/extracted",
            "max_decoys": 50000,
        }
    return {
        **common,
        "data_root": "data/raw/lit_pcba/AVE_unbiased",
        "split": "V",
        "max_inactives": 50000,
    }


def run_one(dataset: str, cfg: dict, target: str, seed: int, clean_weights: list[float], novelty_weights: list[float]) -> list[dict]:
    if dataset == "dude":
        pool, features, partitions, n_partitions = prepare_dude_pool(cfg, target, seed)
    else:
        pool, features, partitions, n_partitions = prepare_lit_pool(cfg, target, seed)

    labels = pool.labels
    # Match the exact target-string convention used by each primary runner;
    # LIT-PCBA task names are uppercase and therefore must not be normalized.
    offset_name = target.lower() if dataset == "dude" else target
    rng = np.random.default_rng(seed + stable_target_offset(offset_name))
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    init_pos_n = min(len(positives), max(1, cfg["initial_labeled"] // 32))
    init_neg_n = min(len(negatives), cfg["initial_labeled"] - init_pos_n)
    labeled = np.unique(
        np.concatenate(
            [
                rng.choice(positives, size=init_pos_n, replace=False),
                rng.choice(negatives, size=init_neg_n, replace=False),
            ]
        )
    )
    sampled_per_partition = np.bincount(partitions[labeled], minlength=n_partitions).astype(float)
    method = "rabvs_budgeted_no_ucb_pareto_rrf_no_conf"

    for round_no in range(cfg["rounds"]):
        unlabeled = np.setdiff1d(np.arange(len(labels)), labeled, assume_unique=False)
        pred = predict_with_uncertainty(
            features,
            labeled,
            labels,
            seed=seed + round_no,
            retrieval_blend=cfg["retrieval_round_blend"],
            ensemble_size=cfg["ensemble_size"],
        )
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

    final_pred = predict_with_uncertainty(
        features,
        labeled,
        labels,
        seed=seed + cfg["rounds"],
        retrieval_blend=cfg["retrieval_final_blend"],
        ensemble_size=cfg["ensemble_size"],
    )
    rows = []
    for wc in clean_weights:
        for wn in novelty_weights:
            ranking = rrf_ranking(final_pred.mu, pool.risk_alert, pool.scaffolds, wc, wn)
            top = ranking[: min(1000, len(ranking))]
            rows.append(
                {
                    "dataset": dataset.upper(),
                    "target": target,
                    "seed": seed,
                    "clean_weight": wc,
                    "novelty_weight": wn,
                    "Hit@1000": int(labels[top].sum()),
                    "BEDROC_alpha20": bedroc_score(labels, ranking, alpha=20.0),
                    "alert_free_ratio_at1000": float(1.0 - pool.risk_alert[top].mean()),
                    "unique_scaffolds_at1000": int(len(np.unique(pool.scaffolds[top]))),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dude", "lit"], required=True)
    parser.add_argument("--targets", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--jobs", type=int, default=45)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    default_targets = DUDE_TARGETS if args.dataset == "dude" else LIT_TARGETS
    targets = [x.strip() for x in args.targets.split(",") if x.strip()] or default_targets
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    clean_weights = [0.10, 0.15, 0.18, 0.20, 0.25]
    novelty_weights = [0.15, 0.20, 0.25, 0.30, 0.35]
    cfg = make_cfg(args.dataset, targets, args.jobs)
    jobs = [(target, seed) for target in targets for seed in seeds]
    results = Parallel(n_jobs=args.jobs, prefer="processes")(
        delayed(run_one)(args.dataset, cfg, target, seed, clean_weights, novelty_weights)
        for target, seed in tqdm(jobs, desc=f"{args.dataset} RRF sensitivity")
    )
    frame = pd.DataFrame([row for group in results for row in group])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary = (
        frame.groupby(["dataset", "clean_weight", "novelty_weight"], as_index=False)
        .agg(
            hit_mean=("Hit@1000", "mean"),
            hit_std=("Hit@1000", "std"),
            bedroc_mean=("BEDROC_alpha20", "mean"),
            alert_free_mean=("alert_free_ratio_at1000", "mean"),
            scaffold_mean=("unique_scaffolds_at1000", "mean"),
        )
    )
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(out)


if __name__ == "__main__":
    main()
