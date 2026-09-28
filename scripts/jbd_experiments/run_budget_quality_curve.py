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

from rabvs.bandit import predict_with_uncertainty, topmean
from rabvs.metrics import bedroc_score
from run_dude_pilot import prepare_pool, stable_target_offset


DEFAULT_TARGETS = [
    "aa2ar", "abl1", "ace", "aces", "ada", "akt1", "akt2", "aldr", "ampc", "andr",
    "aofb", "bace1", "braf", "cah2", "casp3", "cdk2", "comt", "cp2c9", "cp3a4", "csf1r",
]
METHODS = ["rabvs", "random_partition", "greedy_partition", "boba_ucb", "uncertainty_partition"]


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


def rrf_order(mu: np.ndarray, risk: np.ndarray, scaffolds: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(scaffolds, return_inverse=True, return_counts=True)
    novelty = 1.0 / counts[inverse]
    score = (
        1.0 / (60.0 + rank_desc(mu))
        + 0.18 / (60.0 + rank_asc(risk))
        + 0.25 / (60.0 + rank_desc(novelty))
    )
    return np.argsort(-score, kind="stable")


def choose_initial(labels: np.ndarray, size: int, seed: int, target: str) -> np.ndarray:
    rng = np.random.default_rng(seed + stable_target_offset(target))
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    positive_n = min(len(positive), max(1, size // 32))
    negative_n = min(len(negative), size - positive_n)
    return np.unique(
        np.concatenate(
            [
                rng.choice(positive, size=positive_n, replace=False),
                rng.choice(negative, size=negative_n, replace=False),
            ]
        )
    )


def partition_members(partitions: np.ndarray, n_parts: int) -> list[np.ndarray]:
    return [np.flatnonzero(partitions == part) for part in range(n_parts)]


def reservoir_sample(members: list[np.ndarray], seed: int, per_part: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = []
    part_ids = []
    for part, idx in enumerate(members):
        if not len(idx):
            continue
        chosen = rng.choice(idx, size=min(per_part, len(idx)), replace=False)
        indices.extend(chosen.tolist())
        part_ids.extend([part] * len(chosen))
    return np.asarray(indices, dtype=int), np.asarray(part_ids, dtype=int)


def score_partitions(method: str, pred, pool, partitions: np.ndarray, reservoir: np.ndarray, reservoir_parts: np.ndarray, labeled: np.ndarray, n_parts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method == "random_partition":
        values = rng.random(n_parts)
        return np.argsort(-values, kind="stable")

    sampled = np.bincount(partitions[labeled], minlength=n_parts)
    values = np.full(n_parts, -np.inf, dtype=float)
    for part in range(n_parts):
        pos = np.flatnonzero(reservoir_parts == part)
        if not len(pos):
            continue
        idx = reservoir[pos]
        if method == "greedy_partition":
            values[part] = float(pred.mu[idx].mean())
        elif method == "uncertainty_partition":
            values[part] = topmean(pred.uncertainty[idx], q=0.5)
        else:
            diversity = len(np.unique(pool.scaffolds[idx])) / len(idx)
            values[part] = topmean(pred.mu[idx], q=0.25) + 0.10 * diversity
            if method == "boba_ucb":
                values[part] += 0.5 * np.sqrt(2.0 * np.log(2.0) / max(1.0, sampled[part]))
    return np.argsort(-values, kind="stable")


def select_scored(order: np.ndarray, members: list[np.ndarray], reservoir: np.ndarray, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = set(int(x) for x in reservoir)
    for part in order:
        remaining = budget - len(selected)
        if remaining <= 0:
            break
        candidates = np.asarray([x for x in members[int(part)] if int(x) not in selected], dtype=int)
        if len(candidates) > remaining:
            candidates = rng.choice(candidates, size=remaining, replace=False)
        selected.update(int(x) for x in candidates)
    return np.asarray(sorted(selected), dtype=int)


def run_one(target: str, seed: int, ratios: list[float]) -> list[dict]:
    cfg = {
        "data_root": "data/raw/dude/extracted",
        "max_decoys": 50000,
        "n_partitions": 64,
        "n_components": 96,
        "initial_labeled": 256,
    }
    pool, features, partitions, n_parts = prepare_pool(cfg, target, seed)
    labels = pool.labels
    labeled = choose_initial(labels, cfg["initial_labeled"], seed, target)
    pred = predict_with_uncertainty(
        features,
        labeled,
        labels,
        seed=seed,
        ensemble_size=5,
        retrieval_blend=0.5,
    )
    members = partition_members(partitions, n_parts)
    reservoir, reservoir_parts = reservoir_sample(members, seed + stable_target_offset(target), per_part=4)
    full_order = rrf_order(pred.mu, pool.risk_alert, pool.scaffolds)
    full_top = full_order[: min(1000, len(full_order))]
    full_hits = int(labels[full_top].sum())
    rows = []
    for ratio in ratios:
        budget = min(len(labels), max(min(1000, len(labels)), int(np.ceil(ratio * len(labels))), len(reservoir)))
        for method in METHODS:
            part_order = score_partitions(method, pred, pool, partitions, reservoir, reservoir_parts, labeled, n_parts, seed + 1009)
            scored = select_scored(part_order, members, reservoir, budget, seed + 7919)
            local_order = rrf_order(pred.mu[scored], pool.risk_alert[scored], pool.scaffolds[scored])
            ranked_scored = scored[local_order]
            unscored = np.setdiff1d(np.arange(len(labels)), scored, assume_unique=False)
            ranking = np.concatenate([ranked_scored, unscored])
            top = ranking[: min(1000, len(ranking))]
            hits = int(labels[top].sum())
            overlap = len(set(top.tolist()) & set(full_top.tolist())) / max(1, len(full_top))
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "method": method,
                    "nominal_budget_fraction": ratio,
                    "realized_budget_fraction": len(scored) / len(labels),
                    "pool_size": len(labels),
                    "scored_unique": len(scored),
                    "Hit@1000": hits,
                    "Hit@1000_retention": hits / max(1, full_hits),
                    "BEDROC_alpha20": bedroc_score(labels, ranking, alpha=20.0),
                    "hits_per_1000_scores": hits / max(1, len(scored)) * 1000.0,
                    "topK_overlap": overlap,
                    "alert_free_ratio_at1000": float(1.0 - pool.risk_alert[top].mean()),
                    "unique_scaffolds_at1000": int(len(np.unique(pool.scaffolds[top]))),
                    "full_reference_Hit@1000": full_hits,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ratios", default="0.01,0.025,0.05,0.10,0.20,0.50")
    parser.add_argument("--jobs", type=int, default=45)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    targets = [x.strip().lower() for x in args.targets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    jobs = [(target, seed) for target in targets for seed in seeds]
    results = Parallel(n_jobs=args.jobs, prefer="processes")(
        delayed(run_one)(target, seed, ratios)
        for target, seed in tqdm(jobs, desc="budget-quality curve")
    )
    frame = pd.DataFrame([row for group in results for row in group])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary = (
        frame.groupby(["method", "nominal_budget_fraction"], as_index=False)
        .agg(
            realized_budget_fraction=("realized_budget_fraction", "mean"),
            hit_retention=("Hit@1000_retention", "mean"),
            bedroc=("BEDROC_alpha20", "mean"),
            hits_per_1000_scores=("hits_per_1000_scores", "mean"),
            topK_overlap=("topK_overlap", "mean"),
            alert_free_ratio=("alert_free_ratio_at1000", "mean"),
            unique_scaffolds=("unique_scaffolds_at1000", "mean"),
        )
    )
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(out)


if __name__ == "__main__":
    main()
