from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rabvs.dude import load_dude_target
from rabvs.bandit import mean_pairwise_distance, topmean
from rabvs.utils import ensure_dir
from run_dude_pilot import stable_target_offset
from run_true_budgeted_dude import fit_subset_predictor


def peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2))


def initial_labeled_indices(labels: np.ndarray, size: int, seed: int, target: str) -> np.ndarray:
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


def sampled_large_pool(base_size: int, source_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if base_size <= source_n:
        return rng.choice(source_n, size=base_size, replace=False)
    return rng.choice(source_n, size=base_size, replace=True)


def local_screening_indices(
    features: np.ndarray,
    predictor,
    partitions: np.ndarray,
    n_partitions: int,
    selected_partitions: int,
    reservoir_per_partition: int,
    seed: int,
) -> tuple[np.ndarray, float, float, int]:
    rng = np.random.default_rng(seed)
    reservoir_parts = []
    reservoir_chunks = []
    start = time.perf_counter()
    for part in range(n_partitions):
        members = np.flatnonzero(partitions == part)
        if len(members) == 0:
            continue
        sample_n = min(len(members), reservoir_per_partition)
        chosen = rng.choice(members, size=sample_n, replace=False)
        reservoir_chunks.append(chosen)
        reservoir_parts.extend([part] * len(chosen))
    reservoir_idx = np.concatenate(reservoir_chunks) if reservoir_chunks else np.empty(0, dtype=int)
    if len(reservoir_idx):
        reservoir_pred = predictor.predict(features[reservoir_idx])
        reservoir_parts_arr = np.asarray(reservoir_parts, dtype=int)
        scores = np.full(n_partitions, -np.inf, dtype=float)
        for part in np.unique(reservoir_parts_arr):
            scores[part] = topmean(reservoir_pred.mu[reservoir_parts_arr == part], q=0.05)
    else:
        scores = np.full(n_partitions, -np.inf, dtype=float)
    selected = np.argsort(-scores)[: min(selected_partitions, n_partitions)]
    candidates = np.flatnonzero(np.isin(partitions, selected))
    scored = np.unique(np.concatenate([reservoir_idx, candidates]))
    selection_seconds = time.perf_counter() - start

    score_start = time.perf_counter()
    _ = predictor.predict(features[scored])
    score_seconds = time.perf_counter() - score_start
    return scored, selection_seconds, score_seconds, int(len(np.unique(reservoir_idx)))


def run_one(args, source_pool, reduced: np.ndarray, size: int, seed: int) -> dict:
    source_n = len(reduced)
    sample_idx = sampled_large_pool(size, source_n, seed)
    features = reduced[sample_idx].astype(np.float32, copy=True)
    scaffolds = source_pool.scaffolds[sample_idx]
    risk_alert = source_pool.risk_alert[sample_idx]

    labeled = initial_labeled_indices(source_pool.labels, args.initial_labeled, seed, args.target)
    predictor = fit_subset_predictor(reduced, labeled, source_pool.labels, seed, args.ensemble_size)

    partition_start = time.perf_counter()
    n_partitions = min(args.n_partitions, max(2, size // max(1, args.molecules_per_partition)))
    partitions = MiniBatchKMeans(
        n_clusters=n_partitions,
        batch_size=args.batch_size,
        n_init=1,
        random_state=seed,
    ).fit_predict(features)
    partition_seconds = time.perf_counter() - partition_start

    full_start = time.perf_counter()
    full_pred = predictor.predict(features)
    full_score_seconds = time.perf_counter() - full_start
    full_rank = np.argsort(-full_pred.mu)

    scored, local_select_seconds, local_score_seconds, reservoir_count = local_screening_indices(
        features,
        predictor,
        partitions,
        n_partitions,
        args.selected_partitions,
        args.reservoir_per_partition,
        seed,
    )
    local_pred = predictor.predict(features[scored])
    local_score = local_pred.mu - 0.1 * risk_alert[scored]
    local_rank = scored[np.argsort(-local_score)]

    top_k = min(args.top_k, len(local_rank), len(full_rank))
    full_top = set(full_rank[:top_k].tolist())
    local_top = set(local_rank[:top_k].tolist())
    top = local_rank[:top_k]

    return {
        "source": "DUD-E-derived stress pool",
        "target": args.target,
        "seed": seed,
        "stress_pool_size": int(size),
        "source_pool_size": int(source_n),
        "n_partitions": int(n_partitions),
        "selected_partitions": int(min(args.selected_partitions, n_partitions)),
        "reservoir_per_partition": int(args.reservoir_per_partition),
        "reservoir_scored_molecules": int(reservoir_count),
        "local_scored_molecules": int(len(scored)),
        "local_scored_fraction": float(len(scored) / size),
        "inference_saved_ratio": float(1.0 - len(scored) / size),
        "partition_seconds": partition_seconds,
        "full_score_seconds": full_score_seconds,
        "local_select_seconds": local_select_seconds,
        "local_score_seconds": local_score_seconds,
        "local_total_seconds": local_select_seconds + local_score_seconds,
        "score_speedup": float(full_score_seconds / max(local_score_seconds, 1e-12)),
        "online_speedup_excluding_partition": float(full_score_seconds / max(local_select_seconds + local_score_seconds, 1e-12)),
        "topk_overlap_with_full": float(len(full_top & local_top) / max(1, top_k)),
        "topk_alert_free_ratio": float(1.0 - risk_alert[top].mean()) if len(top) else 0.0,
        "topk_unique_scaffolds": int(len(np.unique(scaffolds[top]))) if len(top) else 0,
        "topk_mean_pairwise_cosine_distance": mean_pairwise_distance(features, local_rank, top_k),
        "peak_rss_gib": peak_rss_gib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/dude/extracted")
    parser.add_argument("--target", default="aa2ar")
    parser.add_argument("--out", default="tables/large_pool_scalability_stress.csv")
    parser.add_argument("--sizes", default="10000,50000,100000")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-components", type=int, default=96)
    parser.add_argument("--n-partitions", type=int, default=1024)
    parser.add_argument("--molecules-per-partition", type=int, default=250)
    parser.add_argument("--selected-partitions", type=int, default=32)
    parser.add_argument("--reservoir-per-partition", type=int, default=32)
    parser.add_argument("--initial-labeled", type=int, default=512)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    pool = load_dude_target(args.data_root, args.target, max_decoys=50000, seed=0)
    n_components = min(args.n_components, pool.features.shape[1] - 1, pool.features.shape[0] - 1)
    reduced = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(pool.features).astype(np.float32)

    rows = []
    for seed in seeds:
        for size in sizes:
            rows.append(run_one(args, pool, reduced, size=size, seed=seed))
    out = Path(args.out)
    ensure_dir(out.parent)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
