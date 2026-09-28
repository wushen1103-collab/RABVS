from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rabvs.dude import load_dude_target
from run_true_budgeted_dude import fit_subset_predictor, initial_labeled_indices


N_BITS = 2048
N_COMPONENTS = 96


def projection_matrix() -> np.ndarray:
    bit = np.arange(N_BITS, dtype=np.uint64)
    hashed = bit * np.uint64(11400714819323198485) + np.uint64(7046029254386353131)
    dims = (hashed % np.uint64(N_COMPONENTS)).astype(np.int32)
    signs = np.where(((hashed >> np.uint64(32)) & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float32)
    matrix = np.zeros((N_BITS, N_COMPONENTS), dtype=np.float32)
    matrix[np.arange(N_BITS), dims] = signs
    return matrix


def hashed_features(dense_bits: np.ndarray) -> np.ndarray:
    matrix = dense_bits @ projection_matrix()
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norm, 1e-12)
    return matrix.astype(np.float32)


def peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2))


def build_predictor(
    target: str,
    seed: int,
    data_root: str,
    ensemble_size: int,
    representation: str,
    svd_components: np.ndarray | None,
):
    pool = load_dude_target(data_root, target, max_decoys=50000, seed=seed)
    if representation == "landmark-svd":
        if svd_components is None:
            raise ValueError("Landmark-SVD components are required")
        features = normalize(
            np.asarray(pool.features @ svd_components.T, dtype=np.float32), norm="l2"
        ).astype(np.float32)
    else:
        features = hashed_features(pool.features)
    labeled = initial_labeled_indices(pool.labels, 512, seed, target)
    return fit_subset_predictor(features, labeled, pool.labels, seed, ensemble_size)


def partition_reservoir(partitions: np.ndarray, n_parts: int, per_part: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.argsort(partitions, kind="stable")
    counts = np.bincount(partitions, minlength=n_parts)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    indices = []
    part_ids = []
    for part in range(n_parts):
        members = order[bounds[part] : bounds[part + 1]]
        if not len(members):
            continue
        chosen = rng.choice(members, size=min(per_part, len(members)), replace=False)
        indices.extend(chosen.tolist())
        part_ids.extend([part] * len(chosen))
    return np.asarray(indices, dtype=int), np.asarray(part_ids, dtype=int)


def run_target(features: np.ndarray, partitions: np.ndarray, n_parts: int, predictor, target: str, size: int, selected_fraction: float, reservoir_per_part: int, top_k: int, seed: int) -> dict:
    all_idx = np.arange(size, dtype=int)
    full_start = time.perf_counter()
    full_pred = predictor.predict(features)
    full_seconds = time.perf_counter() - full_start
    full_rank = np.argsort(-full_pred.mu, kind="stable")

    online_start = time.perf_counter()
    reservoir, reservoir_parts = partition_reservoir(partitions, n_parts, reservoir_per_part, seed)
    reservoir_pred = predictor.predict(features[reservoir])
    utility = np.full(n_parts, -np.inf, dtype=float)
    for part in range(n_parts):
        pos = np.flatnonzero(reservoir_parts == part)
        if not len(pos):
            continue
        scores = reservoir_pred.mu[pos]
        k = max(1, int(np.ceil(0.25 * len(scores))))
        top_mean = float(np.partition(scores, -k)[-k:].mean())
        diversity = float(np.mean(np.std(features[reservoir[pos]], axis=0)))
        utility[part] = top_mean + 0.10 * diversity
    selected_n = max(1, int(np.ceil(selected_fraction * n_parts)))
    selected_parts = np.argsort(-utility, kind="stable")[:selected_n]
    candidates = np.flatnonzero(np.isin(partitions, selected_parts))
    scored = np.unique(np.concatenate([reservoir, candidates]))
    reservoir_lookup = {int(idx): pos for pos, idx in enumerate(reservoir)}
    new_idx = np.asarray([idx for idx in scored if int(idx) not in reservoir_lookup], dtype=int)
    new_pred = predictor.predict(features[new_idx])
    mu = np.empty(len(scored), dtype=float)
    new_lookup = {int(idx): pos for pos, idx in enumerate(new_idx)}
    for pos, idx in enumerate(scored):
        key = int(idx)
        mu[pos] = reservoir_pred.mu[reservoir_lookup[key]] if key in reservoir_lookup else new_pred.mu[new_lookup[key]]
    local_rank = scored[np.argsort(-mu, kind="stable")]
    online_seconds = time.perf_counter() - online_start

    k = min(top_k, len(local_rank), len(full_rank))
    overlap = len(set(local_rank[:k].tolist()) & set(full_rank[:k].tolist())) / max(1, k)
    return {
        "source": "ChEMBL 36",
        "target": target,
        "seed": seed,
        "library_size": size,
        "n_partitions": n_parts,
        "selected_partitions": selected_n,
        "reservoir_scored": len(reservoir),
        "locally_scored": len(scored),
        "scores_avoided": 1.0 - len(scored) / size,
        "full_score_seconds": full_seconds,
        "online_seconds": online_seconds,
        "online_speedup": full_seconds / max(online_seconds, 1e-12),
        "topK_overlap": overlap,
        "peak_rss_gib": peak_rss_gib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--data-root", default="data/raw/dude/extracted")
    parser.add_argument(
        "--targets",
        default="aa2ar,abl1,bace1,braf,cdk2,egfr,esr1,akt1,kit,pparg",
    )
    parser.add_argument("--sizes", default="100000,250000,500000,1000000,2000000")
    parser.add_argument("--selected-fraction", type=float, default=0.10)
    parser.add_argument("--reservoir-per-part", type=int, default=16)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    metadata_path = feature_dir / "chembl36_features.json"
    if not metadata_path.exists():
        metadata_path = feature_dir / "chembl36_hashed_morgan96.json"
    metadata = json.loads(metadata_path.read_text())
    valid = int(metadata["usable_rows"])
    feature_path = feature_dir / Path(metadata["feature_file"]).name
    all_features = np.load(feature_path, mmap_mode="r")
    representation = metadata.get("projection_mode", "signed-hash")
    components_path = feature_dir / "chembl36_landmark_svd96_components.npy"
    svd_components = (
        np.load(components_path)
        if representation == "landmark-svd"
        else None
    )
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    targets = [x.strip().lower() for x in args.targets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if max(sizes) > valid:
        raise ValueError(f"Requested {max(sizes)} molecules, cache has {valid}")
    rows = []
    index_rows = []
    for seed in seeds:
        predictors = {
            target: build_predictor(
                target,
                seed,
                args.data_root,
                args.ensemble_size,
                representation,
                svd_components,
            )
            for target in targets
        }
        for size in sizes:
            features = np.asarray(all_features[:size], dtype=np.float32)
            n_parts = min(2048, max(256, int(np.ceil(size / 1000))))
            index_start = time.perf_counter()
            partitions = MiniBatchKMeans(
                n_clusters=n_parts,
                batch_size=16384,
                n_init=1,
                max_iter=50,
                random_state=seed,
            ).fit_predict(features)
            index_seconds = time.perf_counter() - index_start
            index_rows.append(
                {
                    "seed": seed,
                    "library_size": size,
                    "n_partitions": n_parts,
                    "index_seconds": index_seconds,
                    "index_peak_rss_gib": peak_rss_gib(),
                }
            )
            for target, predictor in predictors.items():
                row = run_target(
                    features,
                    partitions,
                    n_parts,
                    predictor,
                    target,
                    size,
                    args.selected_fraction,
                    args.reservoir_per_part,
                    args.top_k,
                    seed,
                )
                row["selection_seed"] = metadata.get("selection_seed")
                row["nested_random_prefixes"] = bool(metadata.get("nested_prefixes", False))
                row["representation"] = metadata.get("projection_mode", "signed-hash")
                row["unique_scores_avoided"] = row["scores_avoided"]
                row["index_seconds"] = index_seconds
                row["total_first_target_seconds"] = index_seconds + row["online_seconds"]
                rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    index_frame = pd.DataFrame(index_rows)
    index_frame.to_csv(out.with_name(out.stem + "_index.csv"), index=False)

    summary = (
        frame.groupby("library_size", as_index=False)
        .agg(
            n_target_seed=("topK_overlap", "size"),
            unique_scores_avoided_mean=("unique_scores_avoided", "mean"),
            unique_scores_avoided_std=("unique_scores_avoided", "std"),
            topK_overlap_mean=("topK_overlap", "mean"),
            topK_overlap_std=("topK_overlap", "std"),
            full_score_seconds_mean=("full_score_seconds", "mean"),
            online_seconds_mean=("online_seconds", "mean"),
            online_speedup_mean=("online_speedup", "mean"),
            online_speedup_std=("online_speedup", "std"),
            peak_rss_gib_mean=("peak_rss_gib", "mean"),
        )
    )
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)

    largest = frame[frame.library_size == frame.library_size.max()]
    index_cost = float(
        index_frame[index_frame.library_size == index_frame.library_size.max()].index_seconds.mean()
    )
    full_mean = float(largest.full_score_seconds.mean())
    online_mean = float(largest.online_seconds.mean())
    amortized = []
    for target_count in [1, 2, 5, 10, 20, 50, 100]:
        amortized.append(
            {
                "target_count": target_count,
                "exhaustive_seconds": target_count * full_mean,
                "rabvs_seconds_including_index": index_cost + target_count * online_mean,
                "amortized_speedup": target_count * full_mean / max(index_cost + target_count * online_mean, 1e-12),
            }
        )
    pd.DataFrame(amortized).to_csv(out.with_name(out.stem + "_amortization.csv"), index=False)
    print(out)


if __name__ == "__main__":
    main()
