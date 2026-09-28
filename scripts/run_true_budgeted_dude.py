from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rabvs.bandit import PredictionBundle, final_ranking, mean_pairwise_distance, topmean
from rabvs.metrics import aubc, aubc_recall, brier_score, expected_calibration_error
from rabvs.utils import cpu_worker_cap, ensure_dir
from run_dude_pilot import prepare_pool, stable_target_offset


@dataclass
class SubsetPredictor:
    models: list[LogisticRegression]
    nearest_neighbors: NearestNeighbors | None
    constant_probability: float | None

    def predict(self, features: np.ndarray) -> PredictionBundle:
        if len(features) == 0:
            empty = np.empty(0, dtype=float)
            return PredictionBundle(empty, empty.copy(), empty.copy(), empty.copy())
        if self.constant_probability is not None:
            base = np.full(len(features), self.constant_probability, dtype=float)
            ones = np.ones(len(features), dtype=float)
            return PredictionBundle(base, ones, ones, ones)
        probabilities = np.vstack([model.predict_proba(features)[:, 1] for model in self.models])
        mu = probabilities.mean(axis=0)
        u_epi = probabilities.var(axis=0)
        if self.nearest_neighbors is None:
            raise RuntimeError("Nearest-neighbor model was not fitted")
        distances, _ = self.nearest_neighbors.kneighbors(features)
        density = distances.mean(axis=1)
        return PredictionBundle(mu, u_epi, density.copy(), density)


def fit_subset_predictor(
    features: np.ndarray,
    labeled: np.ndarray,
    labels: np.ndarray,
    seed: int,
    ensemble_size: int,
) -> SubsetPredictor:
    x_labeled = features[labeled]
    y_labeled = labels[labeled]
    if len(np.unique(y_labeled)) < 2:
        probability = float(y_labeled.mean()) if len(y_labeled) else 0.01
        return SubsetPredictor([], None, probability)

    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y_labeled == 1)
    negative = np.flatnonzero(y_labeled == 0)
    models = []
    for member in range(ensemble_size):
        positive_boot = rng.choice(positive, size=len(positive), replace=True)
        negative_boot = rng.choice(negative, size=len(negative), replace=True)
        boot = np.concatenate([positive_boot, negative_boot])
        rng.shuffle(boot)
        model = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed + member,
        )
        model.fit(x_labeled[boot], y_labeled[boot])
        models.append(model)

    nearest_neighbors = NearestNeighbors(n_neighbors=min(5, len(labeled)))
    nearest_neighbors.fit(x_labeled)
    return SubsetPredictor(models, nearest_neighbors, None)


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


def partition_members(partitions: np.ndarray, n_partitions: int) -> list[np.ndarray]:
    order = np.argsort(partitions, kind="stable")
    counts = np.bincount(partitions, minlength=n_partitions)
    boundaries = np.concatenate([[0], np.cumsum(counts)])
    return [order[boundaries[part] : boundaries[part + 1]] for part in range(n_partitions)]


def build_reservoirs(
    members: list[np.ndarray],
    fraction: float,
    maximum: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    reservoir_indices = []
    reservoir_positions = []
    offset = 0
    for indices in members:
        size = min(len(indices), maximum, max(1, int(np.ceil(fraction * len(indices)))))
        chosen = np.sort(rng.choice(indices, size=size, replace=False))
        reservoir_indices.append(chosen)
        reservoir_positions.append(np.arange(offset, offset + len(chosen), dtype=int))
        offset += len(chosen)
    return np.concatenate(reservoir_indices), reservoir_positions


def peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2))


def topk_summary(
    cfg: dict,
    labels: np.ndarray,
    scaffolds: np.ndarray,
    risk_alert: np.ndarray,
    features: np.ndarray,
    ranking: np.ndarray,
) -> dict:
    result = {}
    positive_count = max(1, int(labels.sum()))
    for k in cfg["final_k"]:
        top = ranking[: min(k, len(ranking))]
        hits = int(labels[top].sum())
        result[f"Hit@{k}"] = hits
        result[f"Recall@{k}"] = float(hits / positive_count)
        result[f"unique_scaffold_count@{k}"] = int(len(np.unique(scaffolds[top])))
    top1000 = ranking[: min(1000, len(ranking))]
    result["mean_pairwise_cosine_distance_at1000"] = mean_pairwise_distance(
        features, ranking, 1000
    )
    result["PAINS_Brenk_free_ratio_at1000"] = float(
        1.0 - risk_alert[top1000].mean()
    )
    return result


def final_metrics(
    cfg: dict,
    pool,
    features: np.ndarray,
    ranking: np.ndarray,
    prediction: PredictionBundle,
    prediction_indices: np.ndarray,
    labeled: np.ndarray,
    round_hits: list[int],
    neural_inferences: int,
    full_reference: int,
    online_seconds: float,
    fit_seconds: float,
    score_seconds: float,
    selection_seconds: float,
    method: str,
    target: str,
    seed: int,
    pool_prepare_seconds: float,
    initial_hits: int,
) -> dict:
    top_metrics = topk_summary(
        cfg, pool.labels, pool.scaffolds, pool.risk_alert, features, ranking
    )
    evaluation_labels = pool.labels[prediction_indices]
    return {
        "seed": seed,
        "target": target,
        "method": method,
        "pool_size": int(len(pool.labels)),
        "active_count": int(pool.labels.sum()),
        "queried_total": int(len(labeled)),
        "queried_hits": int(pool.labels[labeled].sum()),
        "AUBC_legacy_shape": aubc(round_hits, cfg["query_per_round"]),
        "AUBC_recall": aubc_recall(
            round_hits,
            cfg["query_per_round"],
            int(pool.labels.sum()),
            initial_hits,
        ),
        "ECE_scored": expected_calibration_error(evaluation_labels, prediction.mu),
        "Brier_scored": brier_score(evaluation_labels, prediction.mu),
        "neural_inferences_actual": int(neural_inferences),
        "full_inference_reference": int(full_reference),
        "inference_saved_ratio_actual": float(1.0 - neural_inferences / full_reference),
        "scored_unique": int(len(np.unique(prediction_indices))),
        "scored_fraction": float(len(np.unique(prediction_indices)) / len(pool.labels)),
        "final_candidate_count": int(len(ranking)),
        "reservoir_refresh": str(cfg.get("reservoir_refresh", "fixed")),
        "reservoir_fraction": float(cfg.get("reservoir_fraction", 0.0)),
        "candidate_multiplier": int(cfg.get("candidate_multiplier", 0)),
        "selected_partitions": int(cfg.get("selected_partitions", 0)),
        "hits_per_1000_inferences": float(
            top_metrics["Hit@1000"] / max(1, neural_inferences) * 1000.0
        ),
        "pool_prepare_seconds": pool_prepare_seconds,
        "online_wall_seconds": online_seconds,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "selection_seconds": selection_seconds,
        "peak_rss_gib": peak_rss_gib(),
        **top_metrics,
    }


def run_full_greedy(
    cfg: dict,
    target: str,
    seed: int,
    pool,
    features: np.ndarray,
    pool_prepare_seconds: float,
) -> tuple[list[dict], dict]:
    start = time.perf_counter()
    labels = pool.labels
    all_indices = np.arange(len(labels), dtype=int)
    labeled = initial_labeled_indices(labels, cfg["initial_labeled"], seed, target)
    initial_hits = int(labels[labeled].sum())
    round_hits = []
    round_rows = []
    fit_seconds = score_seconds = selection_seconds = 0.0
    neural_inferences = 0

    for round_no in range(cfg["rounds"]):
        fit_start = time.perf_counter()
        predictor = fit_subset_predictor(
            features, labeled, labels, seed + round_no, cfg["ensemble_size"]
        )
        fit_seconds += time.perf_counter() - fit_start
        score_start = time.perf_counter()
        prediction = predictor.predict(features)
        score_seconds += time.perf_counter() - score_start
        neural_inferences += len(all_indices)
        unlabeled = np.setdiff1d(all_indices, labeled, assume_unique=False)
        select_start = time.perf_counter()
        batch = unlabeled[np.argsort(-prediction.mu[unlabeled])[: cfg["query_per_round"]]]
        selection_seconds += time.perf_counter() - select_start
        labeled = np.unique(np.concatenate([labeled, batch]))
        cumulative_hits = int(labels[labeled].sum())
        round_hits.append(cumulative_hits)
        round_rows.append(
            {
                "seed": seed,
                "target": target,
                "method": "full_greedy",
                "round": round_no + 1,
                "reservoir_inferences": 0,
                "candidate_inferences": len(all_indices),
                "cumulative_neural_inferences": neural_inferences,
                "queried_total": len(labeled),
                "new_hits": int(labels[batch].sum()),
                "cumulative_hits": cumulative_hits,
                "selected_partition_count": 0,
                "new_selected_partitions": 0,
                "candidate_pool_size": len(all_indices),
                "candidate_set_size": len(all_indices),
                "candidate_active_rate_eval": float(labels.mean()),
            }
        )

    fit_start = time.perf_counter()
    predictor = fit_subset_predictor(
        features, labeled, labels, seed + cfg["rounds"], cfg["ensemble_size"]
    )
    fit_seconds += time.perf_counter() - fit_start
    score_start = time.perf_counter()
    prediction = predictor.predict(features)
    score_seconds += time.perf_counter() - score_start
    neural_inferences += len(all_indices)
    ranking = np.argsort(-prediction.mu)
    full_reference = len(labels) * (cfg["rounds"] + 1)
    summary = final_metrics(
        cfg,
        pool,
        features,
        ranking,
        prediction,
        all_indices,
        labeled,
        round_hits,
        neural_inferences,
        full_reference,
        time.perf_counter() - start,
        fit_seconds,
        score_seconds,
        selection_seconds,
        "full_greedy",
        target,
        seed,
        pool_prepare_seconds,
        initial_hits,
    )
    return round_rows, summary


def run_true_budgeted(
    cfg: dict,
    target: str,
    seed: int,
    pool,
    features: np.ndarray,
    partitions: np.ndarray,
    n_partitions: int,
    pool_prepare_seconds: float,
) -> tuple[list[dict], dict]:
    start = time.perf_counter()
    labels = pool.labels
    all_indices = np.arange(len(labels), dtype=int)
    labeled = initial_labeled_indices(labels, cfg["initial_labeled"], seed, target)
    initial_hits = int(labels[labeled].sum())
    sampled_per_partition = np.bincount(
        partitions[labeled], minlength=n_partitions
    ).astype(float)
    rng = np.random.default_rng(seed + stable_target_offset(target) + 1847)
    members = partition_members(partitions, n_partitions)
    reservoir_refresh = str(cfg.get("reservoir_refresh", "fixed"))
    if reservoir_refresh not in {"fixed", "rotate"}:
        raise ValueError(f"Unknown reservoir_refresh: {reservoir_refresh}")
    fixed_reservoir = None
    if reservoir_refresh == "fixed":
        fixed_reservoir = build_reservoirs(
            members,
            cfg["reservoir_fraction"],
            cfg["max_reservoir_per_partition"],
            rng,
        )

    round_hits = []
    round_rows = []
    ever_scored: set[int] = set()
    fit_seconds = score_seconds = selection_seconds = 0.0
    neural_inferences = 0
    selected_parts_ever: set[int] = set()
    reported_method = str(cfg.get("reported_method", "true_budgeted_reliable"))

    for round_no in range(cfg["rounds"]):
        if fixed_reservoir is None:
            reservoir_indices, reservoir_positions = build_reservoirs(
                members,
                cfg["reservoir_fraction"],
                cfg["max_reservoir_per_partition"],
                rng,
            )
        else:
            reservoir_indices, reservoir_positions = fixed_reservoir
        fit_start = time.perf_counter()
        predictor = fit_subset_predictor(
            features, labeled, labels, seed + round_no, cfg["ensemble_size"]
        )
        fit_seconds += time.perf_counter() - fit_start

        score_start = time.perf_counter()
        reservoir_prediction = predictor.predict(features[reservoir_indices])
        score_seconds += time.perf_counter() - score_start
        neural_inferences += len(reservoir_indices)
        ever_scored.update(int(idx) for idx in reservoir_indices)

        select_start = time.perf_counter()
        partition_scores = np.full(n_partitions, -np.inf, dtype=float)
        exploration_time = max(2, round_no + 2)
        for part, positions in enumerate(reservoir_positions):
            if len(positions) == 0:
                continue
            exploration = np.sqrt(
                2.0 * np.log(exploration_time) / max(1.0, sampled_per_partition[part])
            )
            diversity = len(np.unique(pool.scaffolds[reservoir_indices[positions]])) / len(
                positions
            )
            partition_scores[part] = (
                topmean(reservoir_prediction.mu[positions]) + exploration + 0.1 * diversity
            )
        selected_parts = np.argsort(-partition_scores)[: cfg["selected_partitions"]]
        selected_parts_set = set(int(part) for part in selected_parts)
        new_selected_parts = len(selected_parts_set - selected_parts_ever)
        selected_parts_ever.update(selected_parts_set)
        unlabeled = np.setdiff1d(all_indices, labeled, assume_unique=False)
        candidate_pool = unlabeled[np.isin(partitions[unlabeled], selected_parts)]
        candidate_cap = max(
            cfg["query_per_round"], cfg["candidate_multiplier"] * cfg["query_per_round"]
        )
        if len(candidate_pool) > candidate_cap:
            candidate_indices = np.sort(
                rng.choice(candidate_pool, size=candidate_cap, replace=False)
            )
        else:
            candidate_indices = candidate_pool
        if len(candidate_indices) < cfg["query_per_round"]:
            extra_pool = np.setdiff1d(unlabeled, candidate_indices, assume_unique=False)
            extra = rng.choice(
                extra_pool,
                size=cfg["query_per_round"] - len(candidate_indices),
                replace=False,
            )
            candidate_indices = np.sort(np.concatenate([candidate_indices, extra]))
        selection_seconds += time.perf_counter() - select_start

        reservoir_lookup = {int(idx): pos for pos, idx in enumerate(reservoir_indices)}
        new_indices = np.asarray(
            [idx for idx in candidate_indices if int(idx) not in reservoir_lookup], dtype=int
        )
        score_start = time.perf_counter()
        new_prediction = predictor.predict(features[new_indices])
        score_seconds += time.perf_counter() - score_start
        neural_inferences += len(new_indices)
        ever_scored.update(int(idx) for idx in new_indices)

        mu = np.empty(len(candidate_indices), dtype=float)
        new_lookup = {int(idx): pos for pos, idx in enumerate(new_indices)}
        for position, idx in enumerate(candidate_indices):
            idx_int = int(idx)
            if idx_int in reservoir_lookup:
                mu[position] = reservoir_prediction.mu[reservoir_lookup[idx_int]]
            else:
                mu[position] = new_prediction.mu[new_lookup[idx_int]]
        batch = candidate_indices[np.argsort(-mu)[: cfg["query_per_round"]]]
        labeled = np.unique(np.concatenate([labeled, batch]))
        sampled_per_partition += np.bincount(partitions[batch], minlength=n_partitions)
        cumulative_hits = int(labels[labeled].sum())
        round_hits.append(cumulative_hits)
        round_rows.append(
            {
                "seed": seed,
                "target": target,
                "method": reported_method,
                "round": round_no + 1,
                "reservoir_inferences": len(reservoir_indices),
                "candidate_inferences": len(new_indices),
                "cumulative_neural_inferences": neural_inferences,
                "queried_total": len(labeled),
                "new_hits": int(labels[batch].sum()),
                "cumulative_hits": cumulative_hits,
                "selected_partition_count": len(selected_parts),
                "new_selected_partitions": new_selected_parts,
                "candidate_pool_size": len(candidate_pool),
                "candidate_set_size": len(candidate_indices),
                "candidate_active_rate_eval": float(labels[candidate_indices].mean()),
            }
        )

    final_indices = np.asarray(sorted(ever_scored | set(int(idx) for idx in labeled)), dtype=int)
    fit_start = time.perf_counter()
    predictor = fit_subset_predictor(
        features, labeled, labels, seed + cfg["rounds"], cfg["ensemble_size"]
    )
    fit_seconds += time.perf_counter() - fit_start
    score_start = time.perf_counter()
    prediction = predictor.predict(features[final_indices])
    score_seconds += time.perf_counter() - score_start
    neural_inferences += len(final_indices)
    final_ranking_method = str(
        cfg.get("final_ranking_method", "rabvs_budgeted_reliable")
    )
    local_ranking = final_ranking(
        final_ranking_method,
        prediction,
        pool.scaffolds[final_indices],
        pool.risk_alert[final_indices],
    )
    ranking = final_indices[local_ranking]
    full_reference = len(labels) * (cfg["rounds"] + 1)
    summary = final_metrics(
        cfg,
        pool,
        features,
        ranking,
        prediction,
        final_indices,
        labeled,
        round_hits,
        neural_inferences,
        full_reference,
        time.perf_counter() - start,
        fit_seconds,
        score_seconds,
        selection_seconds,
        reported_method,
        target,
        seed,
        pool_prepare_seconds,
        initial_hits,
    )
    return round_rows, summary


def run_job(cfg: dict, target: str, seed: int, method: str) -> tuple[list[dict], dict]:
    prepare_start = time.perf_counter()
    pool, features, partitions, n_partitions = prepare_pool(cfg, target, seed)
    pool_prepare_seconds = time.perf_counter() - prepare_start
    if method == "full_greedy":
        return run_full_greedy(cfg, target, seed, pool, features, pool_prepare_seconds)
    if method == "true_budgeted_reliable":
        return run_true_budgeted(
            cfg,
            target,
            seed,
            pool,
            features,
            partitions,
            n_partitions,
            pool_prepare_seconds,
        )
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ensure_dir(cfg["output_dir"])
    jobs = [
        (target, seed, method)
        for target in cfg["targets"]
        for seed in cfg["seeds"]
        for method in cfg["methods"]
    ]
    results = Parallel(n_jobs=cpu_worker_cap(cfg.get("n_jobs")), prefer="processes")(
        delayed(run_job)(cfg, target, seed, method)
        for target, seed, method in tqdm(jobs, desc="True-inference DUD-E")
    )
    round_rows = [row for rows, _ in results for row in rows]
    summaries = [summary for _, summary in results]
    pd.DataFrame(round_rows).to_csv(out_dir / "round_logs.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    print(out_dir)


if __name__ == "__main__":
    main()
