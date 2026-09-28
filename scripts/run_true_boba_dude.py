from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rabvs.bandit import final_ranking
from rabvs.utils import cpu_worker_cap, ensure_dir
from run_dude_pilot import prepare_pool
from run_true_budgeted_dude import (
    final_metrics,
    fit_subset_predictor,
    initial_labeled_indices,
    partition_members,
)


def initialize_bandit(
    members: list[np.ndarray],
    labeled: np.ndarray,
    labels: np.ndarray,
    prior_pulls: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if prior_pulls < 0:
        raise ValueError("arm_prior_pulls must be non-negative")
    initial_rate = float(labels[labeled].mean()) if len(labeled) else 0.0
    reward_sum = np.full(len(members), initial_rate * prior_pulls, dtype=float)
    pull_count = np.full(len(members), prior_pulls, dtype=float)
    labeled_set = set(int(idx) for idx in labeled)
    for part, indices in enumerate(members):
        observed = np.asarray([idx for idx in indices if int(idx) in labeled_set], dtype=int)
        if len(observed):
            reward_sum[part] += float(labels[observed].mean())
            pull_count[part] += 1.0
    return reward_sum, pull_count


def select_ucb_partitions(
    reward_sum: np.ndarray,
    pull_count: np.ndarray,
    round_no: int,
    count: int,
    coefficient: float,
    rng: np.random.Generator,
) -> np.ndarray:
    empirical_mean = np.divide(
        reward_sum,
        np.maximum(pull_count, 1.0),
    )
    bonus = coefficient * np.sqrt(
        2.0 * np.log(max(2, round_no + 2)) / np.maximum(pull_count, 1.0)
    )
    score = empirical_mean + bonus
    score = np.where(pull_count == 0, 1e6 + rng.random(len(score)), score)
    return np.argsort(-score)[:count]


def run_one(cfg: dict, target: str, seed: int) -> tuple[list[dict], dict]:
    prepare_start = time.perf_counter()
    pool, features, partitions, n_partitions = prepare_pool(cfg, target, seed)
    pool_prepare_seconds = time.perf_counter() - prepare_start

    start = time.perf_counter()
    labels = pool.labels
    all_indices = np.arange(len(labels), dtype=int)
    labeled = initial_labeled_indices(labels, cfg["initial_labeled"], seed, target)
    initial_hits = int(labels[labeled].sum())
    members = partition_members(partitions, n_partitions)
    reward_sum, pull_count = initialize_bandit(
        members,
        labeled,
        labels,
        float(cfg.get("arm_prior_pulls", 0.0)),
    )
    rng = np.random.default_rng(seed + 65537)

    round_hits = []
    round_rows = []
    ever_scored: set[int] = set()
    selected_ever: set[int] = set()
    fit_seconds = score_seconds = selection_seconds = 0.0
    neural_inferences = 0
    method = str(cfg.get("reported_method", "true_boba_risk_ranked"))

    for round_no in range(cfg["rounds"]):
        fit_start = time.perf_counter()
        predictor = fit_subset_predictor(
            features, labeled, labels, seed + round_no, cfg["ensemble_size"]
        )
        fit_seconds += time.perf_counter() - fit_start

        select_start = time.perf_counter()
        selected_parts = select_ucb_partitions(
            reward_sum,
            pull_count,
            round_no,
            cfg["selected_partitions"],
            cfg["ucb_coefficient"],
            rng,
        )
        selected_set = set(int(part) for part in selected_parts)
        new_selected = len(selected_set - selected_ever)
        selected_ever.update(selected_set)
        unlabeled = np.setdiff1d(all_indices, labeled, assume_unique=False)
        candidate_indices = unlabeled[np.isin(partitions[unlabeled], selected_parts)]
        selection_seconds += time.perf_counter() - select_start

        score_start = time.perf_counter()
        prediction = predictor.predict(features[candidate_indices])
        score_seconds += time.perf_counter() - score_start
        neural_inferences += len(candidate_indices)
        ever_scored.update(int(idx) for idx in candidate_indices)

        quota = max(1, int(np.ceil(cfg["query_per_round"] / len(selected_parts))))
        batch_parts = []
        selected_positions: set[int] = set()
        for part in selected_parts:
            positions = np.flatnonzero(partitions[candidate_indices] == part)
            if len(positions) == 0:
                continue
            chosen = positions[np.argsort(-prediction.mu[positions])[:quota]]
            batch_parts.append(candidate_indices[chosen])
            selected_positions.update(int(position) for position in chosen)
        batch = np.unique(np.concatenate(batch_parts)) if batch_parts else np.empty(0, dtype=int)
        if len(batch) < cfg["query_per_round"]:
            remaining_positions = np.asarray(
                [
                    position
                    for position in np.argsort(-prediction.mu)
                    if int(position) not in selected_positions
                ],
                dtype=int,
            )
            fill = candidate_indices[
                remaining_positions[: cfg["query_per_round"] - len(batch)]
            ]
            batch = np.unique(np.concatenate([batch, fill]))
        batch = batch[: cfg["query_per_round"]]

        for part in selected_parts:
            part_batch = batch[partitions[batch] == part]
            reward = float(labels[part_batch].mean()) if len(part_batch) else 0.0
            reward_sum[part] += reward
            pull_count[part] += 1.0

        labeled = np.unique(np.concatenate([labeled, batch]))
        cumulative_hits = int(labels[labeled].sum())
        round_hits.append(cumulative_hits)
        round_rows.append(
            {
                "seed": seed,
                "target": target,
                "method": method,
                "round": round_no + 1,
                "reservoir_inferences": 0,
                "candidate_inferences": len(candidate_indices),
                "cumulative_neural_inferences": neural_inferences,
                "queried_total": len(labeled),
                "new_hits": int(labels[batch].sum()),
                "cumulative_hits": cumulative_hits,
                "selected_partition_count": len(selected_parts),
                "new_selected_partitions": new_selected,
                "candidate_pool_size": len(candidate_indices),
                "candidate_set_size": len(candidate_indices),
                "candidate_active_rate_eval": float(labels[candidate_indices].mean()),
                "mean_selected_arm_reward": float(
                    np.mean(
                        np.divide(
                            reward_sum[selected_parts],
                            np.maximum(pull_count[selected_parts], 1.0),
                        )
                    )
                ),
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
    local_ranking = final_ranking(
        cfg.get("final_ranking_method", "rabvs_budgeted_reliable_no_density"),
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
        method,
        target,
        seed,
        pool_prepare_seconds,
        initial_hits,
    )
    summary["ucb_coefficient"] = float(cfg.get("ucb_coefficient", 1.0))
    summary["arm_prior_pulls"] = float(cfg.get("arm_prior_pulls", 0.0))
    summary["n_partitions"] = int(n_partitions)
    return round_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ensure_dir(cfg["output_dir"])
    jobs = [(target, seed) for target in cfg["targets"] for seed in cfg["seeds"]]
    results = Parallel(n_jobs=cpu_worker_cap(cfg.get("n_jobs")), prefer="processes")(
        delayed(run_one)(cfg, target, seed)
        for target, seed in tqdm(jobs, desc="True BOBA DUD-E")
    )
    round_rows = [row for rows, _ in results for row in rows]
    summaries = [summary for _, summary in results]
    pd.DataFrame(round_rows).to_csv(out_dir / "round_logs.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    print(out_dir)


if __name__ == "__main__":
    main()
