from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rabvs.bandit import final_ranking, mean_pairwise_distance, predict_with_uncertainty, select_batch
from rabvs.calibration import calibration_diagnostics
from rabvs.lit_pcba import load_lit_pcba_target
from rabvs.metrics import (
    aubc,
    bedroc_score,
    brier_score,
    enrichment_factor,
    expected_calibration_error,
    negative_log_likelihood,
    hit_at,
    mean_pairwise_tanimoto,
    recall_at,
    unique_scaffolds,
)
from rabvs.utils import cpu_worker_cap, ensure_dir


def prepare_pool(cfg: dict, target: str, seed: int):
    pool = load_lit_pcba_target(
        data_root=cfg["data_root"],
        target=target,
        split=cfg.get("split", "V"),
        max_inactives=cfg.get("max_inactives"),
        seed=seed,
    )
    n_components = min(cfg["n_components"], pool.features.shape[1] - 1, pool.features.shape[0] - 1)
    reduced = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(pool.features)
    k = min(cfg["n_partitions"], max(2, pool.features.shape[0] // 25))
    partitions = MiniBatchKMeans(
        n_clusters=k,
        batch_size=4096,
        n_init=3,
        random_state=seed,
    ).fit_predict(reduced)
    return pool, reduced.astype(np.float32), partitions.astype(int), k


def stable_target_offset(target: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(target))


def model_kind_for_method(method: str, cfg: dict) -> str:
    if method.startswith("rf_"):
        return "rf"
    if method.startswith("et_"):
        return "et"
    if method == "knn_similarity":
        return "knn"
    return str(cfg.get("model_kind", "logreg"))


def run_one_prepared(
    cfg: dict,
    target: str,
    seed: int,
    method: str,
    pool,
    features: np.ndarray,
    partitions: np.ndarray,
    n_partitions: int,
) -> tuple[list[dict], dict]:
    labels = pool.labels
    rng = np.random.default_rng(seed + stable_target_offset(target))

    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if "initial_positive_count" in cfg:
        init_pos_n = min(len(positives), max(0, int(cfg["initial_positive_count"])))
    else:
        init_pos_n = min(len(positives), max(1, cfg["initial_labeled"] // 32))
    init_neg_n = min(len(negatives), cfg["initial_labeled"] - init_pos_n)
    init_pos = rng.choice(positives, size=init_pos_n, replace=False)
    init_neg = rng.choice(negatives, size=init_neg_n, replace=False)
    labeled = np.unique(np.concatenate([init_pos, init_neg]))
    initial_scaffolds = set(pool.scaffolds[labeled].tolist())
    sampled_per_partition = np.bincount(partitions[labeled], minlength=n_partitions).astype(float)

    round_rows = []
    round_hits = []
    last_pred = None
    for round_no in range(cfg["rounds"]):
        model_kind = model_kind_for_method(method, cfg)
        unlabeled = np.setdiff1d(np.arange(len(labels)), labeled, assume_unique=False)
        pred = predict_with_uncertainty(
            features,
            labeled,
            labels,
            seed=seed + round_no,
            retrieval_blend=float(cfg.get("retrieval_round_blend", cfg.get("retrieval_blend", 0.0))),
            ensemble_size=int(cfg.get("ensemble_size", 5)),
            model_kind=model_kind,
        )
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
        cumulative_hits = int(labels[labeled].sum())
        round_hits.append(cumulative_hits)
        round_rows.append(
            {
                "seed": seed,
                "target": target,
                "method": method,
                "round": round_no + 1,
                "pool_size": int(len(labels)),
                "active_count": int(labels.sum()),
                "queried_total": int(len(labeled)),
                "new_hits": int(labels[batch].sum()),
                "cumulative_hits": cumulative_hits,
                "risk_alert_rate_queried": float(pool.risk_alert[labeled].mean()),
            }
        )

    if last_pred is None:
        raise RuntimeError("No predictions produced")
    final_blend = float(cfg.get("retrieval_final_blend", cfg.get("retrieval_blend", 0.0)))
    round_blend = float(cfg.get("retrieval_round_blend", cfg.get("retrieval_blend", 0.0)))
    if final_blend != round_blend:
        model_kind = model_kind_for_method(method, cfg)
        last_pred = predict_with_uncertainty(
            features,
            labeled,
            labels,
            seed=seed + cfg["rounds"],
            retrieval_blend=final_blend,
            ensemble_size=int(cfg.get("ensemble_size", 5)),
            model_kind=model_kind,
        )
    if method == "random":
        rank_rng = np.random.default_rng(seed + stable_target_offset(target) + 99991)
        ranking = rank_rng.permutation(len(labels))
    else:
        ranking = final_ranking(method, last_pred, pool.scaffolds, pool.risk_alert)
    topk_metrics = {}
    for k in cfg["final_k"]:
        topk_metrics[f"Hit@{k}"] = hit_at(labels, ranking, k)
        topk_metrics[f"Recall@{k}"] = recall_at(labels, ranking, k)
        topk_metrics[f"unique_scaffold_count@{k}"] = unique_scaffolds(pool.scaffolds, ranking, k)

    top1000 = ranking[: min(1000, len(ranking))]
    unseen_mask = np.asarray([scaf not in initial_scaffolds for scaf in pool.scaffolds[top1000]])
    unseen_hits = labels[top1000][unseen_mask]
    unseen_initial_scaffold_hit_rate = float(unseen_hits.mean()) if len(unseen_hits) else 0.0

    high_score_mask = last_pred.mu >= np.quantile(last_pred.mu, 0.9)
    high_score_idx = np.flatnonzero(high_score_mask)
    if len(high_score_idx):
        final_uncertainty = last_pred.uncertainty
        high_unc = high_score_idx[
            final_uncertainty[high_score_idx] >= np.quantile(final_uncertainty[high_score_idx], 0.5)
        ]
        low_unc = high_score_idx[
            final_uncertainty[high_score_idx] < np.quantile(final_uncertainty[high_score_idx], 0.5)
        ]
        high_score_high_unc_hit_rate = float(labels[high_unc].mean()) if len(high_unc) else 0.0
        high_score_low_unc_hit_rate = float(labels[low_unc].mean()) if len(low_unc) else 0.0
    else:
        high_score_high_unc_hit_rate = 0.0
        high_score_low_unc_hit_rate = 0.0
    false_confidence_gap = high_score_low_unc_hit_rate - high_score_high_unc_hit_rate
    calibration_result = calibration_diagnostics(
        features,
        labels,
        labeled,
        seed=seed + cfg["rounds"] + 1009,
        retrieval_blend=final_blend,
        ensemble_size=int(cfg.get("ensemble_size", 5)),
        include_reliability_bins=bool(cfg.get("export_calibration_bins", False)),
    )
    if cfg.get("export_calibration_bins", False):
        calibration, calibration_bins = calibration_result
        calibration_bins = [
            {"seed": seed, "target": target, "method": method, **row}
            for row in calibration_bins
        ]
    else:
        calibration = calibration_result
        calibration_bins = []

    full_inference_budget = len(labels) * max(1, cfg["rounds"])
    if method in {
        "greedy_surrogate",
        "uncertainty_sampling",
        "rabvs_no_partition_reliable",
        "rf_greedy",
        "rf_uncertainty",
        "rf_ucb",
        "et_greedy",
        "et_uncertainty",
        "et_ucb",
        "knn_similarity",
    }:
        neural_inferences = full_inference_budget
    else:
        avg_partition_size = len(labels) / n_partitions
        reservoir_budget = n_partitions * min(256, max(1, int(round(0.05 * avg_partition_size))))
        neural_inferences = cfg["initial_labeled"] + reservoir_budget + cfg["rounds"] * cfg["query_per_round"]

    summary = {
        "seed": seed,
        "target": target,
        "method": method,
        "ensemble_size": int(cfg.get("ensemble_size", 5)),
        "pool_size": int(len(labels)),
        "active_count": int(labels.sum()),
        "active_ratio_pool": float(labels.mean()),
        "queried_total": int(len(labeled)),
        "queried_hits": int(labels[labeled].sum()),
        "AUBC": aubc(round_hits, cfg["query_per_round"]),
        "ECE": expected_calibration_error(labels, last_pred.mu),
        "Brier": brier_score(labels, last_pred.mu),
        "NLL": negative_log_likelihood(labels, last_pred.mu),
        "EF_0.1pct": enrichment_factor(labels, ranking, 0.001),
        "EF_1pct": enrichment_factor(labels, ranking, 0.01),
        "BEDROC_alpha20": bedroc_score(labels, ranking, alpha=20.0),
        "neural_inferences": int(neural_inferences),
        "inference_saved_ratio": float(1.0 - neural_inferences / full_inference_budget),
        "hits_per_1000_inferences": float(topk_metrics["Hit@1000"] / max(1, neural_inferences) * 1000.0),
        "mean_pairwise_cosine_distance_at1000": mean_pairwise_distance(features, ranking, 1000),
        "mean_pairwise_tanimoto_at1000": mean_pairwise_tanimoto(pool.features, ranking, 1000),
        "PAINS_Brenk_free_ratio_at1000": float(1.0 - pool.risk_alert[ranking[:1000]].mean()),
        "PAINS_free_ratio_at1000": float(1.0 - pool.pains_alert[ranking[:1000]].mean()),
        "Brenk_free_ratio_at1000": float(1.0 - pool.brenk_alert[ranking[:1000]].mean()),
        "unseen_initial_scaffold_hit_rate_at1000": unseen_initial_scaffold_hit_rate,
        "high_score_high_unc_hit_rate": high_score_high_unc_hit_rate,
        "high_score_low_unc_hit_rate": high_score_low_unc_hit_rate,
        "false_confidence_gap": false_confidence_gap,
        **calibration,
        **topk_metrics,
    }
    return round_rows, summary, calibration_bins


def run_group(cfg: dict, target: str, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    pool, features, partitions, n_partitions = prepare_pool(cfg, target, seed)
    all_rounds: list[dict] = []
    summaries: list[dict] = []
    all_calibration_bins: list[dict] = []
    for method in cfg["methods"]:
        rows, summary, calibration_bins = run_one_prepared(cfg, target, seed, method, pool, features, partitions, n_partitions)
        all_rounds.extend(rows)
        summaries.append(summary)
        all_calibration_bins.extend(calibration_bins)
    return all_rounds, summaries, all_calibration_bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ensure_dir(cfg["output_dir"])

    jobs = [(target, seed) for target in cfg["targets"] for seed in cfg["seeds"]]
    results = Parallel(n_jobs=cpu_worker_cap(cfg.get("n_jobs")), prefer="processes")(
        delayed(run_group)(cfg, target, seed)
        for target, seed in tqdm(jobs, desc="LIT-PCBA pilot")
    )
    round_rows = [row for rows, _, _ in results for row in rows]
    summaries = [summary for _, group_summaries, _ in results for summary in group_summaries]
    calibration_bins = [row for _, _, group_bins in results for row in group_bins]
    pd.DataFrame(round_rows).to_csv(out_dir / "round_logs.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    if cfg.get("export_calibration_bins", False):
        pd.DataFrame(calibration_bins).to_csv(out_dir / "calibration_bins.csv", index=False)
    print(out_dir)


if __name__ == "__main__":
    main()
