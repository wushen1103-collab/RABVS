from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from .utils import zscore


BOBA_ALLOCATION_METHODS = {
    "boba_ucb",
    "boba_risk_final",
    "rabvs_budgeted_reliable",
    "rabvs_budgeted_reliable_q50",
    "rabvs_budgeted_reliable_q90",
    "rabvs_budgeted_reliable_no_alert",
    "rabvs_budgeted_reliable_no_novelty",
    "rabvs_budgeted_reliable_no_density",
    "rabvs_budgeted_no_ucb_soft_reliable",
    "rabvs_budgeted_no_ucb_hit_guard",
    "rabvs_budgeted_no_ucb_pareto_rrf",
    "rabvs_budgeted_no_ucb_pareto_rrf_clean",
    "rabvs_budgeted_no_ucb_pareto_rrf_novel",
    "rabvs_budgeted_no_ucb_pareto_rrf_mu_only",
    "rabvs_budgeted_no_ucb_pareto_rrf_no_clean",
    "rabvs_budgeted_no_ucb_pareto_rrf_no_conf",
    "rabvs_budgeted_no_ucb_pareto_rrf_no_novel",
    "rabvs_budgeted_no_ucb_pareto_rrf_balanced",
    "rabvs_budgeted_no_ucb_scaffaug_rerank",
    "rabvs_budgeted_no_ucb",
    "rabvs_budgeted_no_ucb_q90",
    "rabvs_budgeted_no_ucb_no_density",
    "rabvs_budgeted_decay_ucb",
    "rabvs_budgeted_decay_no_density",
    "rabvs_budgeted_elite_decay",
    "rabvs_budgeted_late_exploit",
    "rabvs_budgeted_clean_acq",
    "rabvs_budgeted_random_parts",
}

NO_PARTITION_METHOD = "rabvs_no_partition_reliable"


@dataclass
class PredictionBundle:
    mu: np.ndarray
    u_epi: np.ndarray
    u_conf: np.ndarray
    density_penalty: np.ndarray

    @property
    def uncertainty(self) -> np.ndarray:
        return zscore(self.u_epi) + zscore(self.u_conf)


def _fit_one_model(x: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(x, y)
    return model


def predict_with_uncertainty(
    features: np.ndarray,
    labeled_idx: np.ndarray,
    labels: np.ndarray,
    seed: int,
    ensemble_size: int = 5,
    retrieval_blend: float = 0.0,
    model_kind: str = "logreg",
) -> PredictionBundle:
    rng = np.random.default_rng(seed)
    x_labeled = features[labeled_idx]
    y_labeled = labels[labeled_idx]

    if len(np.unique(y_labeled)) < 2:
        base = np.full(len(features), y_labeled.mean() if len(y_labeled) else 0.01)
        return PredictionBundle(base, np.ones_like(base), np.ones_like(base), np.ones_like(base))

    if model_kind == "rf":
        n_estimators = max(32, 16 * max(1, ensemble_size))
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=1,
            random_state=seed,
        )
        model.fit(x_labeled, y_labeled)
        class_index = int(np.flatnonzero(model.classes_ == 1)[0])
        pred = np.vstack(
            [estimator.predict_proba(features)[:, class_index] for estimator in model.estimators_]
        )
        mu = pred.mean(axis=0)
        u_epi = pred.var(axis=0)
    elif model_kind == "et":
        n_estimators = max(64, 16 * max(1, ensemble_size))
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=1,
            random_state=seed,
        )
        model.fit(x_labeled, y_labeled)
        class_index = int(np.flatnonzero(model.classes_ == 1)[0])
        pred = np.vstack(
            [estimator.predict_proba(features)[:, class_index] for estimator in model.estimators_]
        )
        mu = pred.mean(axis=0)
        u_epi = pred.var(axis=0)
    elif model_kind == "knn":
        pos_idx = labeled_idx[y_labeled == 1]
        neg_idx = labeled_idx[y_labeled == 0]
        pos_nn = NearestNeighbors(n_neighbors=min(5, len(pos_idx))).fit(features[pos_idx])
        neg_nn = NearestNeighbors(n_neighbors=min(5, len(neg_idx))).fit(features[neg_idx])
        pos_dist, _ = pos_nn.kneighbors(features)
        neg_dist, _ = neg_nn.kneighbors(features)
        raw_score = zscore(-pos_dist.mean(axis=1)) - zscore(-neg_dist.mean(axis=1))
        mu = 1.0 / (1.0 + np.exp(-raw_score))
        u_epi = 1.0 / (np.abs(raw_score) + 1e-3)
    elif model_kind == "logreg":
        probs = []
        pos_local = np.flatnonzero(y_labeled == 1)
        neg_local = np.flatnonzero(y_labeled == 0)
        for member in range(ensemble_size):
            if len(pos_local) and len(neg_local):
                pos_boot = rng.choice(pos_local, size=max(1, len(pos_local)), replace=True)
                neg_boot = rng.choice(neg_local, size=max(1, len(neg_local)), replace=True)
                boot = np.concatenate([pos_boot, neg_boot])
                rng.shuffle(boot)
            else:
                boot = rng.choice(len(labeled_idx), size=len(labeled_idx), replace=True)
            model = _fit_one_model(x_labeled[boot], y_labeled[boot], seed + member)
            probs.append(model.predict_proba(features)[:, 1])
        pred = np.vstack(probs)
        mu = pred.mean(axis=0)
        u_epi = pred.var(axis=0)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    nn = NearestNeighbors(n_neighbors=min(5, len(labeled_idx)))
    nn.fit(x_labeled)
    dists, _ = nn.kneighbors(features)
    density_penalty = dists.mean(axis=1)
    u_conf = density_penalty.copy()

    if retrieval_blend > 0:
        pos_idx = labeled_idx[y_labeled == 1]
        neg_idx = labeled_idx[y_labeled == 0]
        if len(pos_idx) and len(neg_idx):
            pos_nn = NearestNeighbors(n_neighbors=1).fit(features[pos_idx])
            neg_nn = NearestNeighbors(n_neighbors=1).fit(features[neg_idx])
            pos_dist, _ = pos_nn.kneighbors(features)
            neg_dist, _ = neg_nn.kneighbors(features)
            retrieval_score = zscore(-pos_dist[:, 0]) - 0.5 * zscore(-neg_dist[:, 0])
            blended = zscore(mu) + retrieval_blend * retrieval_score
            mu = 1.0 / (1.0 + np.exp(-blended))

    return PredictionBundle(mu=mu, u_epi=u_epi, u_conf=u_conf, density_penalty=density_penalty)


def topmean(values: np.ndarray, q: float = 0.05) -> float:
    if len(values) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(values) * q)))
    return float(np.partition(values, -k)[-k:].mean())


def scaffold_diversity(scaffolds: np.ndarray) -> float:
    if len(scaffolds) == 0:
        return 0.0
    return len(np.unique(scaffolds)) / len(scaffolds)


def select_batch(
    method: str,
    pred: PredictionBundle,
    labels_seen: np.ndarray,
    unlabeled_idx: np.ndarray,
    partitions: np.ndarray,
    scaffolds: np.ndarray,
    risk_alert: np.ndarray,
    sampled_per_partition: np.ndarray,
    query_size: int,
    round_no: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919 * round_no)
    if len(unlabeled_idx) <= query_size:
        return unlabeled_idx.copy()

    if method == "random":
        return rng.choice(unlabeled_idx, size=query_size, replace=False)

    if method == "greedy_surrogate":
        score = pred.mu[unlabeled_idx]
        return unlabeled_idx[np.argsort(-score)[:query_size]]

    if method == "uncertainty_sampling":
        score = zscore(pred.u_epi[unlabeled_idx]) + zscore(pred.u_conf[unlabeled_idx])
        return unlabeled_idx[np.argsort(-score)[:query_size]]

    if method in {"rf_greedy", "et_greedy", "knn_similarity"}:
        score = pred.mu[unlabeled_idx]
        return unlabeled_idx[np.argsort(-score)[:query_size]]

    if method in {"rf_uncertainty", "et_uncertainty"}:
        score = zscore(pred.u_epi[unlabeled_idx]) + zscore(pred.u_conf[unlabeled_idx])
        return unlabeled_idx[np.argsort(-score)[:query_size]]

    if method in {"rf_ucb", "et_ucb"}:
        score = pred.mu[unlabeled_idx] + 0.5 * zscore(pred.u_epi[unlabeled_idx])
        return unlabeled_idx[np.argsort(-score)[:query_size]]

    if method == NO_PARTITION_METHOD:
        score = pred.mu[unlabeled_idx]
        return unlabeled_idx[np.argsort(-score)[:query_size]]

    if method in BOBA_ALLOCATION_METHODS or method.startswith("rabvs"):
        selected_parts = _select_partitions(
            method=method,
            pred=pred,
            candidate_idx=unlabeled_idx,
            partitions=partitions,
            scaffolds=scaffolds,
            risk_alert=risk_alert,
            sampled_per_partition=sampled_per_partition,
            round_no=round_no,
            seed=seed,
            max_parts=max(1, min(32, query_size // 8)),
        )
        mask = np.isin(partitions[unlabeled_idx], selected_parts)
        candidates = unlabeled_idx[mask]
        if len(candidates) < query_size:
            extra = np.setdiff1d(unlabeled_idx, candidates, assume_unique=False)
            fill = rng.choice(extra, size=query_size - len(candidates), replace=False)
            candidates = np.concatenate([candidates, fill])
        score = _molecule_score(method, pred, candidates, risk_alert)
        return candidates[np.argsort(-score)[:query_size]]

    raise ValueError(f"Unknown method: {method}")


def _select_partitions(
    method: str,
    pred: PredictionBundle,
    candidate_idx: np.ndarray,
    partitions: np.ndarray,
    scaffolds: np.ndarray,
    risk_alert: np.ndarray,
    sampled_per_partition: np.ndarray,
    round_no: int,
    seed: int,
    max_parts: int,
) -> np.ndarray:
    unique_parts = np.unique(partitions[candidate_idx])
    if method == "rabvs_budgeted_random_parts":
        rng = np.random.default_rng(seed + 104729 * (round_no + 1))
        size = min(max_parts, len(unique_parts))
        return rng.choice(unique_parts, size=size, replace=False)

    scores = []
    t = max(2, round_no + 2)
    for part in unique_parts:
        idx = candidate_idx[partitions[candidate_idx] == part]
        explore = np.sqrt(2.0 * np.log(t) / max(1.0, sampled_per_partition[part]))
        diversity = scaffold_diversity(scaffolds[idx])
        if method in BOBA_ALLOCATION_METHODS:
            elite_q = 0.01 if method == "rabvs_budgeted_elite_decay" else 0.05
            if method.startswith("rabvs_budgeted_no_ucb"):
                explore_term = 0.0
            elif method in {"rabvs_budgeted_decay_ucb", "rabvs_budgeted_decay_no_density", "rabvs_budgeted_elite_decay"}:
                explore_term = explore / np.sqrt(round_no + 1.0)
            elif method == "rabvs_budgeted_late_exploit":
                explore_term = explore if round_no < 2 else 0.0
            else:
                explore_term = explore
            value = topmean(pred.mu[idx], q=elite_q) + explore_term + 0.1 * diversity
        else:
            u_coef = 1.5
            if method == "rabvs_alloc_u05_mean":
                u_coef = 0.5
            elif method in {
                "rabvs_alloc_u0_mean",
                "rabvs_alloc_u0_risk",
                "rabvs_hit",
                "rabvs_reliable",
                "rabvs_reliable_combo",
                "rabvs_reliable_q90",
            }:
                u_coef = 0.0
            risk = 0.5 * risk_alert[idx].mean() + 0.5 * zscore(pred.u_conf[idx]).mean()
            value = (
                topmean(pred.mu[idx])
                + u_coef * pred.uncertainty[idx].mean()
                + explore
                + 0.1 * diversity
                - 0.5 * risk
            )
        scores.append(value)
    order = np.argsort(-np.asarray(scores))[:max_parts]
    return unique_parts[order]


def _molecule_score(
    method: str,
    pred: PredictionBundle,
    candidates: np.ndarray,
    risk_alert: np.ndarray,
) -> np.ndarray:
    if method in BOBA_ALLOCATION_METHODS:
        score = pred.mu[candidates]
        if method == "rabvs_budgeted_clean_acq":
            score = score - 0.05 * risk_alert[candidates]
        return score
    conf_weight = 0.7
    if method in {"rabvs_conf03", "rabvs_alloc_u05_mean"}:
        conf_weight = 0.3
    if method in {
        "rabvs_no_conf",
        "rabvs_alloc_u0_mean",
        "rabvs_alloc_u0_risk",
        "rabvs_hit",
        "rabvs_reliable",
        "rabvs_reliable_combo",
        "rabvs_reliable_q90",
    }:
        conf_weight = 0.0
    return (
        pred.mu[candidates]
        + 1.0 * zscore(pred.u_epi[candidates])
        - conf_weight * zscore(pred.u_conf[candidates])
        - 0.2 * risk_alert[candidates]
        - 0.1 * zscore(pred.density_penalty[candidates])
    )


def final_ranking(
    method: str,
    pred: PredictionBundle,
    scaffolds: np.ndarray,
    risk_alert: np.ndarray,
) -> np.ndarray:
    if method == "random":
        return np.arange(len(scaffolds))
    counts = np.bincount(scaffolds)
    novelty = 1.0 / np.maximum(counts[scaffolds], 1)
    if method == NO_PARTITION_METHOD:
        score = _risk_constrained_score("rabvs_budgeted_no_ucb", pred, novelty, risk_alert, scaffolds)
    elif method in BOBA_ALLOCATION_METHODS - {"boba_ucb"}:
        score = _risk_constrained_score(method, pred, novelty, risk_alert, scaffolds)
    elif method.startswith(("rf_", "et_")) or method == "knn_similarity":
        score = pred.mu
    elif method.startswith("rabvs"):
        if method == "rabvs_mean_final":
            score = pred.mu
        elif method in {"rabvs_alloc_u05_mean", "rabvs_alloc_u0_mean", "rabvs_hit"}:
            score = pred.mu
        elif method in {"rabvs_risk_constrained", "rabvs_alloc_u0_risk", "rabvs_reliable"}:
            risk_threshold = np.quantile(pred.u_conf, 0.75)
            score = pred.mu + 0.05 * zscore(novelty) - 0.1 * risk_alert
            score = np.where(pred.u_conf <= risk_threshold, score, score - 1.0)
        elif method == "rabvs_reliable_q90":
            risk_threshold = np.quantile(pred.u_conf, 0.90)
            score = pred.mu + 0.05 * zscore(novelty) - 0.1 * risk_alert
            score = np.where(pred.u_conf <= risk_threshold, score, score - 1.0)
        elif method == "rabvs_reliable_combo":
            risk_signal = pred.uncertainty
            risk_threshold = np.quantile(risk_signal, 0.75)
            score = pred.mu + 0.05 * zscore(novelty) - 0.1 * risk_alert
            score = np.where(risk_signal <= risk_threshold, score, score - 1.0)
        elif method == "rabvs_conf03":
            score = pred.mu - 0.3 * zscore(pred.u_conf) + 0.1 * zscore(novelty) - 0.1 * risk_alert
        elif method == "rabvs_no_conf":
            score = pred.mu + 0.1 * zscore(novelty) - 0.1 * risk_alert
        else:
            score = pred.mu - 0.7 * zscore(pred.u_conf) + 0.1 * zscore(novelty) - 0.1 * risk_alert
    else:
        score = pred.mu
    return np.argsort(-score)


def _risk_constrained_score(
    method: str,
    pred: PredictionBundle,
    novelty: np.ndarray,
    risk_alert: np.ndarray,
    scaffolds: np.ndarray,
) -> np.ndarray:
    risk_quantile = 0.75
    if method == "rabvs_budgeted_no_ucb_soft_reliable":
        return (
            pred.mu
            + 0.02 * zscore(novelty)
            - 0.05 * risk_alert
            - 0.03 * zscore(pred.u_conf)
        )
    if method == "rabvs_budgeted_no_ucb_hit_guard":
        protected = pred.mu >= np.quantile(pred.mu, 0.90)
        clean_score = pred.mu + 0.03 * zscore(novelty) - 0.08 * risk_alert - 0.02 * zscore(pred.u_conf)
        hit_score = pred.mu - 0.01 * risk_alert
        return np.where(protected, hit_score, clean_score)
    if method == "rabvs_budgeted_no_ucb_pareto_rrf":
        return _pareto_rrf_score(pred, novelty, risk_alert)
    if method == "rabvs_budgeted_no_ucb_pareto_rrf_clean":
        return _pareto_rrf_score(pred, novelty, risk_alert, clean_weight=0.35, conf_weight=0.12, novelty_weight=0.08)
    if method == "rabvs_budgeted_no_ucb_pareto_rrf_novel":
        return _pareto_rrf_score(pred, novelty, risk_alert, clean_weight=0.18, conf_weight=0.12, novelty_weight=0.25)
    if method == "rabvs_budgeted_no_ucb_pareto_rrf_mu_only":
        return _pareto_rrf_score(pred, novelty, risk_alert, clean_weight=0.0, conf_weight=0.0, novelty_weight=0.0)
    if method == "rabvs_budgeted_no_ucb_pareto_rrf_no_clean":
        return _pareto_rrf_score(pred, novelty, risk_alert, clean_weight=0.0, conf_weight=0.12, novelty_weight=0.25)
    if method == "rabvs_budgeted_no_ucb_pareto_rrf_no_conf":
        return _pareto_rrf_score(pred, novelty, risk_alert, clean_weight=0.18, conf_weight=0.0, novelty_weight=0.25)
    if method == "rabvs_budgeted_no_ucb_pareto_rrf_no_novel":
        return _pareto_rrf_score(pred, novelty, risk_alert, clean_weight=0.18, conf_weight=0.12, novelty_weight=0.0)
    if method == "rabvs_budgeted_no_ucb_pareto_rrf_balanced":
        return _pareto_rrf_score(pred, novelty, risk_alert, clean_weight=0.30, conf_weight=0.12, novelty_weight=0.18)
    if method == "rabvs_budgeted_no_ucb_scaffaug_rerank":
        base = pred.mu + 0.02 * zscore(novelty) - 0.04 * risk_alert
        return _scaffold_repeat_adjusted_score(base, scaffolds, repeat_penalty=0.025)

    if method == "rabvs_budgeted_reliable_q50":
        risk_quantile = 0.50
    elif method in {"rabvs_budgeted_reliable_q90", "rabvs_budgeted_no_ucb_q90"}:
        risk_quantile = 0.90

    novelty_weight = 0.0 if method == "rabvs_budgeted_reliable_no_novelty" else 0.05
    alert_weight = 0.0 if method == "rabvs_budgeted_reliable_no_alert" else 0.1
    score = pred.mu + novelty_weight * zscore(novelty) - alert_weight * risk_alert
    no_density_methods = {
        "rabvs_budgeted_reliable_no_density",
        "rabvs_budgeted_no_ucb_no_density",
        "rabvs_budgeted_decay_no_density",
    }
    if method not in no_density_methods:
        risk_threshold = np.quantile(pred.u_conf, risk_quantile)
        score = np.where(pred.u_conf <= risk_threshold, score, score - 1.0)
    return score


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values), kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def _rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def _pareto_rrf_score(
    pred: PredictionBundle,
    novelty: np.ndarray,
    risk_alert: np.ndarray,
    clean_weight: float = 0.18,
    conf_weight: float = 0.12,
    novelty_weight: float = 0.08,
) -> np.ndarray:
    k = 60.0
    clean_rank = _rank_asc(risk_alert)
    conf_rank = _rank_asc(pred.u_conf)
    return (
        1.00 / (k + _rank_desc(pred.mu))
        + clean_weight / (k + clean_rank)
        + conf_weight / (k + conf_rank)
        + novelty_weight / (k + _rank_desc(novelty))
    )


def _scaffold_repeat_adjusted_score(
    base_score: np.ndarray,
    scaffolds: np.ndarray,
    repeat_penalty: float,
) -> np.ndarray:
    order = np.argsort(-base_score, kind="stable")
    adjusted = np.empty_like(base_score, dtype=float)
    seen: dict[int, int] = {}
    for idx in order:
        scaffold = int(scaffolds[idx])
        repeat_rank = seen.get(scaffold, 0)
        adjusted[idx] = base_score[idx] - repeat_penalty * np.log1p(repeat_rank)
        seen[scaffold] = repeat_rank + 1
    return adjusted


def mean_pairwise_distance(features: np.ndarray, ranking: np.ndarray, k: int) -> float:
    top = ranking[: min(k, len(ranking))]
    if len(top) < 2:
        return 0.0
    d = pairwise_distances(features[top], metric="cosine")
    tri = d[np.triu_indices_from(d, k=1)]
    return float(tri.mean())
