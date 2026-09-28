from __future__ import annotations

import numpy as np


def hit_at(labels: np.ndarray, ranking: np.ndarray, k: int) -> int:
    top = ranking[: min(k, len(ranking))]
    return int(np.asarray(labels)[top].sum())


def recall_at(labels: np.ndarray, ranking: np.ndarray, k: int) -> float:
    positives = max(1, int(np.asarray(labels).sum()))
    return hit_at(labels, ranking, k) / positives


def enrichment_factor(labels: np.ndarray, ranking: np.ndarray, frac: float) -> float:
    labels = np.asarray(labels)
    k = max(1, int(round(len(labels) * frac)))
    top_rate = labels[ranking[:k]].mean()
    base_rate = max(labels.mean(), 1e-12)
    return float(top_rate / base_rate)


def bedroc_score(
    labels: np.ndarray,
    ranking: np.ndarray,
    alpha: float = 20.0,
) -> float:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC

    ordered = [(int(np.asarray(labels)[idx]),) for idx in np.asarray(ranking, dtype=int)]
    value = CalcBEDROC(ordered, 0, alpha)
    return float(np.clip(value, 0.0, 1.0))


def aubc(round_hits: list[int], query_per_round: int) -> float:
    if not round_hits:
        return 0.0
    xs = np.arange(1, len(round_hits) + 1) * query_per_round
    ys = np.asarray(round_hits, dtype=float)
    denom = max(xs[-1] * max(ys[-1], 1.0), 1.0)
    return float(np.trapz(ys, xs) / denom)


def aubc_recall(
    round_hits: list[int],
    query_per_round: int,
    positive_count: int,
    initial_hits: int = 0,
) -> float:
    if not round_hits:
        return 0.0
    xs = np.arange(0, len(round_hits) + 1) * query_per_round
    ys = np.asarray([initial_hits, *round_hits], dtype=float)
    recall = ys / max(1, positive_count)
    return float(np.trapz(recall, xs) / max(1, xs[-1]))


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi)
        if not mask.any():
            continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(y_true)) ** 2))


def unique_scaffolds(scaffolds: np.ndarray, ranking: np.ndarray, k: int) -> int:
    top = ranking[: min(k, len(ranking))]
    return int(len(set(np.asarray(scaffolds)[top].tolist())))



def negative_log_likelihood(y_true: np.ndarray, prob: np.ndarray, eps: float = 1e-7) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(prob) + (1.0 - y_true) * np.log(1.0 - prob)))


def conformal_residual_threshold(
    y_calib: np.ndarray,
    prob_calib: np.ndarray,
    alpha: float = 0.1,
) -> float:
    residual = np.abs(np.asarray(y_calib, dtype=float) - np.asarray(prob_calib, dtype=float))
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        return float("nan")
    order_stat = int(np.ceil((len(residual) + 1) * (1.0 - alpha))) - 1
    order_stat = min(max(order_stat, 0), len(residual) - 1)
    return float(np.partition(residual, order_stat)[order_stat])


def conformal_residual_coverage(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> float:
    if not np.isfinite(threshold):
        return float("nan")
    residual = np.abs(np.asarray(y_true, dtype=float) - np.asarray(prob, dtype=float))
    if len(residual) == 0:
        return float("nan")
    return float(np.mean(residual <= threshold))


def risk_coverage_auc(
    y_true: np.ndarray,
    prob: np.ndarray,
    uncertainty: np.ndarray,
    min_coverage: float = 0.05,
    points: int = 20,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(prob) & np.isfinite(uncertainty)
    if mask.sum() == 0:
        return float("nan")
    y_true = y_true[mask]
    prob = prob[mask]
    uncertainty = uncertainty[mask]
    order = np.argsort(uncertainty, kind="stable")
    residual = np.abs(y_true - prob)[order]
    coverages = np.linspace(min_coverage, 1.0, points)
    risks = []
    for coverage in coverages:
        k = max(1, int(np.ceil(coverage * len(residual))))
        risks.append(float(residual[:k].mean()))
    return float(np.trapz(np.asarray(risks), coverages) / max(1e-12, 1.0 - min_coverage))


def mean_pairwise_tanimoto(
    binary_features: np.ndarray,
    ranking: np.ndarray,
    k: int,
) -> float:
    from scipy.spatial.distance import pdist

    top = np.asarray(ranking, dtype=int)[: min(k, len(ranking))]
    if len(top) < 2:
        return 1.0
    fingerprints = np.asarray(binary_features[top]) > 0
    distances = pdist(fingerprints, metric="jaccard")
    return float(1.0 - distances.mean())
