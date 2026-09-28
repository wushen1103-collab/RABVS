from __future__ import annotations

import numpy as np

from .bandit import predict_with_uncertainty
from .metrics import (
    brier_score,
    conformal_residual_coverage,
    conformal_residual_threshold,
    expected_calibration_error,
    negative_log_likelihood,
    risk_coverage_auc,
)


def _stratified_calibration_split(
    labeled_idx: np.ndarray,
    labels: np.ndarray,
    seed: int,
    fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labeled_idx = np.asarray(labeled_idx, dtype=int)
    calib_parts = []
    for value in (0, 1):
        class_idx = labeled_idx[labels[labeled_idx] == value]
        if len(class_idx) < 2:
            continue
        calib_n = min(len(class_idx) - 1, max(1, int(round(fraction * len(class_idx)))))
        calib_parts.append(rng.choice(class_idx, size=calib_n, replace=False))
    if not calib_parts:
        return labeled_idx.copy(), np.empty(0, dtype=int)
    calib_idx = np.unique(np.concatenate(calib_parts).astype(int))
    train_idx = np.setdiff1d(labeled_idx, calib_idx, assume_unique=False)
    return train_idx, calib_idx


def _reliability_bin_rows(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    bin_ids = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        empirical_rate = float(labels[mask].mean())
        rows.append(
            {
                "bin": bin_index + 1,
                "bin_lower": bin_index / n_bins,
                "bin_upper": (bin_index + 1) / n_bins,
                "n": count,
                "mean_predicted_probability": float(probabilities[mask].mean()),
                "empirical_active_rate": empirical_rate,
                "binomial_se": float(np.sqrt(empirical_rate * (1.0 - empirical_rate) / count)),
            }
        )
    return rows


def calibration_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    labeled_idx: np.ndarray,
    seed: int,
    retrieval_blend: float = 0.0,
    ensemble_size: int = 5,
    alpha: float = 0.1,
    include_reliability_bins: bool = False,
) -> dict[str, float | int] | tuple[dict[str, float | int], list[dict[str, float | int]]]:
    labels = np.asarray(labels)
    labeled_idx = np.asarray(labeled_idx, dtype=int)
    train_idx, calib_idx = _stratified_calibration_split(labeled_idx, labels, seed=seed)
    all_idx = np.arange(len(labels), dtype=int)
    test_idx = np.setdiff1d(all_idx, labeled_idx, assume_unique=False)

    result: dict[str, float | int] = {
        "calib_train_size": int(len(train_idx)),
        "calib_calibration_size": int(len(calib_idx)),
        "calib_test_size": int(len(test_idx)),
        "calib_train_positive_count": int(labels[train_idx].sum()) if len(train_idx) else 0,
        "calib_calibration_positive_count": int(labels[calib_idx].sum()) if len(calib_idx) else 0,
        "conformal_residual_q90": float("nan"),
        "conformal_coverage90": float("nan"),
        "risk_coverage_auc": float("nan"),
        "calib_ECE_test": float("nan"),
        "calib_Brier_test": float("nan"),
        "calib_NLL_test": float("nan"),
    }
    if len(calib_idx) == 0 or len(test_idx) == 0 or len(np.unique(labels[train_idx])) < 2:
        return (result, []) if include_reliability_bins else result

    pred = predict_with_uncertainty(
        features,
        train_idx,
        labels,
        seed=seed,
        retrieval_blend=retrieval_blend,
        ensemble_size=ensemble_size,
    )
    threshold = conformal_residual_threshold(labels[calib_idx], pred.mu[calib_idx], alpha=alpha)
    result.update(
        {
            "conformal_residual_q90": threshold,
            "conformal_coverage90": conformal_residual_coverage(labels[test_idx], pred.mu[test_idx], threshold),
            "risk_coverage_auc": risk_coverage_auc(labels[test_idx], pred.mu[test_idx], pred.uncertainty[test_idx]),
            "calib_ECE_test": expected_calibration_error(labels[test_idx], pred.mu[test_idx]),
            "calib_Brier_test": brier_score(labels[test_idx], pred.mu[test_idx]),
            "calib_NLL_test": negative_log_likelihood(labels[test_idx], pred.mu[test_idx]),
        }
    )
    if not include_reliability_bins:
        return result
    return result, _reliability_bin_rows(labels[test_idx], pred.mu[test_idx])
