from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticPool:
    features: np.ndarray
    labels: np.ndarray
    partitions: np.ndarray
    scaffolds: np.ndarray
    risk_alert: np.ndarray
    target_ids: list[str]


def make_synthetic_pool(
    pool_size: int,
    n_targets: int,
    n_partitions: int,
    n_features: int,
    n_scaffolds: int,
    seed: int,
) -> SyntheticPool:
    rng = np.random.default_rng(seed)
    target_ids = [f"T{idx:02d}" for idx in range(n_targets)]

    partition_centers = rng.normal(size=(n_partitions, n_features)).astype(np.float32)
    partition_strength = rng.beta(0.8, 8.0, size=(n_targets, n_partitions))
    partition_probs = rng.dirichlet(np.full(n_partitions, 0.8))
    partitions = rng.choice(n_partitions, size=pool_size, p=partition_probs)

    features = (
        partition_centers[partitions]
        + 0.7 * rng.normal(size=(pool_size, n_features)).astype(np.float32)
    )
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-6)

    scaffold_base = rng.integers(0, n_scaffolds, size=n_partitions)
    scaffold_noise = rng.integers(0, 17, size=pool_size)
    scaffolds = (scaffold_base[partitions] + scaffold_noise) % n_scaffolds

    risk_logits = 0.8 * features[:, 0] - 0.5 * features[:, 1] + rng.normal(0, 0.7, pool_size)
    risk_alert = (1.0 / (1.0 + np.exp(-risk_logits)) > 0.72).astype(int)

    target_weights = rng.normal(size=(n_targets, n_features)).astype(np.float32)
    logits = features @ target_weights.T
    logits += 3.8 * partition_strength[:, partitions].T
    logits -= 0.65 * risk_alert[:, None]
    logits += rng.normal(0, 0.35, size=(pool_size, n_targets))

    thresholds = np.quantile(logits, 0.955, axis=0)
    labels = (logits >= thresholds).astype(int)

    return SyntheticPool(
        features=features,
        labels=labels,
        partitions=partitions,
        scaffolds=scaffolds,
        risk_alert=risk_alert,
        target_ids=target_ids,
    )

