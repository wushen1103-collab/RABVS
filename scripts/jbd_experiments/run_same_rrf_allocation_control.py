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
from run_dude_pilot import prepare_pool as prepare_dude_pool
from run_dude_pilot import stable_target_offset
from run_lit_pcba_pilot import prepare_pool as prepare_lit_pool


DUD_TARGET_FILE = ROOT / "configs" / "dude_all102_targets.txt"
LIT_TARGETS = [
    "ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A",
    "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]
METHODS = ["rabvs", "mean_score_greedy", "boba_ucb", "random_partition"]
DISPLAY_NAMES = {
    "rabvs": "RABVS allocation",
    "mean_score_greedy": "Mean-score greedy allocation",
    "boba_ucb": "BOBa-UCB allocation",
    "random_partition": "Random partition allocation",
}


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


def same_rrf_order(mu: np.ndarray, alert: np.ndarray, scaffolds: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(scaffolds, return_inverse=True, return_counts=True)
    novelty = 1.0 / counts[inverse]
    fused = (
        1.0 / (60.0 + rank_desc(mu))
        + 0.18 / (60.0 + rank_asc(alert))
        + 0.25 / (60.0 + rank_desc(novelty))
    )
    return np.argsort(-fused, kind="stable")


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


def reservoir_sample(
    members: list[np.ndarray], seed: int, per_part: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    part_ids: list[int] = []
    for part, idx in enumerate(members):
        if not len(idx):
            continue
        chosen = rng.choice(idx, size=min(per_part, len(idx)), replace=False)
        indices.extend(chosen.tolist())
        part_ids.extend([part] * len(chosen))
    return np.asarray(indices, dtype=int), np.asarray(part_ids, dtype=int)


def allocation_order(
    method: str,
    pred,
    pool,
    partitions: np.ndarray,
    reservoir: np.ndarray,
    reservoir_parts: np.ndarray,
    labeled: np.ndarray,
    n_parts: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method == "random_partition":
        return np.argsort(-rng.random(n_parts), kind="stable")

    sampled = np.bincount(partitions[labeled], minlength=n_parts)
    utility = np.full(n_parts, -np.inf, dtype=float)
    for part in range(n_parts):
        positions = np.flatnonzero(reservoir_parts == part)
        if not len(positions):
            continue
        idx = reservoir[positions]
        if method == "mean_score_greedy":
            utility[part] = float(pred.mu[idx].mean())
            continue
        diversity = len(np.unique(pool.scaffolds[idx])) / len(idx)
        utility[part] = topmean(pred.mu[idx], q=0.25) + 0.10 * diversity
        if method == "boba_ucb":
            utility[part] += 0.5 * np.sqrt(2.0 * np.log(2.0) / max(1.0, sampled[part]))
    return np.argsort(-utility, kind="stable")


def select_scored(
    order: np.ndarray,
    members: list[np.ndarray],
    reservoir: np.ndarray,
    budget: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = set(int(x) for x in reservoir)
    for part in order:
        remaining = budget - len(selected)
        if remaining <= 0:
            break
        candidates = np.asarray(
            [x for x in members[int(part)] if int(x) not in selected], dtype=int
        )
        if len(candidates) > remaining:
            candidates = rng.choice(candidates, size=remaining, replace=False)
        selected.update(int(x) for x in candidates)
    return np.asarray(sorted(selected), dtype=int)


def prepare(dataset: str, target: str, seed: int):
    common = {
        "n_partitions": 64,
        "n_components": 96,
        "initial_labeled": 256,
    }
    if dataset == "dude":
        cfg = {**common, "data_root": "data/raw/dude/extracted", "max_decoys": 50000}
        return prepare_dude_pool(cfg, target, seed)
    cfg = {
        **common,
        "data_root": "data/raw/lit_pcba/AVE_unbiased",
        "split": "V",
        "max_inactives": 50000,
    }
    return prepare_lit_pool(cfg, target, seed)


def run_one(dataset: str, target: str, seed: int, ratio: float) -> list[dict]:
    pool, features, partitions, n_parts = prepare(dataset, target, seed)
    labels = pool.labels
    labeled = choose_initial(labels, 256, seed, target)
    pred = predict_with_uncertainty(
        features,
        labeled,
        labels,
        seed=seed,
        ensemble_size=5,
        retrieval_blend=0.5,
    )
    members = partition_members(partitions, n_parts)
    reservoir, reservoir_parts = reservoir_sample(
        members, seed + stable_target_offset(target), per_part=4
    )
    budget = min(
        len(labels),
        max(min(1000, len(labels)), int(np.ceil(ratio * len(labels))), len(reservoir)),
    )
    all_indices = np.arange(len(labels))
    rows: list[dict] = []
    for method in METHODS:
        part_order = allocation_order(
            method,
            pred,
            pool,
            partitions,
            reservoir,
            reservoir_parts,
            labeled,
            n_parts,
            seed + 1009,
        )
        scored = select_scored(part_order, members, reservoir, budget, seed + 7919)
        local_order = same_rrf_order(
            pred.mu[scored], pool.risk_alert[scored], pool.scaffolds[scored]
        )
        ranked_scored = scored[local_order]
        unscored = np.setdiff1d(all_indices, scored, assume_unique=False)
        ranking = np.concatenate([ranked_scored, unscored])
        top = ranking[: min(1000, len(ranking))]
        rows.append(
            {
                "dataset": "DUD-E" if dataset == "dude" else "LIT-PCBA",
                "target": target,
                "seed": seed,
                "allocation": DISPLAY_NAMES[method],
                "final_ranking": "Same activity/cleanliness/novelty RRF",
                "nominal_budget_fraction": ratio,
                "realized_budget_fraction": len(scored) / len(labels),
                "pool_size": len(labels),
                "scored_unique": len(scored),
                "Hit@1000": int(labels[top].sum()),
                "BEDROC_alpha20": bedroc_score(labels, ranking, alpha=20.0),
                "Alert-free@1000": float(1.0 - pool.risk_alert[top].mean()),
                "Scaffolds@1000": int(len(np.unique(pool.scaffolds[top]))),
            }
        )
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["dataset", "allocation", "final_ranking"], as_index=False)
        .agg(
            n_target_seed=("Hit@1000", "size"),
            realized_budget_fraction=("realized_budget_fraction", "mean"),
            Hit1000_mean=("Hit@1000", "mean"),
            Hit1000_std=("Hit@1000", "std"),
            BEDROC_mean=("BEDROC_alpha20", "mean"),
            BEDROC_std=("BEDROC_alpha20", "std"),
            AlertFree1000_mean=("Alert-free@1000", "mean"),
            AlertFree1000_std=("Alert-free@1000", "std"),
            Scaffolds1000_mean=("Scaffolds@1000", "mean"),
            Scaffolds1000_std=("Scaffolds@1000", "std"),
        )
        .sort_values(["dataset", "Hit1000_mean"], ascending=[True, False])
    )


def paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Hit@1000", "BEDROC_alpha20", "Alert-free@1000", "Scaffolds@1000"]
    reference = frame.loc[
        frame["allocation"] == DISPLAY_NAMES["rabvs"],
        ["dataset", "target", "seed", *metrics],
    ].rename(columns={metric: f"reference_{metric}" for metric in metrics})
    merged = frame.merge(reference, on=["dataset", "target", "seed"], validate="many_to_one")
    for metric in metrics:
        merged[f"delta_{metric}_vs_RABVS"] = merged[metric] - merged[f"reference_{metric}"]
    delta_cols = [f"delta_{metric}_vs_RABVS" for metric in metrics]
    return (
        merged.groupby(["dataset", "allocation"], as_index=False)
        .agg(**{
            f"{column}_mean": (column, "mean") for column in delta_cols
        }, **{
            f"{column}_std": (column, "std") for column in delta_cols
        })
        .sort_values(["dataset", "delta_Hit@1000_vs_RABVS_mean"], ascending=[True, False])
    )


def default_targets(dataset: str) -> list[str]:
    if dataset == "lit":
        return LIT_TARGETS
    return [line.strip() for line in DUD_TARGET_FILE.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dude", "lit"], required=True)
    parser.add_argument("--targets", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.targets:
        targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    else:
        targets = default_targets(args.dataset)
    if args.dataset == "dude":
        targets = [target.lower() for target in targets]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    jobs = [(target, seed) for target in targets for seed in seeds]
    groups = Parallel(n_jobs=min(args.jobs, len(jobs)), prefer="processes")(
        delayed(run_one)(args.dataset, target, seed, args.ratio)
        for target, seed in tqdm(jobs, desc=f"same-RRF {args.dataset}")
    )
    frame = pd.DataFrame([row for group in groups for row in group])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summarize(frame).to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    paired_deltas(frame).to_csv(out.with_name(out.stem + "_paired_deltas.csv"), index=False)
    print(out)


if __name__ == "__main__":
    main()
