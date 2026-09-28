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

from rabvs.bandit import final_ranking, predict_with_uncertainty, select_batch
from rabvs.metrics import bedroc_score
from run_dude_pilot import prepare_pool as prepare_dude_pool
from run_dude_pilot import stable_target_offset
from run_lit_pcba_pilot import prepare_pool as prepare_lit_pool


DUD_TARGET_FILE = ROOT / "configs" / "dude_all102_targets.txt"
LIT_TARGETS = [
    "ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A",
    "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]
ALLOCATION_METHODS = {
    "rabvs_budgeted_no_ucb_pareto_rrf_no_conf": "RABVS allocation",
    "greedy_surrogate": "Mean-score greedy allocation",
    "boba_ucb": "BOBa-UCB allocation",
    "rabvs_budgeted_random_parts": "Random partition allocation",
}
FIXED_FINAL_RRF = "rabvs_budgeted_no_ucb_pareto_rrf_no_conf"


def prepare(dataset: str, target: str, seed: int):
    cfg = {
        "n_partitions": 256,
        "n_components": 96,
        "initial_labeled": 512,
    }
    if dataset == "dude":
        return prepare_dude_pool(
            {**cfg, "data_root": "data/raw/dude/extracted", "max_decoys": 50000},
            target,
            seed,
        )
    return prepare_lit_pool(
        {
            **cfg,
            "data_root": "data/raw/lit_pcba/AVE_unbiased",
            "split": "V",
            "max_inactives": 50000,
        },
        target,
        seed,
    )


def initial_indices(labels: np.ndarray, target: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + stable_target_offset(target))
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    positive_n = min(len(positives), 16)
    negative_n = min(len(negatives), 512 - positive_n)
    return np.unique(
        np.concatenate(
            [
                rng.choice(positives, size=positive_n, replace=False),
                rng.choice(negatives, size=negative_n, replace=False),
            ]
        )
    )


def run_one(dataset: str, target: str, seed: int) -> list[dict]:
    pool, features, partitions, n_parts = prepare(dataset, target, seed)
    labels = pool.labels
    all_indices = np.arange(len(labels))
    rows: list[dict] = []
    for acquisition, display_name in ALLOCATION_METHODS.items():
        labeled = initial_indices(labels, target, seed)
        sampled_per_partition = np.bincount(partitions[labeled], minlength=n_parts).astype(float)
        for round_no in range(8):
            unlabeled = np.setdiff1d(all_indices, labeled, assume_unique=False)
            pred = predict_with_uncertainty(
                features,
                labeled,
                labels,
                seed=seed + round_no,
                retrieval_blend=0.0,
                ensemble_size=5,
            )
            batch = select_batch(
                method=acquisition,
                pred=pred,
                labels_seen=labels[labeled],
                unlabeled_idx=unlabeled,
                partitions=partitions,
                scaffolds=pool.scaffolds,
                risk_alert=pool.risk_alert,
                sampled_per_partition=sampled_per_partition,
                query_size=256,
                round_no=round_no,
                seed=seed,
            )
            labeled = np.unique(np.concatenate([labeled, batch]))
            sampled_per_partition += np.bincount(partitions[batch], minlength=n_parts)

        final_pred = predict_with_uncertainty(
            features,
            labeled,
            labels,
            seed=seed + 8,
            retrieval_blend=0.5,
            ensemble_size=5,
        )
        ranking = final_ranking(FIXED_FINAL_RRF, final_pred, pool.scaffolds, pool.risk_alert)
        top = ranking[: min(1000, len(ranking))]
        rows.append(
            {
                "dataset": "DUD-E" if dataset == "dude" else "LIT-PCBA",
                "target": target,
                "seed": seed,
                "allocation": display_name,
                "acquisition_method": acquisition,
                "final_ranking": "Same activity/cleanliness/novelty RRF",
                "queried_total": len(labeled),
                "queried_hits": int(labels[labeled].sum()),
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
            queried_total=("queried_total", "mean"),
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
        frame["allocation"] == "RABVS allocation",
        ["dataset", "target", "seed", *metrics],
    ].rename(columns={metric: f"reference_{metric}" for metric in metrics})
    merged = frame.merge(reference, on=["dataset", "target", "seed"], validate="many_to_one")
    for metric in metrics:
        merged[f"delta_{metric}_vs_RABVS"] = merged[metric] - merged[f"reference_{metric}"]
    delta_cols = [f"delta_{metric}_vs_RABVS" for metric in metrics]
    aggregations = {}
    for column in delta_cols:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    return (
        merged.groupby(["dataset", "allocation"], as_index=False)
        .agg(**aggregations)
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
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    targets = (
        [value.strip() for value in args.targets.split(",") if value.strip()]
        if args.targets
        else default_targets(args.dataset)
    )
    if args.dataset == "dude":
        targets = [target.lower() for target in targets]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    jobs = [(target, seed) for target in targets for seed in seeds]
    groups = Parallel(n_jobs=min(args.jobs, len(jobs)), prefer="processes")(
        delayed(run_one)(args.dataset, target, seed)
        for target, seed in tqdm(jobs, desc=f"matched same-RRF {args.dataset}")
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
