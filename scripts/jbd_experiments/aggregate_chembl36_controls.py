from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_many(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No input files matched")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def aggregate_scale(input_dir: Path, output_dir: Path, scale_stem: str) -> None:
    raw_paths = sorted(input_dir.glob(f"{scale_stem}_seed[0-9].csv"))
    frame = read_many(raw_paths).drop_duplicates(["target", "seed", "library_size"])
    frame.to_csv(output_dir / f"{scale_stem}_all.csv", index=False)
    summary = (
        frame.groupby("library_size", as_index=False)
        .agg(
            n_target_seed=("topK_overlap", "size"),
            unique_scores_avoided_mean=("unique_scores_avoided", "mean"),
            unique_scores_avoided_std=("unique_scores_avoided", "std"),
            topK_overlap_mean=("topK_overlap", "mean"),
            topK_overlap_std=("topK_overlap", "std"),
            full_score_seconds_mean=("full_score_seconds", "mean"),
            full_score_seconds_std=("full_score_seconds", "std"),
            online_seconds_mean=("online_seconds", "mean"),
            online_seconds_std=("online_seconds", "std"),
            online_speedup_mean=("online_speedup", "mean"),
            online_speedup_std=("online_speedup", "std"),
            peak_rss_gib_mean=("peak_rss_gib", "mean"),
            peak_rss_gib_std=("peak_rss_gib", "std"),
        )
    )
    summary.to_csv(output_dir / f"{scale_stem}_all_summary.csv", index=False)

    index_paths = sorted(input_dir.glob(f"{scale_stem}_seed[0-9]_index.csv"))
    index_frame = read_many(index_paths).drop_duplicates(["seed", "library_size"])
    index_frame.to_csv(output_dir / f"{scale_stem}_all_index.csv", index=False)

    largest_size = int(frame.library_size.max())
    largest = frame[frame.library_size == largest_size]
    largest_index = index_frame[index_frame.library_size == largest_size]
    full_mean = float(largest.full_score_seconds.mean())
    online_mean = float(largest.online_seconds.mean())
    index_mean = float(largest_index.index_seconds.mean())
    rows = []
    for target_count in [1, 2, 5, 10, 20, 50, 100]:
        exhaustive = target_count * full_mean
        rabvs = index_mean + target_count * online_mean
        rows.append(
            {
                "target_count": target_count,
                "exhaustive_seconds": exhaustive,
                "rabvs_seconds_including_index": rabvs,
                "amortized_speedup": exhaustive / rabvs,
                "index_seconds_mean_over_seeds": index_mean,
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / f"{scale_stem}_all_amortization.csv", index=False
    )


def aggregate_representation(input_dir: Path, output_dir: Path) -> None:
    paths = sorted(input_dir.glob("chembl36_representation_seed[0-9].csv"))
    frame = read_many(paths).drop_duplicates(["target", "seed", "representation"])
    frame.to_csv(output_dir / "chembl36_representation_all.csv", index=False)
    summary = (
        frame.groupby("representation", as_index=False)
        .agg(
            n_target_seed=("topK_overlap", "size"),
            unique_scores_avoided_mean=("unique_scores_avoided", "mean"),
            unique_scores_avoided_std=("unique_scores_avoided", "std"),
            topK_overlap_mean=("topK_overlap", "mean"),
            topK_overlap_std=("topK_overlap", "std"),
            online_seconds_mean=("online_seconds", "mean"),
            online_seconds_std=("online_seconds", "std"),
            representation_seconds_mean=("representation_seconds", "mean"),
            representation_seconds_std=("representation_seconds", "std"),
            index_seconds_mean=("index_seconds", "mean"),
            index_seconds_std=("index_seconds", "std"),
            peak_rss_gib_mean=("peak_rss_gib", "mean"),
        )
    )
    summary.to_csv(output_dir / "chembl36_representation_all_summary.csv", index=False)

    metrics = ["unique_scores_avoided", "topK_overlap", "online_seconds", "index_seconds"]
    wide = frame.pivot(index=["target", "seed"], columns="representation", values=metrics)
    paired = wide.copy()
    for metric in metrics:
        paired[(metric, "SVD_minus_hash")] = (
            wide[(metric, "Truncated-SVD-96")] - wide[(metric, "signed-hash-96")]
        )
    paired.columns = [f"{metric}__{representation}" for metric, representation in paired.columns]
    paired.reset_index().to_csv(
        output_dir / "chembl36_representation_all_paired.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale-stem", default="chembl36_scale_svd")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_scale(input_dir, output_dir, args.scale_stem)
    aggregate_representation(input_dir, output_dir)


if __name__ == "__main__":
    main()
