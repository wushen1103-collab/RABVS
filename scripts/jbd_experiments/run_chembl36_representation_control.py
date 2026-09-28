from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rabvs.dude import load_dude_target
from run_chembl36_scalability import N_BITS, N_COMPONENTS, peak_rss_gib, run_target
from run_true_budgeted_dude import fit_subset_predictor, initial_labeled_indices


RDLogger.DisableLog("rdApp.*")


def load_fingerprints(path: Path, limit: int) -> tuple[list[str], sparse.csr_matrix]:
    ids: list[str] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for record in reader:
            if len(ids) >= limit:
                break
            mol = Chem.MolFromSmiles(record["canonical_smiles"])
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_BITS)
            bits = list(fp.GetOnBits())
            row_indices.extend([len(ids)] * len(bits))
            col_indices.extend(bits)
            ids.append(record["chembl_id"])
    values = np.ones(len(row_indices), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)), shape=(len(ids), N_BITS), dtype=np.float32
    )
    return ids, matrix


def projection_matrix() -> sparse.csr_matrix:
    bit = np.arange(N_BITS, dtype=np.uint64)
    hashed = bit * np.uint64(11400714819323198485) + np.uint64(7046029254386353131)
    dims = (hashed % np.uint64(N_COMPONENTS)).astype(np.int32)
    signs = np.where(
        ((hashed >> np.uint64(32)) & np.uint64(1)) == 0, 1.0, -1.0
    ).astype(np.float32)
    return sparse.csr_matrix(
        (signs, (np.arange(N_BITS), dims)), shape=(N_BITS, N_COMPONENTS)
    )


def hash_features(bits: sparse.csr_matrix) -> np.ndarray:
    projected = bits @ projection_matrix()
    if sparse.issparse(projected):
        projected = projected.toarray()
    projected = np.asarray(projected, dtype=np.float32)
    return normalize(projected, norm="l2", copy=False).astype(np.float32)


def transformed_predictor(
    target: str,
    seed: int,
    data_root: str,
    ensemble_size: int,
    representation: str,
    svd: TruncatedSVD | None,
):
    pool = load_dude_target(data_root, target, max_decoys=50000, seed=seed)
    if representation == "signed-hash-96":
        features = hash_features(pool.features)
    else:
        if svd is None:
            raise ValueError("SVD transformer is required")
        features = normalize(svd.transform(pool.features), norm="l2").astype(np.float32)
    labeled = initial_labeled_indices(pool.labels, 512, seed, target)
    return fit_subset_predictor(features, labeled, pool.labels, seed, ensemble_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--data-root", default="data/raw/dude/extracted")
    parser.add_argument("--targets", default="aa2ar,akt1,cdk2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--selected-fraction", type=float, default=0.10)
    parser.add_argument("--reservoir-per-part", type=int, default=16)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    targets = [value.strip().lower() for value in args.targets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    ids, bits = load_fingerprints(Path(args.smiles), args.size)
    if len(ids) != args.size:
        raise ValueError(f"Requested {args.size} valid molecules, found {len(ids)}")

    rows: list[dict] = []
    for seed in seeds:
        representations: list[tuple[str, np.ndarray, TruncatedSVD | None, float]] = []

        start = time.perf_counter()
        hashed = hash_features(bits)
        hash_seconds = time.perf_counter() - start
        representations.append(("signed-hash-96", hashed, None, hash_seconds))

        start = time.perf_counter()
        svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=seed)
        svd_features = normalize(svd.fit_transform(bits), norm="l2").astype(np.float32)
        svd_seconds = time.perf_counter() - start
        representations.append(("Truncated-SVD-96", svd_features, svd, svd_seconds))

        for name, features, transformer, representation_seconds in representations:
            n_parts = min(2048, max(256, int(np.ceil(args.size / 1000))))
            index_start = time.perf_counter()
            partitions = MiniBatchKMeans(
                n_clusters=n_parts,
                batch_size=16384,
                n_init=1,
                max_iter=50,
                random_state=seed,
            ).fit_predict(features)
            index_seconds = time.perf_counter() - index_start
            for target in targets:
                predictor = transformed_predictor(
                    target,
                    seed,
                    args.data_root,
                    args.ensemble_size,
                    name,
                    transformer,
                )
                row = run_target(
                    features,
                    partitions,
                    n_parts,
                    predictor,
                    target,
                    args.size,
                    args.selected_fraction,
                    args.reservoir_per_part,
                    args.top_k,
                    seed,
                )
                row.update(
                    {
                        "representation": name,
                        "representation_seconds": representation_seconds,
                        "index_seconds": index_seconds,
                        "unique_scores_avoided": row["scores_avoided"],
                        "total_offline_seconds": representation_seconds + index_seconds,
                        "peak_rss_gib": peak_rss_gib(),
                    }
                )
                rows.append(row)

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
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
            index_seconds_mean=("index_seconds", "mean"),
            index_seconds_std=("index_seconds", "std"),
            peak_rss_gib_mean=("peak_rss_gib", "mean"),
        )
    )
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(out)


if __name__ == "__main__":
    main()
