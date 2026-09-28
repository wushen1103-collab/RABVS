from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

RDLogger.DisableLog("rdApp.*")
N_BITS = 2048
N_COMPONENTS = 96


def projection_maps() -> tuple[np.ndarray, np.ndarray]:
    bit = np.arange(N_BITS, dtype=np.uint64)
    hashed = bit * np.uint64(11400714819323198485) + np.uint64(7046029254386353131)
    dims = (hashed % np.uint64(N_COMPONENTS)).astype(np.int32)
    signs = np.where(((hashed >> np.uint64(32)) & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float32)
    return dims, signs


DIMS, SIGNS = projection_maps()
HASH_PROJECTION = sparse.csr_matrix(
    (SIGNS, (np.arange(N_BITS), DIMS)), shape=(N_BITS, N_COMPONENTS)
)
PROJECTION_MODE = "signed-hash"
SVD_COMPONENTS: np.ndarray | None = None


def initialize_projection_worker(
    projection_mode: str, svd_components: np.ndarray | None
) -> None:
    global PROJECTION_MODE, SVD_COMPONENTS
    PROJECTION_MODE = projection_mode
    SVD_COMPONENTS = svd_components


def fingerprint_chunk(
    rows: list[tuple[str, str]],
) -> tuple[list[str], list[str], sparse.csr_matrix]:
    ids: list[str] = []
    smiles_values: list[str] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    for chembl_id, smiles in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_BITS)
        on_bits = list(fp.GetOnBits())
        row_indices.extend([len(ids)] * len(on_bits))
        col_indices.extend(on_bits)
        ids.append(chembl_id)
        smiles_values.append(smiles)
    values = np.ones(len(row_indices), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)), shape=(len(ids), N_BITS), dtype=np.float32
    )
    return ids, smiles_values, matrix


def encode_chunk(rows: list[tuple[str, str]]) -> tuple[list[str], list[str], np.ndarray]:
    ids, smiles_values, bits = fingerprint_chunk(rows)
    if PROJECTION_MODE == "signed-hash":
        projected = bits @ HASH_PROJECTION
        matrix = projected.toarray() if sparse.issparse(projected) else np.asarray(projected)
    else:
        if SVD_COMPONENTS is None:
            raise RuntimeError("Landmark SVD components were not initialized")
        matrix = np.asarray(bits @ SVD_COMPONENTS.T)
    matrix = normalize(matrix, norm="l2", copy=False).astype(np.float32)
    return ids, smiles_values, matrix


def fit_landmark_svd(
    rows: list[tuple[str, str]],
    landmark_size: int,
    chunk_size: int,
    workers: int,
    seed: int,
) -> np.ndarray:
    context = mp.get_context("spawn")
    matrices = []
    valid = 0
    with context.Pool(workers) as pool:
        chunks = iter_chunks(rows, chunk_size)
        for _, _, matrix in pool.imap(fingerprint_chunk, chunks, chunksize=1):
            remaining = landmark_size - valid
            if remaining <= 0:
                break
            take = min(matrix.shape[0], remaining)
            matrices.append(matrix[:take])
            valid += take
    if valid < landmark_size:
        raise ValueError(f"Only {valid} valid landmark molecules were available")
    landmark = sparse.vstack(matrices, format="csr")
    model = TruncatedSVD(n_components=N_COMPONENTS, random_state=seed)
    model.fit(landmark)
    return model.components_.astype(np.float32)


def deterministic_rows(path: Path, selection_seed: int) -> list[tuple[str, str]]:
    ordered = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        names = {name.lower(): name for name in (reader.fieldnames or [])}
        id_col = names.get("chembl_id") or names.get("molecule_chembl_id")
        smiles_col = names.get("canonical_smiles") or names.get("smiles")
        if not id_col or not smiles_col:
            raise ValueError(f"Cannot identify ChEMBL ID/SMILES columns: {reader.fieldnames}")
        for row in reader:
            chembl_id = (row.get(id_col) or "").strip()
            smiles = (row.get(smiles_col) or "").strip()
            if not chembl_id or not smiles:
                continue
            digest = hashlib.blake2b(
                f"{selection_seed}:{chembl_id}".encode("utf-8"), digest_size=8
            ).digest()
            ordered.append((int.from_bytes(digest, "big"), chembl_id, smiles))
    ordered.sort(key=lambda item: (item[0], item[1]))
    print(f"source_rows={len(ordered)} deterministic_order_seed={selection_seed}", flush=True)
    return [(chembl_id, smiles) for _, chembl_id, smiles in ordered]


def iter_chunks(rows: list[tuple[str, str]], chunk_size: int):
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-molecules", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--selection-seed", type=int, default=20260928)
    parser.add_argument("--control-smiles", type=int, default=100_000)
    parser.add_argument("--projection", choices=["signed-hash", "landmark-svd"], default="signed-hash")
    parser.add_argument("--landmark-size", type=int, default=100_000)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_stem = (
        "chembl36_hashed_morgan96"
        if args.projection == "signed-hash"
        else "chembl36_landmark_svd96"
    )
    features_path = output / f"{feature_stem}.npy"
    ids_path = output / "chembl36_ids.txt"
    control_smiles_path = output / "chembl36_control100k_smiles.tsv.gz"
    features = open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(args.max_molecules, N_COMPONENTS),
    )

    count = 0
    ordered_rows = deterministic_rows(source, args.selection_seed)
    global PROJECTION_MODE, SVD_COMPONENTS
    PROJECTION_MODE = args.projection
    if args.projection == "landmark-svd":
        SVD_COMPONENTS = fit_landmark_svd(
            ordered_rows,
            args.landmark_size,
            args.chunk_size,
            args.workers,
            args.selection_seed,
        )
        np.save(output / "chembl36_landmark_svd96_components.npy", SVD_COMPONENTS)
        print(f"landmark_svd_fitted={args.landmark_size}", flush=True)
    context = mp.get_context("spawn")
    with (
        ids_path.open("w", encoding="utf-8") as id_handle,
        gzip.open(control_smiles_path, "wt", encoding="utf-8", newline="") as smiles_handle,
        context.Pool(
            args.workers,
            initializer=initialize_projection_worker,
            initargs=(PROJECTION_MODE, SVD_COMPONENTS),
        ) as pool,
    ):
        smiles_handle.write("chembl_id\tcanonical_smiles\n")
        chunks = iter_chunks(ordered_rows, args.chunk_size)
        for ids, smiles_values, matrix in pool.imap(encode_chunk, chunks, chunksize=1):
            if count >= args.max_molecules:
                break
            take = min(len(ids), args.max_molecules - count)
            if take == 0:
                continue
            features[count : count + take] = matrix[:take]
            id_handle.write("\n".join(ids[:take]) + "\n")
            control_take = min(take, max(0, args.control_smiles - count))
            for chembl_id, smiles in zip(ids[:control_take], smiles_values[:control_take]):
                smiles_handle.write(f"{chembl_id}\t{smiles}\n")
            count += take
            if count % 100_000 < take:
                features.flush()
                print(f"encoded={count}", flush=True)

    features.flush()
    metadata = {
        "source": "ChEMBL 36 chemical representations",
        "source_file": str(source),
        "valid_molecules": count,
        "feature_file": str(features_path),
        "feature_shape": [args.max_molecules, N_COMPONENTS],
        "usable_rows": count,
        "source_rows_with_id_and_smiles": len(ordered_rows),
        "selection_order": "BLAKE2b-64 of '<selection_seed>:<ChEMBL ID>', ascending",
        "selection_seed": args.selection_seed,
        "nested_prefixes": True,
        "control_smiles_file": str(control_smiles_path),
        "control_smiles_rows": min(args.control_smiles, count),
        "fingerprint": "Morgan radius 2, 2048 bits",
        "projection": (
            "fixed signed hashing to 96 dimensions followed by L2 normalization"
            if args.projection == "signed-hash"
            else "96-dimensional TruncatedSVD fitted on the fixed first 100000 molecules, then frozen and L2-normalized"
        ),
        "projection_mode": args.projection,
        "landmark_size": args.landmark_size if args.projection == "landmark-svd" else None,
    }
    (output / f"{feature_stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "chembl36_features.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
