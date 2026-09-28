from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, default=2_000_000)
    parser.add_argument("--source-rows", type=int, default=2_854_815)
    parser.add_argument("--landmark-size", type=int, default=100_000)
    args = parser.parse_args()

    directory = Path(args.feature_dir)
    feature_path = directory / "chembl36_landmark_svd96.npy"
    component_path = directory / "chembl36_landmark_svd96_components.npy"
    id_path = directory / "chembl36_ids.txt"
    smiles_path = directory / "chembl36_control100k_smiles.tsv.gz"
    matrix = np.load(feature_path, mmap_mode="r")
    components = np.load(component_path, mmap_mode="r")
    id_count = sum(1 for _ in id_path.open(encoding="utf-8"))
    if matrix.shape != (args.expected_rows, 96):
        raise ValueError(f"Unexpected feature shape: {matrix.shape}")
    if components.shape != (96, 2048):
        raise ValueError(f"Unexpected component shape: {components.shape}")
    if id_count != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} IDs, found {id_count}")
    sample = np.concatenate([np.asarray(matrix[:1000]), np.asarray(matrix[-1000:])])
    if not np.isfinite(sample).all():
        raise ValueError("Feature cache contains non-finite values in the validation sample")
    mean_norm = float(np.linalg.norm(sample, axis=1).mean())
    if not 0.99 <= mean_norm <= 1.01:
        raise ValueError(f"Unexpected mean L2 norm: {mean_norm}")

    metadata = {
        "source": "ChEMBL 36 chemical representations",
        "source_file": args.source_file,
        "valid_molecules": args.expected_rows,
        "feature_file": str(feature_path),
        "feature_shape": [args.expected_rows, 96],
        "usable_rows": args.expected_rows,
        "source_rows_with_id_and_smiles": args.source_rows,
        "selection_order": "BLAKE2b-64 of '<selection_seed>:<ChEMBL ID>', ascending",
        "selection_seed": args.selection_seed,
        "nested_prefixes": True,
        "control_smiles_file": str(smiles_path),
        "control_smiles_rows": args.landmark_size,
        "fingerprint": "Morgan radius 2, 2048 bits",
        "projection": (
            "96-dimensional TruncatedSVD fitted on the fixed first 100000 molecules, "
            "then frozen and L2-normalized"
        ),
        "projection_mode": "landmark-svd",
        "landmark_size": args.landmark_size,
        "component_file": str(component_path),
        "validation_mean_l2_norm": mean_norm,
    }
    text = json.dumps(metadata, indent=2)
    (directory / "chembl36_landmark_svd96.json").write_text(text, encoding="utf-8")
    (directory / "chembl36_features.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
