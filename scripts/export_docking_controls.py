from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_dude_pilot import prepare_pool, stable_target_offset


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def control_rows(pool, indices: np.ndarray, target: str, seed: int, method: str) -> list[dict]:
    rows = []
    for rank, idx in enumerate(indices, start=1):
        rows.append(
            {
                "target": target,
                "seed": seed,
                "method": method,
                "rank": rank,
                "pool_index": int(idx),
                "mol_id": str(pool.mol_ids[idx]),
                "smiles": str(pool.smiles[idx]),
                "label": int(pool.labels[idx]),
                "scaffold": int(pool.scaffolds[idx]),
                "risk_alert": int(pool.risk_alert[idx]),
                "mu": 0.0,
                "u_epi": 0.0,
                "u_conf": 0.0,
                "density_penalty": 0.0,
                "was_queried": 0,
                "queried_total": 0,
                "queried_hits": 0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dude_all102_final_ablation.yaml")
    parser.add_argument("--candidates", default="runs/docking_candidates/candidates.csv")
    parser.add_argument("--targets", default="aa2ar,adrb2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-method", default="rabvs_budgeted_reliable")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", default="runs/docking_control_smoke/candidates.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidates = pd.read_csv(args.candidates)
    rows = []
    for target in parse_csv_arg(args.targets):
        reference = candidates[
            (candidates["target"] == target)
            & (candidates["seed"] == args.seed)
            & (candidates["method"] == args.reference_method)
            & (candidates["rank"] <= args.n)
        ].copy()
        if len(reference) != args.n:
            raise ValueError(f"Expected {args.n} reference rows for {target}, found {len(reference)}")
        rows.extend(reference.to_dict(orient="records"))

        pool, _, _, _ = prepare_pool(cfg, target, args.seed)
        rng = np.random.default_rng(args.seed + stable_target_offset(target) + 9187)
        excluded = set(reference["pool_index"].astype(int))
        active_indices = np.array(
            [idx for idx in np.flatnonzero(pool.labels == 1) if int(idx) not in excluded]
        )
        decoy_indices = np.flatnonzero(pool.labels == 0)
        if len(active_indices) < args.n or len(decoy_indices) < args.n:
            raise ValueError(f"Insufficient controls for {target}")
        random_actives = rng.choice(active_indices, size=args.n, replace=False)
        random_decoys = rng.choice(decoy_indices, size=args.n, replace=False)
        rows.extend(control_rows(pool, random_actives, target, args.seed, "random_active_control"))
        rows.extend(control_rows(pool, random_decoys, target, args.seed, "random_decoy_control"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
