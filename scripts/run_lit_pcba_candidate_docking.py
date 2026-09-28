from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rabvs.utils import cpu_worker_cap
from run_docking_smoke import mol2_coords, prepare_receptor, run_command


STANDARD_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
TWO_LETTER_ELEMENTS = {"BR", "CL", "NA", "MG", "AL", "SI", "CA", "MN", "FE", "CO", "NI", "CU", "ZN", "SE"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def stable_ligand_seed(target: str, smiles: str, seed: int) -> int:
    payload = f"{target}\0{smiles}\0{seed}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value % 2_147_483_000 + 1


def infer_element(atom_name: str, atom_type: str) -> str:
    atom_type_root = atom_type.split(".", 1)[0].strip().upper()
    if atom_type_root in TWO_LETTER_ELEMENTS:
        return atom_type_root.title()
    if atom_type_root and atom_type_root[0].isalpha():
        return atom_type_root[0].title()
    token = "".join(ch for ch in atom_name.strip().upper() if ch.isalpha())
    if len(token) >= 2 and token[:2] in TWO_LETTER_ELEMENTS:
        return token[:2].title()
    return token[:1].title() or "C"


def parse_residue(subst_id: str, subst_name: str) -> tuple[str, int]:
    match = re.match(r"([A-Za-z]{3})(-?\d+)?", subst_name.strip())
    if match:
        res_name = match.group(1).upper()
        res_seq_text = match.group(2)
        if res_seq_text is not None:
            return res_name, int(res_seq_text)
        if subst_id.lstrip("-").isdigit():
            return res_name, int(subst_id)
        return res_name, 1
    if subst_id.lstrip("-").isdigit():
        return subst_name[:3].upper().ljust(3), int(subst_id)
    return subst_name[:3].upper().ljust(3), 1


def mol2_protein_to_pdb(src: Path, dst: Path) -> None:
    rows = []
    in_atom = False
    serial = 1
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>") and in_atom:
            break
        if not in_atom or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        atom_name = parts[1][:4]
        x, y, z = (float(parts[2]), float(parts[3]), float(parts[4]))
        atom_type = parts[5]
        subst_id = parts[6]
        subst_name = parts[7]
        res_name, res_seq = parse_residue(subst_id, subst_name)
        element = infer_element(atom_name, atom_type)
        record = "ATOM" if res_name in STANDARD_RESIDUES else "HETATM"
        rows.append(
            f"{record:<6}{serial:5d} {atom_name:<4} {res_name:>3} A{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}"
        )
        serial += 1
    if not rows:
        raise ValueError(f"No MOL2 atoms found in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + "\nEND\n", encoding="utf-8")


def available_structure_ids(target_dir: Path) -> list[str]:
    proteins = {path.name[: -len("_protein.mol2")] for path in target_dir.glob("*_protein.mol2")}
    ligands = {path.name[: -len("_ligand.mol2")] for path in target_dir.glob("*_ligand.mol2")}
    return sorted(proteins & ligands)


def prepare_target(
    target: str,
    data_root: Path,
    out_root: Path,
    padding: float,
    structure_id: str | None,
) -> dict:
    target_dir = data_root / target
    out_dir = out_root / "targets" / safe_name(target)
    out_dir.mkdir(parents=True, exist_ok=True)

    structure_ids = [structure_id] if structure_id else available_structure_ids(target_dir)
    if not structure_ids:
        raise FileNotFoundError(f"No paired *_protein.mol2/*_ligand.mol2 files under {target_dir}")
    errors = []
    for sid in structure_ids:
        if sid is None:
            continue
        protein_mol2 = target_dir / f"{sid}_protein.mol2"
        ligand_mol2 = target_dir / f"{sid}_ligand.mol2"
        try:
            receptor_pdb = out_dir / f"{sid}.receptor.pdb"
            mol2_protein_to_pdb(protein_mol2, receptor_pdb)
            receptor_pdbqt, deleted_residues = prepare_receptor(
                Path(sys.executable).resolve().parent, receptor_pdb, out_dir
            )
            coords = mol2_coords(ligand_mol2)
            center = coords.mean(axis=0)
            size = np.maximum(coords.max(axis=0) - coords.min(axis=0) + padding, 12.0)
            return {
                "target": target,
                "target_dir": str(target_dir),
                "structure_id": sid,
                "protein_mol2": str(protein_mol2),
                "ligand_mol2": str(ligand_mol2),
                "receptor_pdbqt": str(receptor_pdbqt),
                "center": center.tolist(),
                "size": size.tolist(),
                "receptor_deleted_residues": deleted_residues,
            }
        except Exception as exc:  # try another co-crystal if one target structure is malformed
            errors.append(f"{sid}: {exc}")
    raise RuntimeError(f"No usable receptor/ligand pair for {target}: {' | '.join(errors[:5])}")


def smiles_to_sdf(smiles: str, out: Path, seed: int) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    code = AllChem.EmbedMolecule(mol, params)
    if code != 0:
        code = AllChem.EmbedMolecule(mol, randomSeed=int(seed), useRandomCoords=True)
    if code != 0:
        raise ValueError("RDKit conformer embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    out.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(mol, str(out))


def dock_one(row: dict, target_context: dict, out_root: Path, exhaustiveness: int) -> dict:
    target = row["target"]
    seed = int(row["seed"])
    rank = int(row["rank"])
    pool_index = int(row["pool_index"])
    method = safe_name(str(row["method"]))
    dock_id = int(row["dock_id"])
    ligand_dir = out_root / "ligands" / safe_name(target) / f"{dock_id:05d}_{method}_r{rank}_i{pool_index}"
    ligand_sdf = ligand_dir / "ligand.sdf"
    ligand_pdbqt = ligand_dir / "ligand.pdbqt"
    poses_pdbqt = ligand_dir / "poses.pdbqt"
    log_path = ligand_dir / "prepare_ligand.log"

    result = {
        "dock_id": dock_id,
        "target": target,
        "smiles": row["smiles"],
        "dock_status": "ok",
        "best_vina_score": np.nan,
        "n_poses": 0,
        "error_message": "",
    }
    try:
        ligand_seed = stable_ligand_seed(target, str(row["smiles"]), seed)
        embed_seed = (ligand_seed + 17) % 2_147_483_647 or 1
        docking_seed = (ligand_seed + 29) % 2_147_483_647 or 1
        smiles_to_sdf(str(row["smiles"]), ligand_sdf, seed=embed_seed)
        run_command(
            [
                str(Path(sys.executable).resolve().parent / "mk_prepare_ligand.py"),
                "-i",
                str(ligand_sdf),
                "-o",
                str(ligand_pdbqt),
            ],
            log_path,
        )

        vina = Vina(sf_name="vina", cpu=1, seed=docking_seed, verbosity=0)
        vina.set_receptor(target_context["receptor_pdbqt"])
        vina.set_ligand_from_file(str(ligand_pdbqt))
        vina.compute_vina_maps(center=target_context["center"], box_size=target_context["size"])
        vina.dock(exhaustiveness=exhaustiveness, n_poses=5)
        energies = vina.energies(n_poses=5)
        vina.write_poses(str(poses_pdbqt), n_poses=5, overwrite=True)
        result["best_vina_score"] = float(energies[0][0])
        result["n_poses"] = int(len(energies))
    except Exception as exc:
        result["dock_status"] = "failed"
        result["error_message"] = str(exc)[:500]
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for (target, method), group in frame.groupby(["target", "method"], sort=True):
        ok = group[group["dock_status"] == "ok"]
        grouped.append(
            {
                "target": target,
                "method": method,
                "n_candidates": int(len(group)),
                "n_docked_ok": int(len(ok)),
                "active_rate": float(group["label"].mean()),
                "risk_alert_rate": float(group["risk_alert"].mean()),
                "mean_mu": float(group["mu"].mean()),
                "mean_vina_score": float(ok["best_vina_score"].mean()) if len(ok) else np.nan,
                "median_vina_score": float(ok["best_vina_score"].median()) if len(ok) else np.nan,
                "best_vina_score": float(ok["best_vina_score"].min()) if len(ok) else np.nan,
            }
        )
    return pd.DataFrame(grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="runs/lit_pcba_original_docking_seed0/candidates.csv")
    parser.add_argument("--data-root", default="data/raw/lit_pcba/AVE_unbiased")
    parser.add_argument("--out-dir", default="runs/lit_pcba_original_docking_seed0")
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=60)
    parser.add_argument("--structure-id")
    parser.add_argument("--out", default="runs/lit_pcba_original_docking_seed0/docked_top10.csv")
    parser.add_argument("--summary-out", default="tables/lit_pcba_original_docking_summary_seed0_top10.csv")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    panel = candidates[candidates["rank"] <= args.max_rank].copy()
    panel = panel.sort_values(["target", "seed", "method", "rank"]).reset_index(drop=True)
    unique = panel.drop_duplicates(["target", "smiles"]).reset_index(drop=True)
    unique["dock_id"] = np.arange(len(unique), dtype=int)

    out_root = Path(args.out_dir)
    contexts = {
        target: prepare_target(target, Path(args.data_root), out_root, args.padding, args.structure_id)
        for target in sorted(unique["target"].unique())
    }
    jobs = unique.to_dict(orient="records")
    results = Parallel(n_jobs=cpu_worker_cap(args.n_jobs), prefer="processes")(
        delayed(dock_one)(row, contexts[row["target"]], out_root, args.exhaustiveness) for row in jobs
    )
    docked = pd.DataFrame(results)
    merged = panel.merge(
        unique[["target", "smiles", "dock_id"]], on=["target", "smiles"], how="left"
    ).merge(
        docked[["dock_id", "dock_status", "best_vina_score", "n_poses", "error_message"]],
        on="dock_id",
        how="left",
    )
    for target, context in contexts.items():
        mask = merged["target"] == target
        merged.loc[mask, "structure_id"] = context["structure_id"]
        merged.loc[mask, "protein_mol2"] = context["protein_mol2"]
        merged.loc[mask, "ligand_mol2"] = context["ligand_mol2"]
        merged.loc[mask, "receptor_deleted_residues"] = context["receptor_deleted_residues"]
        merged.loc[mask, "box_center_x"] = context["center"][0]
        merged.loc[mask, "box_center_y"] = context["center"][1]
        merged.loc[mask, "box_center_z"] = context["center"][2]
        merged.loc[mask, "box_size_x"] = context["size"][0]
        merged.loc[mask, "box_size_y"] = context["size"][1]
        merged.loc[mask, "box_size_z"] = context["size"][2]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summarize(merged).to_csv(summary_out, index=False)
    print(out)
    print(summary_out)


if __name__ == "__main__":
    main()
