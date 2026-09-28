from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from vina import Vina


TWO_LETTER_ELEMENTS = {
    "BR",
    "CL",
    "NA",
    "MG",
    "AL",
    "SI",
    "CA",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "SE",
    "AG",
    "CD",
    "HG",
}


def infer_element(atom_name: str, record_name: str) -> str:
    token = "".join(ch for ch in atom_name.strip().upper() if ch.isalpha())
    if not token:
        return ""
    if record_name == "HETATM" and len(token) >= 2 and token[:2] in TWO_LETTER_ELEMENTS:
        return token[:2].title()
    return token[0].title()


def fix_pdb_elements(src: Path, dst: Path) -> None:
    rows = []
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            padded = line.ljust(80)
            element = padded[76:78].strip()
            if not element:
                element = infer_element(padded[12:16], padded[:6].strip())
                padded = f"{padded[:76]}{element:>2}{padded[78:]}"
            line = padded.rstrip()
        rows.append(line)
    dst.write_text("\n".join(rows) + "\n", encoding="utf-8")


def mol2_coords(path: Path) -> np.ndarray:
    coords = []
    in_atom = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>") and in_atom:
            break
        if in_atom and line.strip():
            parts = line.split()
            if len(parts) >= 5:
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
    if not coords:
        raise ValueError(f"No MOL2 coordinates found in {path}")
    return np.asarray(coords, dtype=float)


def run_command(args: list[str], log_path: Path, check: bool = True) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}; see {log_path}")
    return proc.returncode


def bad_receptor_residues(log_path: Path) -> list[str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    residues: list[str] = []
    seen = set()
    for left, right in re.findall(r"\(:([0-9]+), :([0-9]+)\)", text):
        for residue in (left, right):
            if residue not in seen:
                residues.append(residue)
                seen.add(residue)
    for residue in re.findall(r"'(:[0-9]+)'\s*:", text):
        residue = residue.lstrip(":")
        if residue not in seen:
            residues.append(residue)
            seen.add(residue)
    return residues


def ensure_receptor_pdbqt(out_dir: Path) -> Path:
    receptor_pdbqt = out_dir / "receptor.pdbqt"
    if receptor_pdbqt.exists():
        return receptor_pdbqt
    fallback = out_dir / "receptor_rigid.pdbqt"
    if fallback.exists():
        fallback.replace(receptor_pdbqt)
        return receptor_pdbqt
    raise FileNotFoundError(f"receptor PDBQT was not created under {out_dir}")


def prepare_receptor(bin_dir: Path, receptor_fixed: Path, out_dir: Path) -> tuple[Path, str]:
    base_args = [
        str(bin_dir / "mk_prepare_receptor.py"),
        "--read_pdb",
        str(receptor_fixed),
        "-o",
        str(out_dir / "receptor"),
        "-p",
        "-a",
    ]
    first_log = out_dir / "prepare_receptor.log"
    code = run_command(base_args, first_log, check=False)
    deleted = ""
    if code != 0:
        residues = bad_receptor_residues(first_log)
        if residues:
            deleted = ",".join(f":{residue}" for residue in residues)
            retry_log = out_dir / "prepare_receptor_retry_delete.log"
            retry_args = base_args + ["-d", deleted]
            code = run_command(retry_args, retry_log, check=False)
        if code != 0:
            raise RuntimeError(f"receptor preparation failed for {receptor_fixed}; see {first_log}")
    return ensure_receptor_pdbqt(out_dir), deleted


def write_first_sdf(src_gz: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    wrote = False
    with gzip.open(src_gz, "rt", encoding="utf-8", errors="ignore") as src, dst.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            out.write(line)
            wrote = True
            if line.strip() == "$$$$":
                break
    if not wrote:
        raise ValueError(f"No molecules found in {src_gz}")


def prepare_ligand(bin_dir: Path, target_dir: Path, out_dir: Path) -> tuple[Path, Path, str]:
    crystal_pdbqt = out_dir / "crystal_ligand.pdbqt"
    crystal_log = out_dir / "prepare_ligand.log"
    code = run_command(
        [
            str(bin_dir / "mk_prepare_ligand.py"),
            "-i",
            str(target_dir / "crystal_ligand.mol2"),
            "-o",
            str(crystal_pdbqt),
        ],
        crystal_log,
        check=False,
    )
    if code == 0 and crystal_pdbqt.exists():
        return crystal_pdbqt, out_dir / "crystal_ligand.docked.pdbqt", "crystal_ligand"

    active_sdf = out_dir / "active0.sdf"
    active_pdbqt = out_dir / "active0.pdbqt"
    write_first_sdf(target_dir / "actives_final.sdf.gz", active_sdf)
    run_command(
        [
            str(bin_dir / "mk_prepare_ligand.py"),
            "-i",
            str(active_sdf),
            "-o",
            str(active_pdbqt),
        ],
        out_dir / "prepare_active0.log",
    )
    return active_pdbqt, out_dir / "active0.docked.pdbqt", "active0_sdf_fallback"


def dock_one(target: str, data_root: Path, out_root: Path, padding: float, exhaustiveness: int) -> dict:
    target_dir = data_root / target
    out_dir = out_root / target
    out_dir.mkdir(parents=True, exist_ok=True)

    receptor_fixed = out_dir / "receptor.fixed.pdb"

    fix_pdb_elements(target_dir / "receptor.pdb", receptor_fixed)
    bin_dir = Path(sys.executable).resolve().parent
    receptor_pdbqt, deleted_residues = prepare_receptor(bin_dir, receptor_fixed, out_dir)
    ligand_pdbqt, poses_pdbqt, ligand_source = prepare_ligand(bin_dir, target_dir, out_dir)

    coords = mol2_coords(target_dir / "crystal_ligand.mol2")
    center = coords.mean(axis=0)
    size = np.maximum(coords.max(axis=0) - coords.min(axis=0) + padding, 12.0)

    vina = Vina(sf_name="vina")
    vina.set_receptor(str(receptor_pdbqt))
    vina.set_ligand_from_file(str(ligand_pdbqt))
    vina.compute_vina_maps(center=center.tolist(), box_size=size.tolist())
    vina.dock(exhaustiveness=exhaustiveness, n_poses=5)
    energies = vina.energies(n_poses=5)
    vina.write_poses(str(poses_pdbqt), n_poses=5, overwrite=True)

    row = {
        "target": target,
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
        "best_vina_score": float(energies[0][0]),
        "n_poses": int(len(energies)),
        "exhaustiveness": exhaustiveness,
        "ligand_source": ligand_source,
        "receptor_deleted_residues": deleted_residues,
    }
    (out_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="adrb2,aa2ar")
    parser.add_argument("--data-root", default="data/raw/dude/extracted")
    parser.add_argument("--out-dir", default="runs/docking_smoke")
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    args = parser.parse_args()

    targets = [target.strip().lower() for target in args.targets.split(",") if target.strip()]
    rows = [
        dock_one(target, Path(args.data_root), Path(args.out_dir), args.padding, args.exhaustiveness)
        for target in targets
    ]
    out = Path(args.out_dir) / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
