from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.warning")


@dataclass(frozen=True)
class LitPcbaPool:
    target: str
    mol_ids: np.ndarray
    smiles: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    scaffolds: np.ndarray
    risk_alert: np.ndarray
    pains_alert: np.ndarray
    brenk_alert: np.ndarray


def read_smi(path: str | Path, label: int) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smiles = parts[0]
            mol_id = parts[1] if len(parts) > 1 else f"{Path(path).stem}_{len(rows)}"
            rows.append({"smiles": smiles, "mol_id": mol_id, "label": label})
    return pd.DataFrame(rows)


def load_lit_pcba_target(
    data_root: str | Path,
    target: str,
    split: str = "V",
    max_inactives: int | None = None,
    seed: int = 0,
) -> LitPcbaPool:
    target_dir = Path(data_root) / target
    active = read_smi(target_dir / f"active_{split}.smi", 1)
    inactive = read_smi(target_dir / f"inactive_{split}.smi", 0)
    if max_inactives and len(inactive) > max_inactives:
        inactive = inactive.sample(max_inactives, random_state=seed)
    frame = pd.concat([active, inactive], ignore_index=True)
    frame = frame.drop_duplicates("smiles").reset_index(drop=True)

    feature_rows = []
    valid_rows = []
    scaffolds = []
    risk = []
    pains = []
    brenk = []
    pains_catalog = _alert_catalog(FilterCatalogParams.FilterCatalogs.PAINS)
    brenk_catalog = _alert_catalog(FilterCatalogParams.FilterCatalogs.BRENK)
    scaffold_ids: dict[str, int] = {}

    for row in frame.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or row.smiles
        if scaffold not in scaffold_ids:
            scaffold_ids[scaffold] = len(scaffold_ids)
        pains_match = int(pains_catalog.GetFirstMatch(mol) is not None)
        brenk_match = int(brenk_catalog.GetFirstMatch(mol) is not None)
        feature_rows.append(arr)
        valid_rows.append(row)
        scaffolds.append(scaffold_ids[scaffold])
        pains.append(pains_match)
        brenk.append(brenk_match)
        risk.append(max(pains_match, brenk_match))

    if not feature_rows:
        raise ValueError(f"No valid molecules for {target}")

    valid = pd.DataFrame(valid_rows)
    return LitPcbaPool(
        target=target,
        mol_ids=valid["mol_id"].astype(str).to_numpy(),
        smiles=valid["smiles"].astype(str).to_numpy(),
        features=np.vstack(feature_rows).astype(np.float32),
        labels=valid["label"].astype(int).to_numpy(),
        scaffolds=np.asarray(scaffolds, dtype=int),
        risk_alert=np.asarray(risk, dtype=int),
        pains_alert=np.asarray(pains, dtype=int),
        brenk_alert=np.asarray(brenk, dtype=int),
    )


def _alert_catalog(catalog: FilterCatalogParams.FilterCatalogs) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(catalog)
    return FilterCatalog(params)

