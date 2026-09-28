from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/dude/extracted")
    parser.add_argument("--manifest", default="data/raw/dude/manifest.csv")
    parser.add_argument("--out", default="tables/dude_fixed20_dataset_table.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest) if Path(args.manifest).exists() else None
    manifest_lookup = {}
    if manifest is not None:
        manifest_lookup = manifest.set_index("target").to_dict(orient="index")

    rows = []
    for target_dir in sorted(Path(args.data_root).iterdir()):
        if not target_dir.is_dir():
            continue
        active_path = target_dir / "actives_final.ism"
        decoy_path = target_dir / "decoys_final.ism"
        if not active_path.exists() or not decoy_path.exists():
            continue
        active = count_lines(active_path)
        decoy = count_lines(decoy_path)
        total = active + decoy
        manifest_row = manifest_lookup.get(target_dir.name, {})
        rows.append(
            {
                "dataset": "DUD-E",
                "target": target_dir.name,
                "active": active,
                "decoy": decoy,
                "library_size": total,
                "active_ratio": active / total if total else 0.0,
                "sha256": manifest_row.get("sha256", ""),
                "archive_bytes": manifest_row.get("archive_bytes", ""),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
