from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/lit_pcba/AVE_unbiased")
    parser.add_argument("--out", default="tables/lit_pcba_dataset_table.csv")
    args = parser.parse_args()

    rows = []
    for target_dir in sorted(Path(args.data_root).iterdir()):
        if not target_dir.is_dir():
            continue
        row = {"dataset": "LIT-PCBA AVE", "target": target_dir.name}
        for split in ["T", "V"]:
            active = count_lines(target_dir / f"active_{split}.smi")
            inactive = count_lines(target_dir / f"inactive_{split}.smi")
            total = active + inactive
            row[f"active_{split}"] = active
            row[f"inactive_{split}"] = inactive
            row[f"library_size_{split}"] = total
            row[f"active_ratio_{split}"] = active / total if total else 0.0
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

