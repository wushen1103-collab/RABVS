from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"exit={proc.returncode}\n")
        return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="configs/data_sources.yaml")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--execute", action="store_true", help="Actually download URL entries that define a direct_url.")
    args = parser.parse_args()

    sources = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8"))["datasets"]
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    for name, meta in sources.items():
        ds_dir = root / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "license_note.md").write_text(
            f"# {name}\n\nHomepage: {meta['homepage']}\n\n{meta.get('mirror_note', '')}\n",
            encoding="utf-8",
        )
        (ds_dir / "download_log.txt").write_text("metadata initialized\n", encoding="utf-8")
        manifest_path = ds_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["dataset", "homepage", "preferred_source", "local_file", "sha256"])
            writer.writeheader()
            writer.writerow(
                {
                    "dataset": name,
                    "homepage": meta["homepage"],
                    "preferred_source": meta.get("preferred_source", "official"),
                    "local_file": "",
                    "sha256": "",
                }
            )
        if args.execute and meta.get("direct_url"):
            target = ds_dir / Path(meta["direct_url"]).name
            code = run(["aria2c", "-x", "16", "-s", "16", "-o", target.name, "-d", str(ds_dir), meta["direct_url"]], ds_dir / "download_log.txt")
            if code != 0:
                return code
            (ds_dir / "raw_md5.txt").write_text(f"{target.name}\tsha256:{sha256(target)}\n", encoding="utf-8")
        else:
            (ds_dir / "raw_md5.txt").write_text("No raw archive downloaded in metadata-only mode.\n", encoding="utf-8")

    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

