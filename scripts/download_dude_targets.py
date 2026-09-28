from __future__ import annotations

import argparse
import hashlib
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


DEFAULT_TARGETS = [
    "aa2ar",
    "abl1",
    "ace",
    "aces",
    "ada",
    "akt1",
    "akt2",
    "aldr",
    "ampc",
    "andr",
    "aofb",
    "bace1",
    "braf",
    "cah2",
    "casp3",
    "cdk2",
    "comt",
    "cp2c9",
    "cp3a4",
    "csf1r",
]


def parse_targets(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_TARGETS
    path = Path(text)
    if path.exists():
        return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar(archive: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (out_dir / member.name).resolve()
            if out_dir != member_path and out_dir not in member_path.parents:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(out_dir)


def count_nonempty(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def download_file(url: str, out_path: Path, retries: int) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RABVS/0.1"})
            with urllib.request.urlopen(req, timeout=120) as response, tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp_path.replace(out_path)
            return
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def process_target(
    target: str,
    download_root: Path,
    extract_root: Path,
    retries: int,
    extract: bool,
) -> dict:
    target = target.lower()
    url = f"https://dude.docking.org/targets/{target}/{target}.tar.gz"
    archive = download_root / f"{target}.tar.gz"
    row = {
        "dataset": "DUD-E",
        "target": target,
        "url": url,
        "archive": str(archive),
        "status": "ok",
        "error": "",
    }
    try:
        download_file(url, archive, retries)
        row["sha256"] = sha256_file(archive)
        row["archive_bytes"] = archive.stat().st_size
        if extract:
            safe_extract_tar(archive, extract_root)
        target_dir = extract_root / target
        row["active_count"] = count_nonempty(target_dir / "actives_final.ism")
        row["decoy_count"] = count_nonempty(target_dir / "decoys_final.ism")
        if row["active_count"] == 0 or row["decoy_count"] == 0:
            raise RuntimeError("Missing actives_final.ism or decoys_final.ism after extraction")
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=None, help="Comma-separated target codes or a text file.")
    parser.add_argument("--download-root", default="data/raw/dude/downloads")
    parser.add_argument("--extract-root", default="data/raw/dude/extracted")
    parser.add_argument("--manifest", default="data/raw/dude/manifest.csv")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    download_root = Path(args.download_root)
    extract_root = Path(args.extract_root)
    download_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(
                process_target,
                target,
                download_root,
                extract_root,
                args.retries,
                not args.no_extract,
            )
            for target in targets
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['target']}: {row['status']}")

    table = pd.DataFrame(rows).sort_values("target")
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(manifest, index=False)
    failures = table[table["status"] != "ok"]
    print(manifest)
    if len(failures):
        raise SystemExit(f"{len(failures)} DUD-E downloads failed; see {manifest}")


if __name__ == "__main__":
    main()
