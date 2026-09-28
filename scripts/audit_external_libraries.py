from __future__ import annotations

import argparse
import csv
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def request_text(url: str, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None, timeout: int = 30) -> tuple[int, str, str]:
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": "RABVS-scalability-audit/1.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            return int(response.status), response.headers.get("content-type", ""), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return int(exc.code), exc.headers.get("content-type", ""), body
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        return 0, type(exc).__name__, str(exc)


def multipart(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----rabvsBoundary"
    body = b""
    for key, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()
    return body, boundary


def audit_zinc(count: int, polls: int, poll_sleep: float, out_dir: Path) -> dict[str, str | int | float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    body, boundary = multipart({"count": str(count), "subset": "lead-like", "output_fields": "zinc_id,smiles,tranche"})
    status, ctype, text = request_text(
        "https://cartblanche22.docking.org/substance/random.txt",
        method="POST",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=60,
    )
    task = ""
    try:
        task = str(json.loads(text).get("task", ""))
    except json.JSONDecodeError:
        pass
    row: dict[str, str | int | float] = {
        "dataset": "zinc22_lead_like",
        "homepage": "https://cartblanche22.docking.org/search/random",
        "requested_count": count,
        "create_status": status,
        "create_content_type": ctype,
        "task": task,
        "result_status": "not_polled" if task else "no_task",
        "result_lines": 0,
        "local_file": "",
        "note": "",
    }
    if not task:
        row["note"] = text[:200].replace("\n", " ")
        return row
    poll_url = f"https://cartblanche22.docking.org/substance/random/{task}.txt"
    last_body = ""
    for attempt in range(polls):
        status, ctype, result = request_text(poll_url, timeout=45)
        last_body = result
        if status == 200 and not result.lstrip().startswith("{") and len(result.splitlines()) >= 2:
            local = out_dir / f"zinc22_lead_like_random_{count}_{task}.txt"
            local.write_text(result, encoding="utf-8")
            row.update(
                {
                    "result_status": "ready",
                    "result_lines": len(result.splitlines()),
                    "local_file": str(local),
                    "note": "downloaded via CartBlanche random task endpoint",
                }
            )
            return row
        row["result_status"] = f"pending_or_timeout_attempt_{attempt + 1}_http_{status}"
        time.sleep(poll_sleep)
    row["note"] = last_body[:200].replace("\n", " ")
    return row


def audit_enamine() -> list[dict[str, str | int]]:
    urls = [
        ("enamine_real_database", "https://enamine.net/compound-collections/real-compounds/real-database"),
        ("enamine_real_database_subsets", "https://enamine.net/compound-collections/real-compounds/real-database-subsets"),
    ]
    rows = []
    for name, url in urls:
        status, ctype, text = request_text(url, timeout=60)
        lowered = text.lower()
        rows.append(
            {
                "dataset": name,
                "homepage": url,
                "requested_count": "",
                "create_status": status,
                "create_content_type": ctype,
                "task": "",
                "result_status": "page_accessible" if status == 200 else "page_unavailable",
                "result_lines": 0,
                "local_file": "",
                "note": "page mentions SMILES/SDF and subset downloads; direct fixed snapshot download requires site/store access" if "smiles" in lowered or "download" in lowered else text[:200].replace("\n", " "),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tables/external_library_access_audit.csv")
    parser.add_argument("--raw-dir", default="data/raw/zinc22_smoke")
    parser.add_argument("--zinc-count", type=int, default=20)
    parser.add_argument("--polls", type=int, default=6)
    parser.add_argument("--poll-sleep", type=float, default=5.0)
    args = parser.parse_args()

    rows = [audit_zinc(args.zinc_count, args.polls, args.poll_sleep, Path(args.raw_dir)), *audit_enamine()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
