from __future__ import annotations

import json
import os
import subprocess


def cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception as exc:
        return str(exc)


def main() -> None:
    payload = {
        "cpu_total": os.cpu_count(),
        "cpu_usable_with_30_reserved": max(1, (os.cpu_count() or 1) - 30),
        "memory": cmd(["free", "-h"]),
        "disk": cmd(["df", "-h", "."]),
        "gpu_query": cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "gpu_pmon": cmd(["nvidia-smi", "pmon", "-c", "1"]),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

