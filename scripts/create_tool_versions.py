from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def version_for(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def command_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception:
        return None


def main() -> None:
    output = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" in sys.argv else Path("artifacts/tool_versions.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "sklearn",
        "pyarrow": "pyarrow",
        "yaml": "yaml",
        "rdkit": "rdkit",
        "faiss": "faiss",
        "lightgbm": "lightgbm",
        "torch": "torch",
        "torch_geometric": "torch_geometric",
        "chemprop": "chemprop",
        "esm": "esm",
        "openbabel": "openbabel",
        "vina": "vina",
        "meeko": "meeko",
        "gemmi": "gemmi",
        "prody": "prody",
        "biopython": "Bio",
    }
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": {name: version_for(module) for name, module in modules.items()},
        "commands": {
            "nvidia-smi": command_output(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"]),
            "vina": command_output(["vina", "--version"]),
            "obabel": command_output(["obabel", "-V"]),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
