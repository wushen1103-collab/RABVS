from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    required = ["numpy", "pandas", "sklearn", "yaml", "joblib", "tqdm"]
    optional = ["rdkit", "faiss", "lightgbm", "chemprop", "esm", "openbabel", "vina"]
    report = {"required": {}, "optional": {}}
    failed = []
    for name in required:
        ok = importlib.util.find_spec(name) is not None
        report["required"][name] = ok
        if not ok:
            failed.append(name)
    for name in optional:
        report["optional"][name] = importlib.util.find_spec(name) is not None

    from rabvs.synthetic import make_synthetic_pool
    from rabvs.bandit import predict_with_uncertainty

    pool = make_synthetic_pool(512, 1, 16, 12, 128, seed=7)
    labeled = list(range(64))
    pred = predict_with_uncertainty(pool.features, labeled, pool.labels[:, 0], seed=7, ensemble_size=2)
    report["synthetic_prediction_shape"] = list(pred.mu.shape)

    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        print(f"Missing required modules: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

