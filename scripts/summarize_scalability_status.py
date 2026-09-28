from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    Path("tables").mkdir(exist_ok=True)
    stress_paths = [
        Path("tables/large_pool_scalability_stress_smoke.csv"),
        Path("tables/large_pool_scalability_stress_large.csv"),
    ]
    frames = [pd.read_csv(path) for path in stress_paths if path.exists()]
    if frames:
        stress = pd.concat(frames, ignore_index=True).drop_duplicates(["target", "seed", "stress_pool_size"], keep="last")
        stress = stress.sort_values(["target", "seed", "stress_pool_size"])
        stress.to_csv("tables/large_pool_scalability_stress_summary.csv", index=False)
    else:
        stress = pd.DataFrame()

    rows = []
    audit_path = Path("tables/external_library_access_audit.csv")
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        for row in audit.itertuples(index=False):
            status = "not_complete"
            if getattr(row, "result_status") == "ready":
                status = "sample_downloaded"
            elif getattr(row, "dataset").startswith("enamine") and getattr(row, "create_status") == 200:
                status = "page_accessible_direct_snapshot_not_available"
            rows.append(
                {
                    "plan_item": getattr(row, "dataset"),
                    "status": status,
                    "evidence": getattr(row, "local_file") if isinstance(getattr(row, "local_file"), str) and getattr(row, "local_file") else "tables/external_library_access_audit.csv",
                    "note": getattr(row, "note"),
                }
            )
    if not stress.empty:
        best = stress.sort_values("stress_pool_size").iloc[-1]
        rows.append(
            {
                "plan_item": "DUD-E-derived large-pool scalability stress",
                "status": "complete_stress_only_not_external_snapshot",
                "evidence": "tables/large_pool_scalability_stress_summary.csv",
                "note": (
                    f"largest_pool={int(best.stress_pool_size)}; "
                    f"inference_saved_ratio={best.inference_saved_ratio:.3f}; "
                    f"online_speedup_excluding_partition={best.online_speedup_excluding_partition:.3f}; "
                    f"topk_overlap_with_full={best.topk_overlap_with_full:.3f}"
                ),
            }
        )
    pd.DataFrame(rows).to_csv("tables/scalability_table4_status.csv", index=False)
    print("tables/scalability_table4_status.csv")
    if not stress.empty:
        print("tables/large_pool_scalability_stress_summary.csv")


if __name__ == "__main__":
    main()
