from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_DIR = Path("tables")


def _metric_row(frame: pd.DataFrame, comparison: str, metric: str) -> pd.Series:
    rows = frame[(frame["comparison"] == comparison) & (frame["metric"] == metric)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {comparison}/{metric}, found {len(rows)}")
    return rows.iloc[0]


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_table1() -> None:
    rows = []
    specs = [
        (
            "LIT-PCBA AVE",
            "V split, max 50k inactives",
            "runs/exact_metrics_lit_all15/summary.csv",
        ),
        (
            "DUD-E",
            "full extracted target pool",
            "runs/exact_metrics_dude_all102_fair/summary.csv",
        ),
    ]
    for dataset, split, path in specs:
        frame = pd.read_csv(path)
        cols = ["target", "pool_size", "active_count", "active_ratio_pool"]
        unique = frame[cols].drop_duplicates("target").sort_values("target")
        for row in unique.itertuples(index=False):
            rows.append(
                {
                    "dataset": dataset,
                    "target": row.target,
                    "split_or_pool": split,
                    "library_size": int(row.pool_size),
                    "active_count": int(row.active_count),
                    "inactive_or_decoy_count": int(row.pool_size - row.active_count),
                    "active_ratio": float(row.active_ratio_pool),
                    "source_run": path,
                    "status": "actual_screening_pool",
                }
            )
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(exist_ok=True)
    table.to_csv(TABLE_DIR / "table1_dataset_overview.csv", index=False)

    summary_rows = []
    for dataset, group in table.groupby("dataset", sort=True):
        summary_rows.append(
            {
                "dataset": dataset,
                "n_targets": int(group["target"].nunique()),
                "library_size_sum": int(group["library_size"].sum()),
                "library_size_mean": float(group["library_size"].mean()),
                "library_size_min": int(group["library_size"].min()),
                "library_size_max": int(group["library_size"].max()),
                "active_count_sum": int(group["active_count"].sum()),
                "active_ratio_mean": float(group["active_ratio"].mean()),
                "active_ratio_min": float(group["active_ratio"].min()),
                "active_ratio_max": float(group["active_ratio"].max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "table1_dataset_overview_summary.csv", index=False)


def build_table5() -> None:
    rows: list[dict[str, object]] = []

    missing = pd.read_csv(TABLE_DIR / "missing_component_ablation_paired_summary.csv")
    allocation = pd.read_csv(TABLE_DIR / "allocation_ablation_summary.csv")
    final = pd.read_csv(TABLE_DIR / "final_recommendation_ablation_summary_v2.csv")
    retrieval = pd.read_csv(TABLE_DIR / "retrieval_final_decision_table.csv")

    def missing_ablation(comparison: str, ablation: str, variant: str, takeaway: str) -> None:
        metrics = {
            metric: _metric_row(missing, comparison, metric)
            for metric in [
                "AUBC",
                "Hit@1000",
                "BEDROC_alpha20",
                "NLL",
                "PAINS_Brenk_free_ratio_at1000",
                "unique_scaffold_count@1000",
                "inference_saved_ratio",
            ]
            if len(missing[(missing["comparison"] == comparison) & (missing["metric"] == metric)])
        }
        rows.append(
            {
                "ablation_group": "component",
                "ablation": ablation,
                "dataset_scope": "DUD-E fixed20, 3 seeds",
                "variant": variant,
                "baseline": "rabvs_budgeted_reliable, ensemble_size=5",
                "comparison_direction": "variant_minus_baseline",
                "AUBC_delta": _float_or_none(metrics.get("AUBC", {}).get("mean_delta")),
                "Hit@1000_delta": _float_or_none(metrics.get("Hit@1000", {}).get("mean_delta")),
                "BEDROC_alpha20_delta": _float_or_none(metrics.get("BEDROC_alpha20", {}).get("mean_delta")),
                "NLL_delta": _float_or_none(metrics.get("NLL", {}).get("mean_delta")),
                "PAINS_Brenk_free_ratio_at1000_delta": _float_or_none(
                    metrics.get("PAINS_Brenk_free_ratio_at1000", {}).get("mean_delta")
                ),
                "unique_scaffold_count_at1000_delta": _float_or_none(
                    metrics.get("unique_scaffold_count@1000", {}).get("mean_delta")
                ),
                "inference_saved_ratio_delta": _float_or_none(
                    metrics.get("inference_saved_ratio", {}).get("mean_delta")
                ),
                "primary_p": _float_or_none(
                    metrics.get("inference_saved_ratio" if "partition" in ablation else "AUBC", {}).get(
                        "wilcoxon_p_nonzero"
                    )
                ),
                "takeaway": takeaway,
                "evidence_table": "tables/missing_component_ablation_paired_summary.csv",
            }
        )

    missing_ablation(
        "no_partition_minus_main_ensemble5",
        "remove target-aware partitioning",
        "rabvs_no_partition_reliable",
        "Accuracy is not significantly improved by partitioning, but removing it eliminates the inference-saving mechanism.",
    )
    missing_ablation(
        "single_model_minus_ensemble5",
        "single surrogate instead of 5-model ensemble",
        "ensemble_size=1",
        "The ensemble materially improves AUBC, BEDROC, NLL, and held-out calibration quality.",
    )

    def allocation_ablation(method: str, ablation: str, takeaway: str) -> None:
        rows_found = allocation[
            (allocation["dataset"] == "DUD-E fixed20 standard") & (allocation["method"] == method)
        ]
        if len(rows_found) != 1:
            raise ValueError(f"Expected one allocation row for {method}, found {len(rows_found)}")
        row = rows_found.iloc[0]
        rows.append(
            {
                "ablation_group": "allocation",
                "ablation": ablation,
                "dataset_scope": row["dataset"],
                "variant": method,
                "baseline": row["baseline"],
                "comparison_direction": "variant_minus_baseline",
                "AUBC_delta": float(row["AUBC_diff_vs_q75"]),
                "Hit@1000_delta": float(row["Hit@1000_diff_vs_q75"]),
                "BEDROC_alpha20_delta": None,
                "NLL_delta": None,
                "PAINS_Brenk_free_ratio_at1000_delta": float(
                    row["PAINS_Brenk_free_ratio_at1000_diff_vs_q75"]
                ),
                "unique_scaffold_count_at1000_delta": float(
                    row["unique_scaffold_count@1000_diff_vs_q75"]
                ),
                "inference_saved_ratio_delta": None,
                "primary_p": float(row["AUBC_p_vs_q75"]),
                "takeaway": takeaway,
                "evidence_table": "tables/allocation_ablation_summary.csv",
            }
        )

    allocation_ablation(
        "rabvs_budgeted_no_ucb",
        "remove UCB exploration bonus",
        "No-UCB is stronger for standard-budget DUD-E final hits and became the high-hit branch.",
    )
    allocation_ablation(
        "rabvs_budgeted_random_parts",
        "random partition allocation",
        "Random partition selection is much worse, supporting target-aware allocation.",
    )

    def final_ablation(method: str, ablation: str, takeaway: str) -> None:
        rows_found = final[(final["dataset"] == "DUD-E all102 standard") & (final["method"] == method)]
        if len(rows_found) != 1:
            raise ValueError(f"Expected one final-ranking row for {method}, found {len(rows_found)}")
        row = rows_found.iloc[0]
        rows.append(
            {
                "ablation_group": "final_ranking",
                "ablation": ablation,
                "dataset_scope": row["dataset"],
                "variant": method,
                "baseline": "rabvs_budgeted_reliable",
                "comparison_direction": "variant_minus_baseline",
                "AUBC_delta": float(row["AUBC_vs_q75_diff"]),
                "Hit@1000_delta": float(row["Hit@1000_vs_q75_diff"]),
                "BEDROC_alpha20_delta": None,
                "NLL_delta": None,
                "PAINS_Brenk_free_ratio_at1000_delta": float(
                    row["PAINS_Brenk_free_ratio_at1000_vs_q75_diff"]
                ),
                "unique_scaffold_count_at1000_delta": float(
                    row["unique_scaffold_count@1000_vs_q75_diff"]
                ),
                "inference_saved_ratio_delta": None,
                "primary_p": float(row["Hit@1000_vs_q75_p"]),
                "takeaway": takeaway,
                "evidence_table": "tables/final_recommendation_ablation_summary_v2.csv",
            }
        )

    final_ablation(
        "rabvs_budgeted_reliable_no_alert",
        "remove structural-alert penalty",
        "Removing the alert penalty can recover hits but destroys most alert-free gain.",
    )
    final_ablation(
        "rabvs_budgeted_reliable_no_novelty",
        "remove novelty term",
        "Removing novelty sharply reduces scaffold diversity.",
    )
    final_ablation(
        "rabvs_budgeted_reliable_q50",
        "stricter conformal threshold q50",
        "Too-strict conformal filtering loses final hits.",
    )
    final_ablation(
        "rabvs_budgeted_reliable_q90",
        "relaxed conformal threshold q90",
        "q90 is close to q75 for hits but slightly changes reliability/diversity balance.",
    )

    retrieval_row = retrieval[
        (retrieval["experiment"] == "all102_standard_finalretrieval0p5")
        & (retrieval["comparison"] == "vs_finalretrieval_boba")
    ]
    if len(retrieval_row) == 1:
        row = retrieval_row.iloc[0]
        rows.append(
            {
                "ablation_group": "retrieval_calibration",
                "ablation": "final-only retrieval blend 0.5",
                "dataset_scope": "DUD-E all102 standard",
                "variant": row["adopted_method"],
                "baseline": "BOBa with same final retrieval",
                "comparison_direction": "variant_minus_baseline",
                "AUBC_delta": float(row["AUBC_mean_delta"]),
                "Hit@1000_delta": float(row["Hit_at_1000_mean_delta"]),
                "BEDROC_alpha20_delta": None,
                "NLL_delta": None,
                "PAINS_Brenk_free_ratio_at1000_delta": float(
                    row["PAINS_Brenk_free_ratio_at1000_mean_delta"]
                ),
                "unique_scaffold_count_at1000_delta": float(
                    row["unique_scaffold_count_at_1000_mean_delta"]
                ),
                "inference_saved_ratio_delta": None,
                "primary_p": float(row["Hit_at_1000_p"]),
                "takeaway": "Final-only retrieval gives the strongest standard-budget DUD-E high-hit branch.",
                "evidence_table": "tables/retrieval_final_decision_table.csv",
            }
        )

    pd.DataFrame(rows).to_csv(TABLE_DIR / "table5_ablation_summary.csv", index=False)


def build_completion_audit() -> None:
    rows = [
        {
            "plan_item": "Table 1 dataset overview",
            "status": "complete",
            "evidence": "tables/table1_dataset_overview.csv",
            "claim_boundary": "Uses actual screening pools from exact-metric runs; LIT-PCBA reflects max_inactives=50000.",
            "remaining_action": "",
        },
        {
            "plan_item": "Table 2 LIT-PCBA main results",
            "status": "complete",
            "evidence": "tables/exact_metrics_lit_all15/main_table.csv; tables/exact_metrics_paired_summary.csv",
            "claim_boundary": "Reliability/diversity win; enrichment is a trade-off.",
            "remaining_action": "",
        },
        {
            "plan_item": "Table 3 DUD-E main results",
            "status": "complete",
            "evidence": "tables/exact_metrics_dude_all102_fair/main_table.csv; tables/exact_metrics_paired_summary.csv",
            "claim_boundary": "Strong vs original BOBa; fair-baseline BEDROC is negative.",
            "remaining_action": "",
        },
        {
            "plan_item": "Table 4 ZINC-22 / Enamine scalability",
            "status": "blocked_external_snapshot",
            "evidence": "tables/scalability_table4_status.csv; tables/external_library_access_audit.csv",
            "claim_boundary": "Only DUD-E-derived 1M stress is complete; do not claim external ZINC/Enamine benchmark.",
            "remaining_action": "Needs stable fixed ZINC-22/Enamine SMILES/CXSMILES snapshot or licensed export.",
        },
        {
            "plan_item": "Table 5 ablations",
            "status": "complete",
            "evidence": "tables/table5_ablation_summary.csv",
            "claim_boundary": "Partitioning is a scalability mechanism; ensemble improves quality; alert/novelty terms trade hits for reliability/diversity.",
            "remaining_action": "",
        },
        {
            "plan_item": "Appendix per-target tables",
            "status": "complete",
            "evidence": "tables/exact_metrics_lit_all15/per_target_table.csv; tables/exact_metrics_dude_all102_fair/per_target_table.csv",
            "claim_boundary": "Include stopped/negative smoke results separately, not as main wins.",
            "remaining_action": "",
        },
        {
            "plan_item": "Docking oracle targets",
            "status": "complete",
            "evidence": "tables/lit_pcba_original_docking_multiseed_*_top10.csv",
            "claim_boundary": "Lower alerts and better Vina score; active-rate trade-off.",
            "remaining_action": "",
        },
        {
            "plan_item": "Calibration appendix",
            "status": "complete",
            "evidence": "tables/calibration_appendix_summary.csv; docs/CALIBRATION_APPENDIX_NOTE.md",
            "claim_boundary": "DUD-E calibration improves; LIT-PCBA calibration is neutral; risk-coverage AUC is diagnostic.",
            "remaining_action": "",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE_DIR / "experiment_plan_completion_audit.csv", index=False)

    lines = [
        "# Experiment Completion Audit",
        "",
        "This audit maps `docs/EXPERIMENT_PLAN_EXTRACT.md` to generated evidence.",
        "It is text-only; plotting is deferred.",
        "",
        "| Plan item | Status | Evidence | Claim boundary | Remaining action |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {plan_item} | {status} | `{evidence}` | {claim_boundary} | {remaining_action} |".format(
                **row
            )
        )
    Path("docs/EXPERIMENT_COMPLETION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    build_table1()
    build_table5()
    build_completion_audit()
    print(TABLE_DIR / "table1_dataset_overview.csv")
    print(TABLE_DIR / "table1_dataset_overview_summary.csv")
    print(TABLE_DIR / "table5_ablation_summary.csv")
    print(TABLE_DIR / "experiment_plan_completion_audit.csv")
    print("docs/EXPERIMENT_COMPLETION_AUDIT.md")


if __name__ == "__main__":
    main()
