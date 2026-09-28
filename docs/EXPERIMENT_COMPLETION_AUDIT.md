# Experiment Completion Audit

This audit maps `docs/EXPERIMENT_PLAN_EXTRACT.md` to generated evidence.
It is text-only; plotting is deferred.

| Plan item | Status | Evidence | Claim boundary | Remaining action |
|---|---|---|---|---|
| Table 1 dataset overview | complete | `tables/table1_dataset_overview.csv` | Uses actual screening pools from exact-metric runs; LIT-PCBA reflects max_inactives=50000. |  |
| Table 2 LIT-PCBA main results | complete_strengthened | `tables/pareto_no_conf_main_claim_summary.csv; tables/pareto_component_ablation_full_paired_tests.csv` | Final no_conf branch improves enrichment, BEDROC, alert-free ratio, scaffold diversity, Tanimoto, and NLL versus BOBa. |  |
| Table 3 DUD-E main results | complete_strengthened | `tables/pareto_no_conf_main_claim_summary.csv; tables/pareto_component_ablation_full_paired_tests.csv` | Final no_conf branch improves AUBC, Hit@500/1000, BEDROC, alert-free ratio, scaffolds, Tanimoto, and NLL; Hit@100 is neutral. |  |
| Table 4 ZINC-22 / Enamine scalability | blocked_external_snapshot | `tables/scalability_table4_status.csv; tables/external_library_access_audit.csv` | Only DUD-E-derived 1M stress is complete; do not claim external ZINC/Enamine benchmark. | Needs stable fixed ZINC-22/Enamine SMILES/CXSMILES snapshot or licensed export. |
| Table 5 ablations | complete_strengthened | `tables/table5_pareto_component_ablation_full.csv; tables/pareto_component_ablation_full_paired_tests.csv` | Clean and novelty terms are necessary; confidence/density is redundant and removed in final no_conf branch. |  |
| Appendix per-target tables | complete | `tables/exact_metrics_lit_all15/per_target_table.csv; tables/exact_metrics_dude_all102_fair/per_target_table.csv` | Include stopped/negative smoke results separately, not as main wins. |  |
| Docking oracle targets | complete_strengthened | `tables/lit_pcba_pareto_no_conf_docking_top10_multiseed/` | Final no_conf branch improves active rate and lowers alert rate, with non-degraded/slightly better Vina scores; do not claim significant docking-affinity gain. |  |
| Calibration appendix | complete | `tables/calibration_appendix_summary.csv; docs/CALIBRATION_APPENDIX_NOTE.md` | DUD-E calibration improves; LIT-PCBA calibration is neutral; risk-coverage AUC is diagnostic. |  |

