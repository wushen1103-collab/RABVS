# Experiment Plan Status

This file maps the current repository state to `docs/EXPERIMENT_PLAN_EXTRACT.md`.
It is a planning/checklist artifact, not a manuscript table.

## Final Method Update

The final recommended retrospective and true-inference branch is
`rabvs_budgeted_no_ucb_pareto_rrf_no_conf`, superseding the earlier
`rabvs_budgeted_no_ucb_pareto_rrf_novel` branch. Full component ablation showed
that the clean and novelty ranks are necessary, while the confidence/density
rank is redundant and slightly hurts chemical quality.

See `docs/PARETO_NO_CONF_FINAL_RESULTS.md` for the final evidence and claim
boundaries.

## Main Benchmarks

| Plan item | Current status | Evidence | Remaining action |
|---|---|---|---|
| LIT-PCBA all15 | Complete and strengthened by final no_conf Pareto-RRF | `tables/pareto_no_conf_main_claim_summary.csv`, `tables/pareto_component_ablation_full_paired_tests.csv` | Use no_conf as the main hit-list method. |
| DUD-E all102 | Complete and strengthened by final no_conf Pareto-RRF | `tables/pareto_no_conf_main_claim_summary.csv`, `tables/pareto_component_ablation_full_paired_tests.csv` | Use no_conf as the main hit-list method; Hit@100 is neutral. |
| ZINC-22 lead-like 50M | Blocked as an external fixed snapshot | `tables/external_library_access_audit.csv`, `tables/scalability_table4_status.csv` | Needs stable fixed ZINC-22 SMILES/CXSMILES export before a Table 4 external claim. |
| Enamine REAL 10M | Blocked as an external fixed snapshot | `tables/external_library_access_audit.csv`, `tables/scalability_table4_status.csv` | Needs stable fixed Enamine REAL subset/export before a Table 4 external claim. |
| Docking oracle targets | Complete for final no_conf on original LIT-PCBA six-target top10 panel across three seeds | `tables/lit_pcba_pareto_no_conf_docking_top10_multiseed/` | Report as active-rate and alert-rate improvement with non-degraded Vina, not as significant affinity gain. |

## Metrics Coverage

| Metric family | Current status | Notes |
|---|---|---|
| Hit/recall/EF/AUBC/BEDROC | Complete on LIT-PCBA all15 and DUD-E all102 | no_conf improves DUD-E AUBC, Hit@500/1000, BEDROC, alert-free ratio, scaffolds, Tanimoto, and NLL; LIT-PCBA gains are larger across enrichment and chemistry metrics. |
| Calibration | Complete for appendix | no_conf changes the final ranking, not the underlying prediction/calibration model; existing calibration diagnostics remain applicable. |
| Efficiency | Complete except external-library snapshots | True-inference fixed20 validates 75.56% actual neural-inference saving with 96.83% Hit@1000 retention; real ZINC/Enamine snapshots remain missing. |
| Diversity/risk | Complete | no_conf improves alert-free ratio and unique scaffolds on both main datasets versus BOBa, and improves scaffolds versus the previous novel branch. |
| Docking | Complete for original six-target top10 panel | no_conf raises top10 active rate from 0.7167 to 0.7778, lowers risk-alert rate from 0.5111 to 0.4500, and has slightly better but non-significant Vina scores. |

## Table Targets

| Table target | Current status | Evidence | Claim boundary |
|---|---|---|---|
| Table 1 dataset overview | Complete | `tables/table1_dataset_overview.csv`, `tables/table1_dataset_overview_summary.csv` | Uses actual exact-metric screening pools; LIT-PCBA reflects `max_inactives=50000`. |
| Table 2 LIT-PCBA main results | Complete and strengthened | `tables/pareto_no_conf_main_claim_summary.csv` | no_conf improves Hit@100/500/1000, BEDROC, alert-free ratio, unique scaffolds, Tanimoto, and NLL versus BOBa. |
| Table 3 DUD-E main results | Complete and strengthened | `tables/pareto_no_conf_main_claim_summary.csv` | no_conf improves AUBC, Hit@500/1000, BEDROC, alert-free ratio, unique scaffolds, Tanimoto, and NLL; Hit@100 is neutral. |
| Table 4 ZINC-22 / Enamine scalability | Blocked external snapshot | `tables/scalability_table4_status.csv`, `tables/external_library_access_audit.csv` | DUD-E-derived 1M stress is complete, but external fixed-snapshot benchmark is not claimable. |
| Table 5 ablations | Complete and updated | `tables/table5_pareto_component_ablation_full.csv` | Clean and novelty are necessary; confidence/density is removed from the final branch. |
| Appendix per-target tables | Complete | `tables/exact_metrics_lit_all15/per_target_table.csv`, `tables/exact_metrics_dude_all102_fair/per_target_table.csv` | Keep stopped smokes and negative results as appendix/audit, not main wins. |

## Current Paper-Strength Story

The strongest main story is the Pareto-RRF final reranker without the confidence
term. Against fair BOBa, no_conf improves DUD-E all102 by +0.0277 AUBC,
+7.52 Hit@500, +4.94 Hit@1000, +0.0149 BEDROC, +0.0546 alert-free@1000,
+13.89 unique scaffolds@1000, -0.0010 mean Tanimoto, and -0.0065 NLL across
306 target-seed pairs.

On LIT-PCBA all15 it improves +2.62 Hit@100, +22.27 Hit@500, +32.80 Hit@1000,
+0.1767 BEDROC, +0.0414 alert-free@1000, +44.91 unique scaffolds, -0.0041 mean
Tanimoto, and -0.0068 NLL across 45 target-seed pairs.

True-inference fixed20 validates the practical-budget story: no_conf keeps
96.83% of full-greedy Hit@1000 by ratio of means, saves 75.56% actual neural
inference, improves alert-free@1000 by +6.32 points, adds +34.35 unique
scaffolds, and increases hits per 1000 inferences from 2.97 to 11.70.

The original LIT-PCBA docking panel supports candidate-quality improvement:
top10 active rate rises from 0.7167 to 0.7778 (p=0.0078), risk-alert rate drops
from 0.5111 to 0.4500, and mean Vina score is slightly better but not
significant.

Table 4 still has an engineering stress result, not a completed external
ZINC/Enamine benchmark. Real fixed ZINC-22/Enamine snapshots remain unavailable.

## Next Experiments By Priority

1. Obtain real ZINC-22/Enamine fixed snapshots for Table 4; the access audit is complete, but external snapshots remain unavailable in the current environment.
2. Optional only if external snapshots remain blocked: expand final no_conf docking from top10 to top25. Current top10 multiseed evidence is already enough for a non-degradation/candidate-quality claim.

