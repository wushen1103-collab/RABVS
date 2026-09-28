# Experiment Log

## Execution Rule

- New experiment branches default to the smallest representative smoke run
  that can validate the pipeline, metric direction, and failure modes.
- Promote a smoke run to a full run only when its key metrics are directionally
  useful, no implementation/data anomaly remains, and the manuscript claim
  needs broader statistical or target-level evidence.
- A failed or ambiguous smoke run triggers diagnosis or a revised direction
  before more compute is committed.

## 2026-07-07

- Created remote workspace target: `/home/test/kkkk/25RABVS`.
- Paper abbreviation fixed as `RABVS`.
- Loaded the local story line and experimental plan.
- Current 7003 state: 8x RTX 4090D present, but all GPUs have active Python
  jobs. Strictly following the "use unoccupied GPUs" rule, the first run uses
  CPU/IO only and leaves at least 30 CPU cores free.
- Initial implementation scope:
  - S0 environment files and smoke/version scripts.
  - Public-data manifest/download scaffolding.
  - CPU label-oracle smoke benchmark for RABVS vs random, greedy surrogate,
    uncertainty sampling, and BOBa-style partition UCB.
  - Table generation only; no plotting.
- Core smoke passed in the independent `.conda` environment. Optional chemistry
  and GPU packages are intentionally recorded as missing in
  `artifacts/tool_versions.json` until the full stack is installed.
- Synthetic smoke result: fixed conservative RABVS improved risk/diversity
  metrics but lost Hit@100/500/1000 against greedy and uncertainty baselines.
- Focused diagnostic result: `rabvs_mean_final` recovered most hit performance
  while keeping the same exploration loop, so the first culprit is final
  conformal penalty strength rather than partition-UCB allocation alone.
- Literature collision check: BOBa already claims partition-bandit elimination
  of full-library inference, and a 2026 risk-aware prioritization preprint
  already claims split-conformal risk-aware library compression. RABVS should
  avoid framing as only either of those pieces.
- LIT-PCBA AVE unbiased downloaded from the official LIT-PCBA site. SHA256:
  `031703fec8f155a02a7ea0ffc851c0a6e4d57c489756faf9883e8e63a6918ac1`.
- Real-data pilot initially failed because `rdkit-pypi==2022.9.5` is compiled
  against NumPy 1.x and the environment had NumPy 2.2.6. Requirements are now
  pinned to `numpy<2`; rerun after environment downgrade.
- NumPy downgraded to 1.26.4 and RDKit smoke passed.
- LIT-PCBA AVE pilot completed on FEN1, MAPK1, PKM2, and VDR with 3 seeds.
  Main table: `tables/lit_pcba_ave_pilot/main_table.csv`; paired tests:
  `tables/lit_pcba_ave_pilot/paired_tests.csv`.
- Real-data conclusion: `rabvs_mean_final` preserves Hit@1000 relative to
  greedy surrogate while saving ~98.36% inference. `rabvs_risk_constrained`
  preserves near-BOBa hit performance and significantly improves
  PAINS/Brenk-free ratio and scaffold diversity.
- LIT-PCBA AVE all-15 pilot completed with 15 targets and 3 seeds. The original
  uncertainty-heavy allocation underperforms BOBa on Hit@1000.
- Allocation diagnostic completed. `rabvs_alloc_u0_mean` is not significantly
  different from BOBa on Hit@1000 (paired permutation p=0.14581), while
  `rabvs_alloc_u0_risk` significantly improves PAINS/Brenk-free ratio and
  unique scaffold count@1000 at a small hit cost.
- Method direction changed: use `rabvs_alloc_u0_mean` as the main hit mode and
  `rabvs_alloc_u0_risk` as the reliable/diverse recommendation mode. Treat the
  original fixed `+1.5U` allocation and `-0.7U_conf` final ranking as ablations.
- Canonical method names added: `rabvs_hit` and `rabvs_reliable`.
- LIT-PCBA AVE all-15 core table completed with stress metrics. Low-budget
  stress completed with total query budget 1024.
- Low-budget finding: `rabvs_hit` and `rabvs_reliable` are not significantly
  different from BOBa/greedy on Hit@1000, and `rabvs_reliable` significantly
  improves PAINS/Brenk-free ratio and unique scaffold count@1000.
- Reliability diagnostic completed. `rabvs_reliable_combo` is too conservative.
  False-confidence gap remains negative, so current uncertainty proxy cannot
  support a calibrated false-confidence claim yet.
- DUD-E fixed-20 external validation completed. All 20 target archives
  downloaded from the official DUD-E target pages and verified in
  `tables/dude_fixed20_dataset_table.csv`; total library size is 284,969
  molecules with 4,597 actives.
- DUD-E core result: the risk-allocation versions (`rabvs_hit` and
  `rabvs_reliable`) keep Hit@1000 close to BOBa, but significantly reduce AUBC.
  This means the partition-level risk penalty hurts early active acquisition on
  DUD-E and should not be the main allocation rule.
- DUD-E reliability diagnostic isolated the useful component: BOBa-style
  budgeted allocation plus RABVS risk-constrained final ranking keeps BOBa AUBC
  and Hit@1000 while significantly improving PAINS/Brenk-free ratio and
  scaffold diversity. This is now named `rabvs_budgeted_reliable`.
- Revised cross-dataset validation completed:
  - DUD-E fixed20: `rabvs_budgeted_reliable` vs BOBa has equal AUBC, Hit@1000
    difference -0.22 (p=0.465), PAINS/Brenk-free +0.273 (p=0), and
    unique scaffolds@1000 +105.35 (p=0).
  - LIT-PCBA AVE all15: equal AUBC, Hit@1000 difference -0.64 (p=0.0107),
    PAINS/Brenk-free +0.158 (p=0), and unique scaffolds@1000 +79.51 (p=0).
- Current paper direction: present `rabvs_budgeted_reliable` as the reliable
  recommendation mode with a small hit trade-off on LIT-PCBA and no significant
  Hit@1000 loss on DUD-E. Keep risk-aware partition allocation as an ablation,
  not as the main method.
- Final recommendation ablation completed on DUD-E fixed20 and LIT-PCBA AVE
  all15. Removing PAINS/Brenk alert removes almost all PAINS/Brenk-free gain;
  removing scaffold novelty removes almost all scaffold-count gain. Tight q50
  density filtering hurts Hit@1000. The density/conformal threshold itself is
  not the main source of improvement: `rabvs_budgeted_reliable_no_density`
  is statistically similar to q75 on hit, PAINS/Brenk-free ratio, and scaffold
  diversity across both datasets.
- Revised low-budget stress completed with total query budget 1024:
  - DUD-E fixed20 low-budget: `rabvs_budgeted_reliable` vs BOBa has Hit@1000
    difference +0.15 (p=0.790), PAINS/Brenk-free +0.306 (p=0), and
    unique scaffolds@1000 +121.5 (p=0).
  - LIT-PCBA AVE all15 low-budget: Hit@1000 difference -0.22 (p=0.549),
    PAINS/Brenk-free +0.181 (p=0), and unique scaffolds@1000 +91.76 (p=0).
  These results support the revised reliable recommendation mode under a
  stricter screening budget.
- Allocation-mechanics ablation completed. Random partition selection is a
  strong negative control on DUD-E: standard-budget AUBC drops by -0.193 and
  low-budget Hit@1000 drops by -13.87 against `rabvs_budgeted_reliable`. This
  supports target-aware chemical partition prioritization. Removing UCB is
  mixed: it improves standard-budget DUD-E AUBC/Hit@1000, but hurts low-budget
  DUD-E AUBC. Keep UCB/q75 as the main stable budgeted reliable method and
  report no-UCB as a hit-oriented/high-budget variant.
- DUD-E all102 completed. All 102 official target archives downloaded and
  verified in `data/raw/dude/manifest_all102.csv`; dataset table:
  `tables/dude_all102_dataset_table.csv`. Total all102 library size is
  1,434,019 molecules with 22,805 actives.
- DUD-E all102 standard budget: `rabvs_budgeted_reliable` keeps BOBa AUBC
  exactly, loses only -0.30 Hit@1000 (p=0.020), and improves PAINS/Brenk-free
  ratio by +0.314 plus unique scaffolds@1000 by +121.84. The no-UCB variant
  matches greedy-like hit performance while keeping inference savings, but its
  low-budget AUBC remains weaker.
- DUD-E all102 low-budget: `rabvs_budgeted_reliable` keeps BOBa AUBC, loses
  -0.43 Hit@1000 (p=0.044), and improves PAINS/Brenk-free by +0.338 plus
  unique scaffolds@1000 by +136.43.
- DUD-E all102 final-ablation plus LIT low-budget final-ablation show that q90
  and no-density are less conservative than q75 while preserving the
  PAINS/Brenk-free and scaffold-diversity gains. Current reporting plan:
  present q90 or no-density as the relaxed recommended reliable ranking, and
  q75 as the conservative setting; keep q50 as the too-strict ablation.
- Docking feasibility and candidate-panel validation completed on DUD-E AA2AR
  and ADRB2 using AutoDock Vina through the local `.conda` environment. The
  receptor/ligand smoke test now handles missing PDB element fields, Meeko
  receptor failures caused by inferred cross-residue bonds, and invalid crystal
  ligand MOL2 files via documented fallbacks. Smoke summary:
  `runs/docking_smoke/summary.csv`.
- Candidate-level docking panel completed for seed 0, top10 and top25
  candidates from BOBa and three RABVS reliable variants. Outputs:
  `runs/docking_candidates/docked_top10.csv`,
  `runs/docking_candidates/docked_top25.csv`,
  `tables/docking_candidate_summary.csv`,
  `tables/docking_candidate_summary_top25.csv`, and the corresponding
  delta/overlap tables.
- Docking-panel finding: AA2AR RABVS top25 is 92% overlapped with BOBa but
  reduces alert rate from 0.08 to 0.00 and slightly improves mean Vina score
  (-0.027 kcal/mol). ADRB2 is the stronger stress case: RABVS top25 has 0%
  overlap with BOBa, reduces alert rate from 1.00 to 0.00, and pays only a
  small mean Vina-score cost (+0.112 kcal/mol). This supports the claim that
  the reliable ranking can replace problematic high-score candidates with much
  cleaner alternatives without materially degrading docking affinity in this
  small validation panel.
- Expanded top10 docking panel to four additional DUD-E targets: EGFR, BACE1,
  CDK2, and PPARG. CDK2 required the same documented Meeko fallback class
  (delete unknown linking residue `:83`) and then docked successfully. Extra
  panel outputs: `runs/docking_candidates_extra/docked_top10.csv`,
  `tables/docking_candidate_summary_extra_top10.csv`, and
  `tables/docking_candidate_delta_extra_top10.csv`.
- Six-target top10 aggregate table generated:
  `tables/docking_candidate_aggregate_6target_top10.csv`. Across AA2AR, ADRB2,
  BACE1, CDK2, EGFR, and PPARG, all methods retain 100% known-active rate in
  the top10 panel. The three RABVS reliable variants reduce alert rate by
  0.317 on average versus BOBa, while the mean Vina-score change is only
  +0.063 kcal/mol and the median target-level change is +0.0007 kcal/mol.
  Interpretation: the reliable ranking mostly removes problematic chemistry
  while preserving docking quality; EGFR is the main target where the cleaner
  replacement set has a noticeable but still moderate docking-score cost.
- Six-target paired/sign-test table generated:
  `tables/docking_candidate_tests_6target_top10.csv`. RABVS is non-worse than
  BOBa on alert rate for all 6 targets and strictly improves 4/6 targets
  (sign-test p=0.03125). Mean Vina-score changes are not significant
  (Wilcoxon on nonzero deltas p=0.375), so this panel supports a chemistry-risk
  improvement without evidence of a systematic docking-affinity degradation.
- Six-target top10 docking was replicated for seeds 1 and 2, giving 18 paired
  target-seed observations. All 720 method-level candidates docked successfully.
  Across the three seeds, BOBa has alert rate 0.378 +/- 0.375, while all three
  RABVS reliable variants have 0.000 +/- 0.000 and retain 100% known-active
  rate. RABVS strictly improves alert rate in 14/18 pairs, ties in 4/18, and
  never worsens it (Wilcoxon p=9.45e-4; exact sign p=1.22e-4).
- A reproducibility audit found that the original docking worker seeded RDKit
  and Vina from row-order-dependent `dock_id`. This did not bias comparisons
  within a run, but made the same molecule vary across differently ordered
  files. Seeding is now a stable hash of target, canonical input SMILES, and
  experiment seed. Two independent 60-molecule smoke runs are byte-identical
  (docked CSV SHA256
  `5926cbbf0c8ef3b7b2f6cd46405b54645f457811f3f84f42d11420a6bc4df706`).
- The formal three-seed panel was rerun with stable ligand seeds. Mean Vina-score
  delta versus BOBa is +0.023 +/- 0.265 kcal/mol with median 0.000 (8 improved,
  4 tied, 6 worsened; Wilcoxon p=0.952). Thus the replicated result supports
  structural-alert removal without evidence for a systematic docking-affinity
  change; it does not support an affinity-gain claim. Canonical rerun outputs
  use the `tables/docking_stable_*` prefix.
- RABVS and BOBa candidate sets overlap by 0.622 +/- 0.375 across the 18
  target-seed pairs. The q75, q90, and no-density reliable variants are exactly
  identical in all 18 top10 sets, showing that the density threshold is inactive
  at this truncation depth. Multiseed outputs are under
  `tables/docking_stable_multiseed_*_top10.csv`.
- A two-target negative-control smoke compared RABVS top10 with random DUD-E
  actives and decoys. RABVS scores lower than random decoys by -1.271 kcal/mol
  on AA2AR (Mann-Whitney one-sided p=6.57e-4) and -0.836 kcal/mol on ADRB2
  (p=0.027). Random actives do not significantly beat decoys on either target.
  This validates basic discriminative behavior of the docking protocol without
  justifying a larger control sweep; the smoke-first promotion rule therefore
  stops this branch at two targets.


## 2026-07-08 True-Inference Scalability Audit and Smoke

- Audit finding: earlier DUD-E and LIT-PCBA pilot runners reported analytical
  `neural_inferences` savings, but the implemented call path still evaluated
  `predict_with_uncertainty(features, ...)` over the full pool each round. Those
  tables remain useful for hit/risk behavior, but should not be cited as actual
  surrogate-inference savings. New true-inference runners now count only the
  molecules actually scored by the neural surrogate.
- Metric audit: the legacy `aubc(round_hits, query_per_round)` normalizes by
  each method's own final hit count, so it measures curve shape rather than
  absolute recovery. Added `aubc_recall`, normalized by the target's total
  positive count plus initial hits, and used it in the new true-inference runs.
- Literature positioning: BOBa is the closest scalable backbone because it
  performs local inference only within selected partitions/arms. MolPAL and
  Deep Docking reduce acquisition or docking work, but still use full-library
  surrogate/model scoring in their standard loop. Position RABVS here as a
  risk-calibrated ranking and reliability layer on top of a BOBa-style
  local-inference allocation, not as the first partition-bandit inference idea.
  Sources checked: https://arxiv.org/pdf/2606.26657,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8188596/,
  https://www.nature.com/articles/s41596-021-00659-2, and
  https://pubs.acs.org/doi/10.1021/acscentsci.0c00229.
- Implemented true-inference runners and configs:
  `scripts/run_true_budgeted_dude.py`, `scripts/run_true_boba_dude.py`, and the
  `configs/dude_true_*` family. These record actual inference counts,
  scored-pool fractions, wall time, memory, Hit@K, risk-free ratio, and
  `AUBC_recall`.
- BOBA true-inference 5-target smoke results are summarized in
  `tables/true_inference_boba_panel5_smoke_summary.csv`. The best promoted
  setting is `boba_32_5` (`n_partitions=32`, `selected_partitions=5`): median
  Hit@1000 retention 0.973, worst 0.852, mean actual inference saving 0.760.
  More aggressive saving settings did not pass the smoke gate: `36/5` collapsed
  on ADRB2/CDK2, `40/5` saved 0.819 but median retention fell to 0.887, and
  `48/6` had worst retention 0.715.
- Fixed20 seed0 promotion for `boba_32_5` passed the main recall gate: median
  retention 0.948, worst 0.820, mean saving 0.753. This justified a fixed20
  three-seed run, but not an all102 promotion.
- Fixed20 three-seed result (`20 targets x 3 seeds`) is in
  `tables/true_inference_fixed20_parts5_*`. Overall target-seed means: Hit@1000
  retention 0.911 +/- 0.077 (median 0.923), actual inference saving
  0.756 +/- 0.025 (median 0.754), PAINS/Brenk-free@1000 delta +0.310 +/- 0.098,
  and online wall-clock ratio 0.569 +/- 0.101 versus full greedy. AUBC_recall
  is lower by -0.203 +/- 0.100, so this branch supports a scalability/risk
  tradeoff rather than an early-recovery gain claim.
- Important failure mode: AOFb is the weakest target (mean retention 0.787;
  seed1 retention 0.619). ADA seed1 and CP2C9 seed1 also fall below 0.8
  retention. Round logs show that low-active targets can suffer when UCB keeps
  exploring previously unproductive partitions after an initially good round.
- Diagnostic rescue attempts are in `tables/true_inference_failure_diagnostics.csv`.
  Arm-prior smoothing (`arm_prior_pulls` with lower UCB coefficients) and simply
  increasing `selected_partitions` from 5 to 6 did not recover the failing
  target-seed cases. These variants are not adopted.
- Manuscript decision: report this as a supplementary true-inference
  scalability experiment: RABVS/BOBa-style local inference can save about 75% of
  surrogate scoring on fixed20 while improving structural-alert cleanliness, but
  it has target-level worst-case instability. Do not claim uniform retention or
  promote to all102/full scalability until a stronger allocation fix is found.

## 2026-07-08 Retrieval-Calibrated Method Search

- Recent-method scan focused on adaptive virtual-screening ideas rather than
  heavier model training: GLARE-style dynamic relevance/diversity policies,
  BoBa-style arm-local inference, APEX-style approximate exhaustive search, and
  sequence/structure representation work. The implementable hypothesis was to
  keep the stable budgeted acquisition loop and add a retrieval-calibrated
  final ranking signal from labeled positives/negatives.
- Smoke-first branch results:
  - Policy-only allocation variants were not enough for a manuscript-level jump:
    no-UCB improves standard-budget hit/AUBC but remains mixed under low budget.
  - Acquisition-round retrieval can strongly improve standard-budget Hit@1000,
    but it hurts low-budget AUBC often enough that it is not adopted as a stable
    main method.
  - Final-only retrieval preserves the acquisition trajectory and only reranks
    the final recommendation pool. This is the adopted branch for the high-hit
    standard-budget story.
- Fixed20 three-seed promotion passed. Against original BOBa,
  `rabvs_budgeted_no_ucb` with final retrieval 0.5 improves AUBC by +0.0321,
  Hit@1000 by +7.07, PAINS/Brenk-free@1000 by +0.229, and unique
  scaffolds@1000 by +86.75 with no negative Hit@1000 rows across 60
  target-seed pairs. Against a fair BOBa plus the same final retrieval, it still
  improves AUBC by +0.0314, Hit@1000 by +2.32, PAINS/Brenk-free by +0.225,
  and scaffolds by +79.82.
- DUD-E all102 three-seed promotion also passed. Across 306 target-seed pairs,
  `rabvs_budgeted_no_ucb` with final retrieval 0.5 improves over original BOBa
  by +0.0277 AUBC, +5.74 Hit@1000, +0.242 PAINS/Brenk-free@1000, +95.92
  unique scaffolds@1000, and +1.52 hits per 1000 inferences. Hit@1000 is worse
  in only 7/306 rows. Against fair BOBa plus the same final retrieval, it keeps
  +0.0268 AUBC, +1.72 Hit@1000, +0.237 PAINS/Brenk-free, and +83.91 scaffolds.
- Low-budget final-retrieval is not promoted as a main all-metric claim. Fixed20
  low-budget is useful as a supplemental reliability result, but all102 seed0
  has non-negligible negative Hit@1000 cases and no AUBC gain. Following the
  smoke rule, the all102 low-budget final-retrieval branch is stopped rather
  than expanded to seeds 1/2.
- Cross-dataset LIT-PCBA smoke for final-only retrieval was stopped after four
  targets (FEN1, MAPK1, PKM2, VDR; seed0). It improved AUBC on 3/4 targets and
  improved PAINS/Brenk-free ratio plus scaffold diversity on all four, but
  Hit@1000 decreased on 3/4 targets (mean -2.25 for no-UCB vs original BOBa).
  This does not pass the promotion gate for all15. Table:
  `tables/lit_retrieval_final_smoke_summary.csv`.
- New reproducible tables:
  `tables/retrieval_final_comparison_summary.csv`,
  `tables/retrieval_final_decision_table.csv`,
  `tables/retrieval_final_fixed20_multiseed_summary.csv`,
  `tables/retrieval_final_all102_multiseed_summary.csv`, and
  `tables/retrieval_final_weakcases.csv`.

## 2026-07-08 Calibration Appendix Diagnostics

- Added an auxiliary split-calibration diagnostic for final queried labels. For
  each target-seed-method, 20% of the final labeled set is withheld as a
  calibration subset, the surrogate is refit on the remaining labeled molecules,
  and residual q90 coverage, risk-coverage AUC, ECE, Brier, and NLL are measured
  on the final unqueried pool. This diagnostic does not change acquisition or
  final ranking.
- Smoke on DUD-E AA2AR/ADRB2 passed: calibration columns were finite and in a
  sensible range. The residual q90 interval is conservative, giving coverage
  above the nominal 0.90 level.
- DUD-E fixed20 three-seed calibration appendix completed. For
  `rabvs_budgeted_no_ucb` with final retrieval 0.5 versus BOBa plus the same
  final retrieval, held-out test ECE improves by -0.0128, Brier by -0.0153, and
  NLL by -0.0370 over 60 target-seed pairs. Conformal coverage rises from 0.9765
  to 0.9965. Risk-coverage AUC changes only slightly and is not a strong claim.
- DUD-E all102 three-seed calibration appendix completed. Across 306
  target-seed pairs, held-out test ECE improves by -0.0135, Brier by -0.0155,
  and NLL by -0.0356; conformal coverage rises from 0.9705 to 0.9944. However,
  risk-coverage AUC slightly worsens (+0.0005, where lower is better), so the
  manuscript should claim better calibration error and conservative coverage,
  not improved risk-coverage sorting.
- Tables: `tables/calibration_appendix_summary.csv` and
  `tables/calibration_appendix_paired_deltas.csv`.

## 2026-07-08 Table 4 Scalability Status

- Checked the remaining plan gap: ZINC-22 lead-like 50M and Enamine REAL 10M
  require fixed external snapshots. These should not be marked complete without
  a real license-safe SMILES/CXSMILES snapshot.
- ZINC22 CartBlanche random lead-like endpoint is reachable and creates a task,
  but repeated polling of `/substance/random/{task}.txt` stayed pending or timed
  out even for a 20-molecule request. Audit table:
  `tables/external_library_access_audit.csv`.
- Enamine REAL Database and REAL subset pages are reachable and mention SMILES/SDF
  or CXSMILES subsets, but direct fixed subset downloads require site/store access.
  Do not report Enamine REAL 10M as completed until a fixed snapshot is obtained.
- Added a DUD-E-derived large-pool scalability stress test as an engineering
  fallback, explicitly not an external ZINC/Enamine result. At 1,000,000
  molecules, the local screening path scores 70,617 molecules (7.06%), saves
  92.94% surrogate inference, gives 2.48x online scoring speedup excluding
  partition construction, and retains 0.926 top1000 overlap with full scoring.
- Tables: `tables/large_pool_scalability_stress_summary.csv` and
  `tables/scalability_table4_status.csv`.

## 2026-07-08 Missing Component Ablations and BEDROC

- Added the clean reviewer-facing `rabvs_no_partition_reliable` ablation. It
  keeps the adopted retrieval-calibrated final ranking but acquires molecules
  globally by surrogate mean instead of selecting target-aware partitions.
- Added configurable ensemble size and compared the adopted five-model
  ensemble with a single-model surrogate under otherwise identical settings.
- Both branches passed a five-target seed0 smoke and were promoted to DUD-E
  fixed20 across three seeds (60 target-seed pairs). Raw runs are under
  `runs/dude_missing_ablation_fixed20_*`.
- Removing partitioning changes AUBC by +0.00182 (p=0.127), Hit@1000 by +0.283
  (p=0.257), and BEDROC(alpha=20) by +0.00077 (p=0.149). These efficacy
  differences are not significant. The analytical inference-budget proxy falls
  from 0.960 saved to zero; this is a budget accounting result, not a new
  true-inference timing claim. Partitioning should therefore be presented as a
  scalability mechanism rather than an accuracy booster.
- Replacing the five-model ensemble with one model significantly lowers AUBC
  by -0.00537 (p=5.36e-6), lowers BEDROC(alpha=20) by -0.00758 (p=2.09e-5),
  and worsens NLL by +0.00260 (p=1.59e-7). Hit@1000 changes by -0.467
  (p=0.0669), so the strongest ensemble claim is improved early enrichment and
  probability quality, not a large final-hit gain.
- BEDROC(alpha=20) is now implemented in the DUD-E runner and populated for
  this fixed20 experiment. Full LIT-PCBA all15 and DUD-E all102 manuscript
  table population remains separate work.
- Tables: `tables/missing_component_ablation_method_summary.csv` and
  `tables/missing_component_ablation_paired_summary.csv`.

## 2026-07-08 Exact Main-Table Metrics

- Added BEDROC(alpha=20), exact mean pairwise Morgan-fingerprint Tanimoto, and
  separate PAINS-free and Brenk-free ratios to both DUD-E and LIT-PCBA runners.
  The previous combined alert ratio remains the intersection-free ratio and is
  unchanged by the refactor.
- Both datasets passed two-target smoke tests before full promotion. Full runs
  cover DUD-E all102 across three seeds for the adopted high-hit method, the
  fair final-retrieval BOBa baseline, and original BOBa; LIT-PCBA covers all 15
  targets across three seeds and all five main-table methods.
- Against original BOBa on DUD-E all102, the adopted method improves AUBC by
  +0.02766, Hit@1000 by +5.768, BEDROC by +0.00280, PAINS-free ratio by
  +0.0444, Brenk-free ratio by +0.2275, combined alert-free ratio by +0.2415,
  and unique scaffolds by +96.17, while mean Tanimoto decreases by -0.00522.
- Against fair BOBa with the same final retrieval, AUBC (+0.02766), Hit@1000
  (+1.683), alert-free ratios, and scaffold count remain strongly better, but
  BEDROC decreases by -0.0110 and Tanimoto increases slightly by +0.00048.
  Report this as a final-hit/reliability trade-off, not uniform early-enrichment
  dominance.
- On LIT-PCBA all15, the reliable method keeps AUBC equal but changes Hit@1000
  by -0.644 and BEDROC by -0.07596 versus BOBa. It improves PAINS-free by
  +0.0305, Brenk-free by +0.1478, combined alert-free by +0.1585, unique
  scaffolds by +79.51, and lowers mean Tanimoto by -0.00163. This is a clear
  reliability/diversity Pareto result rather than an enrichment win.
- RDKit BEDROC values at perfect ranking were clipped from floating-point
  `1 + epsilon` to the valid [0, 1] interval without rerunning models.

## 2026-07-08 Original LIT-PCBA Docking Targets

- Added a LIT-PCBA docking candidate exporter and candidate-level docking worker
  for the original planned docking targets: ADRB2, ESR1_ago, FEN1, MAPK1,
  PKM2, and VDR. The worker converts LIT-PCBA protein MOL2 files to receptor
  PDB/PDBQT, uses the paired co-crystal ligand MOL2 to define the Vina box, and
  then docks ranked SMILES candidates through the same Meeko/Vina stack used by
  the DUD-E substitute panel.
- Smoke-first validation passed on all six original targets: 12/12 rank-1
  method rows docked successfully under `runs/lit_pcba_original_docking_asset_smoke/`.
- Promoted to the formal original-target top10 panel across three seeds, two
  methods (`boba_ucb` and `rabvs_budgeted_reliable`), and 18 target-seed pairs.
  All 360 method-level candidate rows docked successfully. Raw outputs are under
  `runs/lit_pcba_original_docking_seed{0,1,2}_top10/`.
- Multiseed result versus BOBa: risk-alert rate decreases by -0.1833
  (Wilcoxon p=9.62e-4; exact sign p=1.22e-4), mean Vina score improves by
  -0.3347 kcal/mol (Wilcoxon p=9.34e-3; sign p=4.90e-2), and best Vina score
  improves by -0.2603 kcal/mol (Wilcoxon p=7.81e-3; sign p=3.91e-2). Active
  rate changes by -0.050 and is not significant (p=0.107), so this panel should
  be written as a reliability/docking-affinity Pareto result, not a hit-rate win.
- Docking efficiency accounting is now populated for this protocol. Each method
  docks 10 candidates per target-seed, so docking_saved_ratio is 0.99 versus a
  top1000 docking budget and approximately 0.9979 versus the full LIT pool on
  average. `hits_per_1000_dockings` is 316.7 for BOBa and 266.7 for the reliable
  method; GPU-hours-per-hit is marked not applicable because this protocol uses
  CPU AutoDock Vina rather than GPU docking.
- Tables: `tables/lit_pcba_original_docking_multiseed_*_top10.csv`,
  `tables/lit_pcba_original_docking_efficiency_*_top10.csv`, and the seed-level
  `tables/lit_pcba_original_docking_*_seed*_top10.csv` files.

## 2026-07-08 LIT-PCBA Calibration Appendix and External Snapshot Recheck

- Rechecked the Table 4 external snapshot gap with a fresh live ZINC22 smoke
  request (`zinc-count=5`, 12 polls, 10 seconds per poll). The CartBlanche task
  still did not yield a stable local SMILES file before timeout, so ZINC-22 50M
  and Enamine REAL 10M remain not claimable as completed external fixed-snapshot
  benchmarks in this environment.
- Extended `scripts/summarize_calibration_appendix.py` to include
  `runs/exact_metrics_lit_all15/summary.csv`, producing combined DUD-E +
  LIT-PCBA appendix tables and LIT-only convenience tables.
- LIT-PCBA all15 calibration audit: `rabvs_budgeted_reliable` is exactly tied
  with `boba_ucb` on ECE, Brier, NLL, held-out calibration metrics, conformal
  coverage, residual q90, risk-coverage AUC, and false-confidence gap across
  45 target-seed pairs. This means the LIT reliability/diversity result comes
  from final ranking, not improved probability calibration.
- LIT-PCBA conformal coverage is conservative for both BOBa and the reliable
  method (mean coverage 0.9865 for a nominal 0.90 interval). Greedy surrogate
  has lower calibration errors, but it is a full-inference reference rather than
  the budgeted method.
- Tables: updated `tables/calibration_appendix_summary.csv` and
  `tables/calibration_appendix_paired_deltas.csv`; added
  `tables/lit_pcba_calibration_appendix_summary.csv` and
  `tables/lit_pcba_calibration_appendix_paired_deltas.csv`.

## 2026-07-08 Submission Table and Plan-Completion Audit

- Added `scripts/build_submission_audit_tables.py` to make the remaining
  manuscript-facing plan tables reproducible from existing experiment outputs.
- Generated `tables/table1_dataset_overview.csv` and
  `tables/table1_dataset_overview_summary.csv` from actual exact-metric run
  summaries rather than raw file counts. This matters for LIT-PCBA because the
  main experiments use the V split with `max_inactives=50000`; the table now
  matches the actual screening pools.
- Generated `tables/table5_ablation_summary.csv`, a compact reviewer-facing
  ablation table with nine core rows: no partitioning, single-model surrogate,
  no-UCB allocation, random partition allocation, no alert penalty, no novelty,
  q50/q90 conformal thresholds, and final-only retrieval calibration. The
  headline conclusions are: partitioning is mainly a scalability mechanism,
  the ensemble improves quality/calibration, random partition allocation is much
  worse, alert and novelty terms trade final hits for reliability/diversity, and
  final-only retrieval gives the strongest DUD-E high-hit branch.
- Generated `tables/experiment_plan_completion_audit.csv` and
  `docs/EXPERIMENT_COMPLETION_AUDIT.md`. All planned tables/appendices are now
  complete or explicitly blocked, with the only blocked item being the external
  fixed ZINC-22/Enamine snapshot for Table 4.


## 2026-07-08 Literature-Inspired Pareto-RRF Method Search

- Searched recent preprints (2025-07-08 to 2026-07-08) around structure-aware
  retrieval, scaffold-aware virtual screening reranking, ultra-large approximate
  screening, active-learning search, and multi-stage docking/scoring. The useful
  directions were retrieval-style local evidence plus scaffold/cleanliness-aware
  reciprocal-rank-fusion reranking.
- Added Pareto-RRF final ranking variants:
  `rabvs_budgeted_no_ucb_pareto_rrf`, `_novel`, `_balanced`, and `_clean`.
  Smoke-first policy was followed: `_clean` stopped at smoke; `_novel` and
  `_balanced` were promoted to DUD-E all102 and LIT-PCBA all15 across three seeds.
- Recommended main branch: `rabvs_budgeted_no_ucb_pareto_rrf_novel`. Versus fair
  BOBa on DUD-E all102 it improves AUBC by +0.0277, Hit@500 by +7.08, Hit@1000
  by +4.82, BEDROC by +0.0146, alert-free@1000 by +0.0489, and NLL by -0.0065;
  unique scaffold count is only -2.13 and not significant (paired permutation
  p=0.096), while mean Tanimoto improves by -0.0018.
- On LIT-PCBA all15 the same branch improves Hit@100 by +2.47, Hit@500 by
  +20.64, Hit@1000 by +30.56, BEDROC by +0.1726, alert-free@1000 by +0.0363,
  unique scaffolds by +34.13, mean Tanimoto by -0.0049, and NLL by -0.0068.
- `rabvs_budgeted_no_ucb_pareto_rrf_balanced` is a risk-sensitive appendix branch:
  it gives stronger LIT Hit@1000/BEDROC/risk and DUD alert-free@1000, but has a
  small non-significant DUD Hit@100 decrease and a larger DUD unique-scaffold
  reduction.
- Detailed note: `docs/LITERATURE_INSPIRED_PARETO_RRF_RESULTS.md`. Tables:
  `tables/dude_literature_inspired_rrf_weight_all102/`,
  `tables/lit_pcba_literature_inspired_rrf_weight_all15/`, and
  `tables/literature_inspired_rrf_weight_full_deltas.csv`.
