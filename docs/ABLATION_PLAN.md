# Ablation Plan

This plan follows the ablation-planner rule: every ablation must isolate a
reviewer-visible question and have an expected outcome.

## Component Ablations

| # | Name | What It Tests | Expected If Component Matters | Priority |
|---|------|---------------|-------------------------------|----------|
| 1 | w/o partitioning | Does arm-level allocation help beyond molecule-level scoring? | Lower AUBC and Hit@1000, especially under low inference budgets. | 1 |
| 2 | w/o UCB exploration | Is partition exploration needed to avoid early exploitation traps? | Faster early saturation and worse OOD scaffold hit rate. | 1 |
| 3 | w/o conformal uncertainty | Does calibrated uncertainty reduce false-confidence picks? | Worse ECE, worse risk-coverage AUC, lower PAINS/Brenk-free ratio. | 1 |
| 4 | w/o ensemble uncertainty | Does epistemic uncertainty improve target transfer? | Lower AUBC on sparse targets and unstable target-wise variance. | 2 |
| 5 | w/o risk penalty | Is explicit risk control needed in final recommendations? | More high-risk alerts among top-K and weaker conservative ranking. | 2 |
| 6 | w/o scaffold diversity term | Is diversity active rather than incidental? | Fewer unique scaffolds@1000 and higher mean Tanimoto. | 2 |
| 7 | random partitions | Are chemical partitions better than arbitrary arms? | Worse partition hit concentration and lower budget efficiency. | 2 |
| 8 | mean-score final ranking | Is conservative final ranking useful after optimistic exploration? | Similar raw predicted scores but worse calibration/risk metrics. | 1 |
| 9 | remove partition uncertainty bonus | Does uncertainty-guided arm allocation help sparse targets? | If the bonus is harmful, Hit@1000 recovers when it is removed. | 1 |
| 10 | risk-constrained final ranking | Is reliability/diversity improved at acceptable hit cost? | PAINS/Brenk-free ratio and scaffold diversity improve, with small Hit@1000 cost. | 1 |
| 11 | false-confidence diagnostic | Does current uncertainty identify unreliable high-score compounds? | High-score/high-uncertainty hit rate should be lower; current pilot fails this. | 1 |

## Suggested Order

1. Run the three core ablations: w/o partitioning, w/o UCB exploration,
   w/o conformal uncertainty.
2. Run final-ranking and random-partition checks.
3. Run risk/diversity/ensemble ablations.
4. Run low-budget, OOD scaffold, and false-confidence stress tests.

## Compute Notes

Use a smoke-first promotion policy for every new experiment branch. Start with
the smallest representative target/seed/budget subset that can validate the
pipeline, metric direction, and failure modes. Promote to a full
LIT-PCBA/DUD-E sweep only when the smoke result is directionally useful, no
implementation or data anomaly remains, and broader statistical evidence is
needed for a manuscript claim. Diagnose or redirect failed and ambiguous smoke
runs before committing more compute.

## Completed Diagnostic Update

DUD-E fixed20 showed that partition-level risk penalties are not universally
helpful: they significantly lower AUBC against BOBa. The useful component is
the final risk-constrained recommendation layer. The revised main reliable mode
is therefore `rabvs_budgeted_reliable`, while `rabvs_hit` and
`rabvs_reliable` remain ablations for risk-aware allocation.

Final recommendation ablations are now complete. The reviewer-facing answer is:

- PAINS/Brenk alert is needed for structural-alert cleanliness.
- Scaffold novelty is needed for scaffold-count gains.
- Density/conformal threshold is optional at q75 and harmful when made too
  strict at q50.
- Low-budget stress confirms the revised reliable mode preserves Hit@1000
  relative to BOBa while improving PAINS/Brenk-free ratio and scaffold diversity.

Allocation-mechanics ablations are also complete:

- Random partition selection is a useful negative control and fails on DUD-E,
  especially in low-budget Hit@1000.
- Removing UCB is not uniformly better: it improves standard-budget DUD-E but
  loses DUD-E low-budget AUBC. The main method should keep UCB for budget
  stability, while no-UCB can be reported as a high-hit variant.

All102 DUD-E confirms these ablations at scale. The remaining table work should
separate:

- conservative reliable ranking: q75;
- relaxed reliable ranking: q90 or no-density;
- high-hit allocation variant: no-UCB;
- negative control: random partitions.


## True-Inference Scalability Update

The true-inference branch followed the smoke-first rule. `boba_32_5` is the only
setting that passed both the 5-target smoke and the fixed20 seed0 promotion
check. On fixed20 across three seeds it gives mean Hit@1000 retention 0.911,
mean actual inference saving 0.756, and mean PAINS/Brenk-free@1000 improvement
+0.310 versus full greedy. However, AOFb/ADA/CP2C9 seed-level failures show that
UCB allocation is not uniformly stable on low-active targets.

Reviewer-facing interpretation: use this as a supplementary scalability and
risk-cleanliness tradeoff table, not as a main theorem-style claim. Additional
allocation work is needed before all102 promotion. Negative diagnostics to cite
if asked: arm-prior smoothing and `selected_partitions=6` did not fix the
worst-case failures.

## Retrieval-Calibrated Ranking Update

A recent-method search suggested a stronger reviewer-facing variant: keep the
stable budgeted acquisition loop, then apply retrieval-calibrated final ranking
using nearest labeled positives and negatives. This tests whether the final
recommendation layer can exploit local active evidence without destabilizing
online acquisition.

Adopted branch:

- `retrieval_round_blend=0.0`, `retrieval_final_blend=0.5`.
- Standard-budget high-hit mode: `rabvs_budgeted_no_ucb`.
- Conservative mode remains `rabvs_budgeted_reliable` for low-budget or
  reliability-first reporting.

Reviewer-facing ablation conclusions:

- Acquisition-round retrieval is a negative/ambiguous ablation because it can
  harm low-budget AUBC.
- Final-only retrieval is the clean component: it improves final Hit@1000 while
  preserving AUBC and the risk/scaffold gains from RABVS ranking.
- A fair baseline where BOBa also receives the same final retrieval should be
  reported. RABVS still wins AUBC, PAINS/Brenk-free ratio, and scaffold diversity
  strongly, with a smaller but significant Hit@1000 gain in standard budget.
- Do not promote all102 low-budget final retrieval: seed0 smoke shows useful
  Hit/risk/scaffold gains but no AUBC gain and too many target-level Hit losses.

## Calibration Diagnostic Update

The split-calibration appendix answers the uncertainty reviewer question without
overclaiming conformal control. Final-retrieval no-UCB improves ECE, Brier, and
NLL on the held-out unqueried pool and gives conservative residual q90 coverage.
Risk-coverage AUC is slightly worse on all102, so the paper should not claim that
the uncertainty score ranks low-error molecules better; instead, claim improved
probability calibration and conservative coverage for the adopted high-hit
ranking variant.

## Scalability Table 4 Update

The ZINC/Enamine row should remain conservative. The repository now contains an
access audit and a 1M DUD-E-derived large-pool stress test, but not a licensed
fixed ZINC-22 or Enamine REAL snapshot. Review-facing wording should separate
these: the stress test supports engineering scalability of local scoring, while
ZINC/Enamine are pending external-snapshot benchmarks.

## Missing Component Ablation Update

The clean no-partition and no-ensemble ablations are now complete on DUD-E
fixed20 across three seeds after passing a five-target smoke.

- Removing partitioning does not significantly change AUBC, Hit@1000, or
  BEDROC(alpha=20), while it removes the analytical inference-budget saving.
  This supports partitioning as the mechanism that makes local screening
  scalable, not as an accuracy-improvement module.
- Replacing the five-model ensemble with a single model significantly lowers
  AUBC and BEDROC and worsens NLL. The ensemble contribution is therefore
  strongest for early enrichment and probability quality; the final Hit@1000
  decrease is small and not significant at 0.05.
- These experiments close planned component-ablation items 1 and 4. Their
  canonical statistics are in
  `tables/missing_component_ablation_paired_summary.csv`.
