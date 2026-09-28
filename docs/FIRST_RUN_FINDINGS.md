# First Run Findings

Date: 2026-07-07

## What Ran

- Independent core environment: `/home/test/kkkk/25RABVS/.conda`.
- Smoke test: `scripts/smoke_test.py`.
- Resource snapshot: `artifacts/resource_snapshot.json`.
- Version freeze: `artifacts/tool_versions.json`.
- Synthetic label-oracle smoke:
  - `runs/synthetic_smoke/summary.csv`
  - `tables/synthetic_smoke_main_table.csv`
- Focused conservative-ranking diagnostic:
  - `runs/synthetic_diagnostic/summary.csv`
  - `tables/diagnostic/synthetic_smoke_main_table.csv`

## Main Signal

The fixed RABVS formula has a real early warning:

- It is much more inference-efficient than full greedy surrogate in the
  synthetic accounting.
- It improves risk/diversity style metrics.
- But the fixed final score `mu - 0.7 * U_conf + novelty` loses too many hits.

In the diagnostic, `rabvs_mean_final` keeps the same RABVS exploration loop but
uses mean-score final ranking. It recovers Hit@1000 from about 322 to about 619,
near BOBa and much closer to greedy. Therefore, the immediate issue is final
ranking risk calibration, not just partition allocation.

## Interpretation

The current conformal/density proxy is acting as a harsh OOD penalty. That is
useful for a reliability claim, but it can erase the exact OOD hits that virtual
screening wants. The next version should not hard-code `0.7` as a universal
penalty. It should tune risk aversion by validation coverage or by reporting a
Pareto frontier:

- max-hit ranking
- conservative ranking
- risk-constrained top-K ranking

## Collision Check

Recent related work makes the novelty boundary tight:

- BOBa, submitted June 25, 2026, frames partition-bandit allocation as a way to
  eliminate full-library inference in huge chemical spaces.
  https://arxiv.org/abs/2606.26657
- A risk-aware compound-library prioritization preprint, also submitted
  June 25, 2026, already combines 2D activity proxy, split-conformal intervals,
  ADMET/structural alerts, leakage auditing, and auditable export.
  https://arxiv.org/abs/2606.26624

RABVS should therefore be framed as the joint, target-aware, calibrated,
budgeted closed loop with explicit inference/docking accounting, not as only a
partition bandit or only risk-aware compression.

## Next Decision

Do not launch full LIT-PCBA/DUD-E training with the fixed final penalty yet.
First run a small real-data pilot or a stronger synthetic stress test with:

1. `rabvs_mean_final`
2. `rabvs_conf03`
3. fixed conservative `rabvs`
4. risk-constrained top-K where candidates are filtered by conformal risk then
   ranked by `mu`

## LIT-PCBA AVE Pilot Update

The AVE-unbiased LIT-PCBA archive was downloaded from the official site and
verified by SHA256:

`031703fec8f155a02a7ea0ffc851c0a6e4d57c489756faf9883e8e63a6918ac1`

Pilot targets: FEN1, MAPK1, PKM2, VDR. Pool: validation split, all actives,
and up to 50,000 inactives per target. This is not the final benchmark table,
but it is the first real-label diagnostic.

Key result:

- `rabvs_mean_final` is statistically indistinguishable from greedy surrogate
  on Hit@1000, while saving about 98.36% of neural inference.
- `rabvs_risk_constrained` is also close on Hit@1000 and significantly improves
  PAINS/Brenk-free ratio and unique scaffold count@1000.
- The fixed conservative `rabvs` remains too harsh: it improves risk metrics
  but loses too many hits.

Immediate direction: keep optimistic partition exploration, but present final
recommendation as a choice between mean-score ranking and risk-constrained
ranking. Do not use a fixed `-0.7 * U_conf` penalty as the only main result.

## LIT-PCBA AVE All-15 Update

The all-15 pilot exposed a second issue: the original allocation score's
`+1.5 * uncertainty` term is too aggressive on ultra-low-active targets.
Compared with BOBa, `rabvs_mean_final` loses Hit@1000 significantly across
45 target-seed pairs.

Allocation diagnostic:

- `rabvs_alloc_u0_mean`: remove the partition-level uncertainty bonus while
  keeping TopMean + UCB + diversity - risk. This is statistically
  indistinguishable from BOBa on Hit@1000 and hits per 1000 inferences.
- `rabvs_alloc_u0_risk`: same allocation, risk-constrained final ranking. It
  loses a small amount of Hit@1000 relative to BOBa but significantly improves
  PAINS/Brenk-free ratio and unique scaffold count@1000.

Recommended method update:

- Main hit-enrichment mode: `rabvs_alloc_u0_mean`.
- Reliability/diversity mode: `rabvs_alloc_u0_risk`.
- Original fixed `rabvs` is now an ablation/failure case, not the main method.

## Canonical Naming and Stress Update

The stable method names are now:

- `rabvs_hit`: main hit-enrichment mode. It removes the partition-level
  uncertainty bonus and uses mean-score final ranking.
- `rabvs_reliable`: reliability/diversity mode. It uses the same allocation but
  applies risk-constrained final ranking.

Extreme low-budget pilot:

- Query budget: initial 256 + 6 rounds x 128 = 1024 total queried molecules.
- `rabvs_hit` is not significantly different from BOBa or greedy on Hit@1000
  and keeps the large inference-saving advantage.
- `rabvs_reliable` is not significantly different from BOBa/greedy on Hit@1000
  under this low budget, while significantly improving PAINS/Brenk-free ratio
  and scaffold diversity.

Reliability diagnostic:

- `rabvs_reliable` and `rabvs_reliable_q90` have similar hit/risk behavior.
- `rabvs_reliable_combo` is too conservative: strong risk/diversity gains but
  unacceptable hit loss.
- False-confidence gap remains negative across BOBa and RABVS variants. This
  means the current lightweight uncertainty proxy does not yet support a strong
  "high-score high-uncertainty compounds are less reliable" claim.

Claim discipline:

- Keep the current reliable-mode claim to structural/risk-alert cleanliness and
  scaffold diversity.
- Do not claim calibrated false-confidence control until a proper conformal
  calibration layer is implemented and passes this diagnostic.

## DUD-E External Validation and Method Pivot

DUD-E fixed20 was added as an external validation set: 20 official DUD-E target
archives, 284,969 total molecules, and 4,597 actives. The first DUD-E core run
exposed a real issue: risk-aware partition allocation keeps Hit@1000 close to
BOBa but significantly reduces AUBC. This means risk should not be applied as a
default partition-level penalty.

The diagnostic that separated allocation from final ranking found the better
main method:

- `rabvs_budgeted_reliable`: BOBa-style budgeted allocation plus RABVS
  risk-constrained final recommendation.
- DUD-E fixed20 vs BOBa: equal AUBC, Hit@1000 -0.22 (not significant),
  PAINS/Brenk-free ratio +0.273, and unique scaffolds@1000 +105.35.
- LIT-PCBA AVE all15 vs BOBa: equal AUBC, Hit@1000 -0.64 (significant but
  small), PAINS/Brenk-free ratio +0.158, and unique scaffolds@1000 +79.51.

This changes the paper framing: RABVS should claim budgeted, target-aware,
risk-constrained recommendation with explicit hit/reliability trade-off. The
risk-aware allocation rule remains an ablation/failure case, not the main
algorithm.

## Final Recommendation Ablation and Low-Budget Stress

The final recommendation layer was decomposed into density threshold, structural
alert penalty, and scaffold novelty terms.

- Removing PAINS/Brenk alert collapses the PAINS/Brenk-free gain.
- Removing scaffold novelty collapses the scaffold diversity gain.
- Removing the density threshold is almost neutral; q50 is too strict and hurts
  Hit@1000. Therefore the current density proxy should be described as optional
  caution rather than the core source of reliability.

Low-budget stress with only 1024 queried compounds strengthens the story:

- DUD-E fixed20: `rabvs_budgeted_reliable` has no significant Hit@1000 loss
  relative to BOBa and improves PAINS/Brenk-free ratio by 0.306 plus
  unique scaffolds@1000 by 121.5.
- LIT-PCBA AVE all15: Hit@1000 difference is not significant and
  PAINS/Brenk-free ratio/scaffold diversity both improve strongly.

The next missing reviewer-facing ablation is allocation mechanics: random
partitions and no-UCB allocation, to show whether BOBa-style chemical arms are
actually needed beyond final post-processing.

## Allocation Mechanics Update

The allocation ablation answers that open question:

- Random partition selection is a strong negative control on DUD-E. It sharply
  lowers standard-budget AUBC and low-budget Hit@1000, so chemically meaningful
  partition prioritization is necessary.
- Removing UCB is mixed. It improves standard-budget DUD-E, but loses
  low-budget DUD-E AUBC. On LIT-PCBA the differences are mostly small.

Interpretation: UCB is best framed as a budget-stability mechanism rather than
the source of final hit enrichment. The main method should keep the UCB/q75
configuration for robustness across budgets, and no-UCB should be reported as
a high-budget/high-hit variant.

## DUD-E All102 Update

The DUD-E result is now expanded from fixed20 to all 102 official targets:

- Dataset size: 1,434,019 molecules and 22,805 actives.
- Standard budget: `rabvs_budgeted_reliable` keeps BOBa AUBC unchanged and
  improves PAINS/Brenk-free ratio by 0.314 plus unique scaffolds@1000 by 121.84,
  with only a small Hit@1000 loss.
- Low budget: the same pattern holds, with PAINS/Brenk-free +0.338 and unique
  scaffolds@1000 +136.43.
- Random partitions remain a strong negative control, confirming that target
  aware partition prioritization is necessary.

The all102 final-ranking ablation sharpens the method choice. q75 is a
conservative setting, but q90 and no-density are less conservative and usually
recover Hit@1000 without losing structural-alert cleanliness or scaffold
diversity. For the paper, report q90/no-density as relaxed recommended variants
and q75 as the conservative variant. Do not overclaim conformal calibration:
the density threshold is optional, while alert and novelty terms drive the
reliable recommendation gains.

## Clean Partition and Ensemble Ablations

On DUD-E fixed20 across three seeds, globally acquiring by surrogate mean while
keeping the same reliable final ranking does not significantly improve AUBC,
Hit@1000, or BEDROC over target-aware partition acquisition. It does remove the
entire analytical inference-budget saving, so partitioning is best understood
as the scalability mechanism rather than a source of predictive lift.

Using one surrogate model instead of the five-model ensemble significantly
reduces AUBC and BEDROC and worsens NLL. The ensemble has a reproducible role in
early enrichment and probability quality, although its Hit@1000 effect is small
and does not reach p<0.05.

## Exact Diversity and Early-Enrichment Metrics

The exact metric rerun confirms that the structural-alert result is not an
artifact of combining PAINS and Brenk filters. On DUD-E all102 versus original
BOBa, PAINS-free improves by 0.044 and Brenk-free by 0.228; every target-seed
pair improves on combined alert-free ratio. Mean Morgan Tanimoto also decreases
by 0.0052 while unique scaffold count rises by 96.2.

The fair final-retrieval BOBa comparison exposes an important boundary: RABVS
still improves AUBC, final hits, alert cleanliness, and scaffold count, but its
BEDROC is lower and mean Tanimoto is nearly tied. LIT-PCBA similarly trades
early BEDROC and a small number of final hits for substantially cleaner and
more diverse recommendations. The paper should present the method as a
