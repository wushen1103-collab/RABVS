# Literature-Inspired Pareto-RRF Results

Date: 2026-07-08

This note records the preprint-inspired method search requested after the main
experiment audit. No plots were generated.

Update: later full component ablation superseded the `novel` recommendation.
The final recommended branch is now
`rabvs_budgeted_no_ucb_pareto_rrf_no_conf`; see
`docs/PARETO_NO_CONF_FINAL_RESULTS.md`.

## Preprint Scan

Recent preprints that motivated this round:

| Paper | Link | Transferable idea |
|---|---|---|
| MolE-RAG: Molecular Structure-Enhanced Retrieval-Augmented Generation for Chemistry | https://arxiv.org/abs/2606.05693 | Use structure-aware retrieval/local evidence as a training-free ranking signal. |
| Scaffold-Aware Generative Augmentation and Reranking for Enhanced Virtual Screening | https://arxiv.org/abs/2510.16306 | Add scaffold-aware reranking so high-score regions do not collapse onto one chemistry family. |
| APEX: Approximate-but-exhaustive search for ultra-large combinatorial synthesis libraries | https://arxiv.org/abs/2510.24380 | Treat ultra-large virtual screening as approximate top-k search under strict scoring budgets. |
| Why Pool When You Can Flow? Active Learning with GFlowNets | https://arxiv.org/abs/2509.00704 | Avoid exhaustive pool scoring; use selection policies that preserve exploration and sample efficiency. |
| Boltzina: Efficient and Accurate Virtual Screening via Docking-Guided Binding Prediction with Boltz-2 | https://arxiv.org/abs/2508.17555 | Use multi-stage rescoring only after cheap screening; kept as a docking-stage future direction here. |

The implemented idea is a final-stage Pareto reciprocal-rank-fusion (RRF)
reranker over surrogate probability, structural-alert cleanliness,
confidence/density, and scaffold novelty. The active-learning acquisition is
unchanged from `rabvs_budgeted_no_ucb`, so AUBC remains comparable; the gain is
from the final hit-list order.

## Variants Tested

| Variant | Description | Decision |
|---|---|---|
| `rabvs_budgeted_no_ucb_pareto_rrf` | Base RRF: probability-dominant with light clean/confidence/novelty terms | Strong; superseded by weighted variants. |
| `rabvs_budgeted_no_ucb_pareto_rrf_novel` | Stronger novelty term while preserving probability dominance | Superseded by final `no_conf` branch after full component ablation. |
| `rabvs_budgeted_no_ucb_pareto_rrf_balanced` | Stronger clean + novelty terms | Risk-sensitive appendix branch. |
| `rabvs_budgeted_no_ucb_pareto_rrf_clean` | Stronger clean term only | Stopped at smoke; weaker hit gain. |

Smoke-first policy was followed. `clean` was not promoted because it did not
show a large enough hit/enrichment gain.

## Full Results Versus Fair BOBa

All rows use final-only retrieval blend 0.5 for every method in the comparison.
DUD-E has 306 target-seed pairs; LIT-PCBA has 45 target-seed pairs. Lower NLL
and lower mean Tanimoto are better.

| Dataset | Main method | AUBC delta | Hit@100 delta | Hit@500 delta | Hit@1000 delta | BEDROC delta | Alert-free@1000 delta | Unique scaffolds@1000 delta | Tanimoto@1000 delta | NLL delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DUD-E all102 | `rabvs_budgeted_no_ucb_pareto_rrf_novel` | +0.0277 | +0.0294 | +7.0752 | +4.8203 | +0.0146 | +0.0489 | -2.1307 | -0.0018 | -0.0065 |
| LIT-PCBA all15 | `rabvs_budgeted_no_ucb_pareto_rrf_novel` | +0.0061 | +2.4667 | +20.6444 | +30.5556 | +0.1726 | +0.0363 | +34.1333 | -0.0049 | -0.0068 |
| DUD-E all102 | `rabvs_budgeted_no_ucb_pareto_rrf_balanced` | +0.0277 | -0.0523 | +6.4869 | +4.4641 | +0.0138 | +0.0850 | -12.7516 | -0.0013 | -0.0065 |
| LIT-PCBA all15 | `rabvs_budgeted_no_ucb_pareto_rrf_balanced` | +0.0061 | +3.2889 | +20.2444 | +31.5111 | +0.1748 | +0.0641 | +33.0889 | -0.0046 | -0.0068 |

## Paired Significance Highlights

For the recommended `novel` branch versus fair BOBa:

| Dataset | Metric | Mean delta | Paired permutation p | Direction count |
|---|---|---:|---:|---|
| DUD-E all102 | Hit@500 | +7.0752 | <1e-5 | 171 positive / 131 zero / 4 negative |
| DUD-E all102 | Hit@1000 | +4.8203 | <1e-5 | 150 positive / 155 zero / 1 negative |
| DUD-E all102 | BEDROC alpha20 | +0.0146 | <1e-5 | 216 positive / 16 zero / 74 negative |
| DUD-E all102 | Alert-free@1000 | +0.0489 | <1e-5 | 277 positive / 2 zero / 27 negative |
| DUD-E all102 | Unique scaffolds@1000 | -2.1307 | 0.096 | 137 positive / 8 zero / 161 negative |
| LIT-PCBA all15 | Hit@100 | +2.4667 | 1e-5 | 20 positive / 24 zero / 1 negative |
| LIT-PCBA all15 | Hit@500 | +20.6444 | <1e-5 | 24 positive / 21 zero / 0 negative |
| LIT-PCBA all15 | Hit@1000 | +30.5556 | <1e-5 | 24 positive / 21 zero / 0 negative |
| LIT-PCBA all15 | BEDROC alpha20 | +0.1726 | <1e-5 | 36 positive / 7 zero / 2 negative |
| LIT-PCBA all15 | Alert-free@1000 | +0.0363 | <1e-5 | 29 positive / 13 zero / 3 negative |
| LIT-PCBA all15 | Unique scaffolds@1000 | +34.1333 | <1e-5 | 32 positive / 12 zero / 1 negative |

## Decision

Historical decision: this note originally promoted
`rabvs_budgeted_no_ucb_pareto_rrf_novel`. A later full component ablation found
that removing the confidence/density rank improves or preserves hits, BEDROC,
alert-free ratio, and scaffold diversity on both DUD-E and LIT-PCBA. The final
main branch is therefore `rabvs_budgeted_no_ucb_pareto_rrf_no_conf`.

## Claim Boundaries

- This branch is a final reranking improvement, not a new acquisition policy;
  AUBC matches `rabvs_budgeted_no_ucb` because the queried set is unchanged.
- The external ZINC-22/Enamine fixed-snapshot gap remains open.
- Docking has since been rerun for the final `no_conf` branch and supports
  candidate-quality improvement without significant affinity degradation.
- Do not claim the stopped `clean` smoke variant.

## Artifacts

- Smoke outputs: `runs/dude_literature_inspired_smoke/`,
  `runs/lit_pcba_literature_inspired_smoke/`,
  `runs/dude_literature_inspired_rrf_weight_smoke/`,
  `runs/lit_pcba_literature_inspired_rrf_weight_smoke/`.
- Full outputs: `runs/dude_literature_inspired_rrf_weight_all102/`,
  `runs/lit_pcba_literature_inspired_rrf_weight_all15/`.
- Tables: `tables/dude_literature_inspired_rrf_weight_all102/`,
  `tables/lit_pcba_literature_inspired_rrf_weight_all15/`,
  `tables/literature_inspired_rrf_weight_full_deltas.csv`.
- Final no_conf outputs: `runs/dude_pareto_component_ablation_full/`,
  `runs/lit_pcba_pareto_component_ablation_full/`,
  `tables/pareto_no_conf_main_claim_summary.csv`.
