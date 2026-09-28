# RABVS Experiment Plan Extract

## Claim

Target-aware partitioning, partition-UCB allocation, and conformal risk-aware
candidate ranking should discover reliable, diverse hits with less neural
inference and less docking than full-library scoring.

## Fixed Modules

1. Target-Aware Library Partitioning.
2. Partition-UCB Budget Allocation.
3. Conformal Risk-Aware Candidate Ranking.

## Main Benchmarks

- LIT-PCBA: all 15 targets.
- DUD-E: all 102 targets.
- ZINC-22 lead-like subset: fixed 50M molecules.
- Enamine REAL subset: fixed 10M molecules.
- Docking oracle targets: ADRB2, ESR_ago, FEN1, MAPK1, PKM2, VDR.

## Metrics

- Retrospective: Hit@100, Hit@500, Hit@1000, Recall@1000, EF@0.1%,
  EF@1%, BEDROC(alpha=20), AUBC.
- Uncertainty: ECE, Brier score, NLL, 90% conformal coverage,
  risk-coverage AUC.
- Efficiency: inference_saved_ratio, docking_saved_ratio,
  hits_per_1000_inferences, hits_per_1000_dockings, GPU_hours_per_hit.
- Diversity: unique_scaffold_count@1000, mean pairwise Tanimoto,
  PAINS-free ratio, Brenk-free ratio.

## Table Targets

- Table 1: dataset, target, library size, active ratio.
- Table 2: LIT-PCBA main results.
- Table 3: DUD-E main results.
- Table 4: ZINC-22 / Enamine scalability.
- Table 5: ablations.
- Appendix: complete per-target tables.

