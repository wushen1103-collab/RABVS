# RABVS

Resource-Aware Budgeted Virtual Screening (RABVS) is a budgeted large-scale
data-selection pipeline for virtual screening. It builds a reusable
chemical-space partition index, estimates target-conditioned partition
utility from limited probes, concentrates candidate-level scoring in selected
regions, and forms a fixed-cardinality shortlist with activity-weighted
reciprocal-rank fusion (RRF).

## Evidence in this repository

- Protocol-matched evaluation on 102 DUD-E targets, 15 LIT-PCBA targets, and
  17 MUV tasks over three random seeds.
- Same-RRF allocation controls that hold final activity/cleanliness/novelty
  fusion fixed while changing only the allocation route.
- A six-point budget-quality sweep and explicit candidate-scoring accounting.
- Deterministically ordered, nested ChEMBL 36 prefixes from 100,000 to
  2,000,000 distinct compounds, evaluated with 10 target scorers and three
  execution seeds.
- A 100,000-compound control comparing landmark SVD-96 with signed hashing.

The ChEMBL scale experiment does not use ChEMBL activity labels. It measures
score avoidance, retrieval overlap, execution time, memory, and index
amortization rather than biological validation.

The 10 target scorers are a prespecified executable-protocol panel fixed before
the ChEMBL evaluation. Every scorer is retained at every nested prefix and
execution seed; no target is selected or excluded using ChEMBL overlap, score
avoidance, runtime, or memory.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-jbd-lock.txt
pip install -e .
python scripts/smoke_test.py
```

The exact JBD experiment settings, seeds, targets, and source checksum are in
`configs/jbd_reproducibility.yaml`.

## Same-RRF allocation control

```bash
python scripts/jbd_experiments/run_same_rrf_matched_protocol.py \
  --dataset dude --seeds 0,1,2 --jobs 60 \
  --out tables/jbd/same_rrf_matched_dude_all102.csv

python scripts/jbd_experiments/run_same_rrf_matched_protocol.py \
  --dataset lit --seeds 0,1,2 --jobs 60 \
  --out tables/jbd/same_rrf_matched_lit_all15.csv
```

Every allocation route in this control uses the same final RRF weights:
activity 1.0, cleanliness 0.18, novelty 0.25, and reciprocal-rank constant 60.

## ChEMBL 36 scale experiment

Download the public ChEMBL 36 chemical-representation archive from its
original provider and verify the compressed source file:

```text
f9c2fa730ac5000da9b0c0d4033c99bfd8fe2b99950f19df1d842c6e73dd7cc4
```

Prepare a deterministic nested cache in one frozen landmark-SVD basis:

```bash
python scripts/jbd_experiments/prepare_chembl36_features.py \
  --input data/raw/chembl36/chembl_36_chemreps.txt.gz \
  --output-dir data/processed/chembl36_2m_landmark_svd_seed20260928 \
  --max-molecules 2000000 --workers 60 --selection-seed 20260928 \
  --projection landmark-svd --landmark-size 100000
```

Run the 10-scorer, three-seed scaling experiment:

```bash
python scripts/jbd_experiments/run_chembl36_scalability.py \
  --feature-dir data/processed/chembl36_2m_landmark_svd_seed20260928 \
  --sizes 100000,250000,500000,1000000,2000000 \
  --targets aa2ar,abl1,bace1,braf,cdk2,egfr,esr1,akt1,kit,pparg \
  --seeds 0,1,2 --out tables/jbd/chembl36_scale_svd_all.csv
```

## Public outputs

Manuscript-facing CSV files are under `tables/jbd/`. Large raw datasets,
processed feature matrices, model environments, and external checkpoints are
excluded. Dataset and checkpoint use remains subject to each provider's
original terms.
