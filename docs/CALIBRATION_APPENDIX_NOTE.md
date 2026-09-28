# Calibration Appendix Note

This note is a manuscript-facing interpretation guide for the calibration tables.
It is intentionally text-only; figures are deferred until the plotting pass.

## DUD-E

The DUD-E calibration appendix supports a positive calibration claim for the
adopted final-retrieval branch. Against BOBa, the adopted method lowers ECE,
Brier score, NLL, held-out split-calibration ECE/Brier/NLL, and makes the 90%
conformal residual interval more conservative. The strongest wording is:

> RABVS improves probability calibration and conservative residual coverage on
> DUD-E while also improving the main retrospective discovery metrics.

Risk-coverage AUC should not be described as improved. In the DUD-E appendix it
is neutral-to-slightly worse depending on the comparison, so the correct story is
that calibrated probabilities and residual coverage improve, not that uncertainty
ranking is uniformly better for selective prediction.

## LIT-PCBA

The LIT-PCBA calibration audit is neutral for the reliable budgeted method versus
BOBa. `rabvs_budgeted_reliable` and `boba_ucb` have identical ECE, Brier, NLL,
held-out calibration metrics, conformal coverage, residual q90, risk-coverage
AUC, and false-confidence gap across 45 target-seed pairs. This is expected
because the LIT reliability/diversity improvement comes from the final ranking
constraints rather than a different probability model.

The strongest wording is:

> On LIT-PCBA, calibration is preserved rather than improved; the reliability
> gains come from alert-aware and diversity-aware final ranking.

The conformal coverage remains conservative on LIT-PCBA for both methods
(mean 0.9865 for a nominal 0.90 interval), so it is safe to report coverage as a
sanity check. It should not be framed as a method-specific advantage over BOBa.

## Recommended Claim Boundary

Use calibration improvements as a DUD-E appendix claim. Use LIT-PCBA calibration
as a cross-dataset audit showing no hidden calibration regression. Keep
risk-coverage AUC as a diagnostic, not a headline claim.
