
# SCAX plan (fixed)

The full plan, with the literature review, formal definitions, proofs and experiment design, is the
document "SCAX plan v2" (7 November 2026). This file holds what every notebook must respect.

## Method in one paragraph

A tree ensemble is trained on the training portion. For each patient, a solver produces sound boxes
(abductive explanations and every sound box on their inflation path) for the deployed exported
classifier. Each box B gets a Clopper-Pearson upper bound on R(B) = P(Y != y_B | X in B) from
calibration split 1 at level delta_1 * pi(B), where pi is an Occam prior fixed from the training
portion; the bound holds for all boxes at once, so choosing among boxes is safe. A patient receives
the most succinct box whose bound is at most alpha for the predicted class, otherwise the system
abstains. The frozen release policy is certified on calibration split 2: deployed only if the
Clopper-Pearson bound on its error among accepted patients is at most alpha_sel.

## Fixed choices

- Total failure budget 0.05: delta_1 = delta_2 = 0.025.
- Negative-outcome claims at alpha_0; positive-outcome enrichment claims at alpha_1 = 1 - beta,
  reported separately. Certificates concern the recorded (mostly self-reported) outcome.
- All splits by simple random allocation that ignores the label: training portion 40% (dev-train
  75%, dev-cal 25% for every design choice), calibration split 1 30%, calibration split 2 15%,
  test 15%.
- Freezing rule: model, threshold, box family, prior, number of candidates, selection rule and all
  targets are frozen in notebook 05 with a hash before calibration split 1 is opened. Choosing among
  boxes of the covered family with calibration split 1 is allowed; changing anything else is not.
- Calibration split 2 is used once; a policy search on it needs Bonferroni correction or fresh data.
  Fixed-sequence testing is not used (it is FG-LTT's procedure).
- SCAX certifies and deploys its own exported classifier (exact arithmetic); agreement with the
  native library is reported. Solver timeout or UNKNOWN means not sound. Bounds use beta.isf.
- Validity on the synthetic population is judged by the three-way rule (upper limit at or below
  delta: low enough; lower limit above delta: excess; otherwise inconclusive), with multiplicity
  correction, and always reported with the certification rate.

## Notebook map

| NB | Purpose |
|---|---|
| 00 | Setup: repository from scratch, environment audit, unit tests |
| 01 | Public datasets: download, checks, dataset cards |
| 02 | Rebuild BRFSS 2015 to 2023 from raw CDC files (gate G0: rebuilt 2015 equals the public table) |
| 03 | NHANES and UCI site preparation; outcome map |
| 04 | Planted-rule population (20,000 rows) and exact-risk oracle |
| 05 | Development on the training portion; freeze configuration and hashes |
| 06 | Exported classifier and exact soundness verification (gate G2) |
| 07 | Candidate enumeration with all sound inflation-path boxes |
| 08 | Prior and box certificates on calibration split 1 |
| 09 | Release, policy certification on calibration split 2, main results |
| 10 | Validity study on the synthetic population |
| 11 | Baselines |
| 12 | Ablations |
| 13 | Shift: temporal, geographic, site, measurement; re-certification sample sizes |
| 14 | Subgroups and group-conditional certificates |
| 15 | Runtime |
| 16 | Paper results and claims check |

## Conventions

- Drive root `/content/drive/MyDrive/SCAX_Research/`; datasets in `datasets/` outside the repository;
  `commit_console.ipynb` at the Drive root.
- Every notebook starts with the session bootstrap cell and ends with the standard commit cell;
  `tools/commit_cell.py` is the only commit mechanism.
- Seeds in every results file; figures at 600 dpi with paper names; notebooks tested before delivery.
