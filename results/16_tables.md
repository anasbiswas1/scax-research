# Paper tables

Written by notebook 16 from the committed results files; every table names its source.

## Gates G0 to G5

| gate | status | evidence |
|---|---|---|
| G0 data | passed | 253,680 rebuilt rows against 253,680; equal row by row: True; marginal mismatches: 0; diabetes outcome equal: True |
| G1 model | passed | CDC Diabetes: 0.8269 against 0.8277; BRFSS Heart disease: 0.8523 against 0.8532; Cardio-70k: 0.8012 against 0.8015 |
| G2 soundness | passed | notebook 06: exported against native 0, primary against second check 0, pre-check wrong 0, Z3 disagreements 0 of 240 (5 unknown), brute-force errors 0 of 2,400 / CDC Diabetes frozen model: exported against native 0; 300 boxes, primary against second check 0; Z3 0 disagreements of 30 / BRFSS Heart disease frozen model: exported against native 0; 300 boxes, primary against second check 0; Z3 0 disagreements of 30 / Cardio-70k frozen model: exported against native 0; 300 boxes, primary against second check 0; Z3 0 disagreements of 30 / CDC Diabetes: 63,781 of 63,781 releasable boxes sound by the second exact check (0 unknown); Z3 sample 58 sound, 2 unknown / BRFSS Heart disease: 62,356 of 62,356 releasable boxes sound by the second exact check (0 unknown); Z3 sample 59 sound, 1 unknown / Cardio-70k: 27,192 of 27,192 releasable boxes sound by the second exact check (0 unknown); Z3 sample 60 sound, 0 unknown / LightGBM and random forest (14 models): exported against native beyond rounding 0; releasable boxes left unverified 0 / shift models (11): exported against native beyond rounding 0; rows within float32 rounding of the threshold, decided by the exact sum: 2 / ablation variants (24): exported against native 0, second-check failures 0 / synthetic population models (13): unverified boxes removed 0, searches stopped by the budget 0 |
| G3 utility (development only) | passed for the other tasks; Cardio-70k: cascade followed, shortfall recorded before calibration split 1 was opened (docs/PLAN.md) | CDC Diabetes: 53.9% (target met: True); BRFSS Heart disease: 63.1% (target met: True); Cardio-70k: 8.4% (target met: False) |
| G4 validity | passed | 936 SCAX configurations (two priors): box certificates {'low enough': 936}; policy {'low enough': 936}; largest violation frequency 0.026 (box) and 0.010 (policy); box-certificate repetitions per configuration 500 to 5000 |
| G5 runtime | passed | production medians 0.04 to 0.56 seconds per cell (7 models) |

## Table 1. Main results at the frozen targets (E1)

Source: 16_table_main.csv

| Task | Model | Claim | Policy | Explained test patients with a certified claim (95% CI) | With the outcome among them (95% CI) | One-sided 95% bound | Mean features | Median calibration records |
|---|---|---|---|---|---|---|---|---|
| BRFSS Heart disease | XGBoost | negative outcome: at most 5.0% with the outcome | certified (upper bound 2.8%, target at most 5.0%) | 65.3% (63.9 to 66.6) | 2.3% (1.8 to 2.8) | 2.8% | 5.95 | 7,394 |
| BRFSS Heart disease | XGBoost | enrichment: at least 30.0% with the outcome | certified (lower bound 23.1%, target at least 20.0%) | 26.0% (24.8 to 27.2) | 27.1% (24.7 to 29.5) | 25.1% | 5.29 | 1,798 |
| BRFSS Heart disease | LightGBM | negative outcome: at most 5.0% with the outcome | certified (upper bound 2.7%, target at most 5.0%) | 64.8% (63.4 to 66.1) | 2.0% (1.5 to 2.5) | 2.5% | 6.67 | 5,463 |
| BRFSS Heart disease | LightGBM | enrichment: at least 30.0% with the outcome | certified (lower bound 23.3%, target at least 20.0%) | 26.3% (25.1 to 27.5) | 27.2% (24.9 to 29.7) | 25.2% | 5.36 | 2,350 |
| BRFSS Heart disease | Random forest | negative outcome: at most 5.0% with the outcome | certified (upper bound 3.1%, target at most 5.0%) | 63.2% (61.9 to 64.5) | 2.5% (1.9 to 3.0) | 3.0% | 6.23 | 21,627 |
| BRFSS Heart disease | Random forest | enrichment: at least 30.0% with the outcome | certified (lower bound 21.0%, target at least 20.0%) | 27.5% (26.2 to 28.7) | 24.2% (22.0 to 26.5) | 22.3% | 3.10 | 3,091 |
| CDC Diabetes | XGBoost | negative outcome: at most 5.0% with the outcome | certified (upper bound 4.5%, target at most 5.0%) | 58.6% (57.3 to 60.0) | 4.0% (3.3 to 4.7) | 4.6% | 5.16 | 6,445 |
| CDC Diabetes | XGBoost | enrichment: at least 30.0% with the outcome | not certified (lower bound 28.8%, target at least 30.0%) | 35.0% (33.7 to 36.3) | 32.2% (30.1 to 34.4) | 30.3% | 6.18 | 4,053 |
| CDC Diabetes | LightGBM | negative outcome: at most 5.0% with the outcome | not certified (upper bound 5.1%, target at most 5.0%) | 60.5% (59.1 to 61.9) | 4.6% (3.8 to 5.3) | 5.2% | 5.34 | 4,203 |
| CDC Diabetes | LightGBM | enrichment: at least 30.0% with the outcome | not certified (lower bound 29.2%, target at least 30.0%) | 33.6% (32.3 to 34.9) | 32.0% (29.8 to 34.2) | 30.1% | 6.20 | 3,169 |
| CDC Diabetes | Random forest | negative outcome: at most 5.0% with the outcome | not certified (upper bound 5.9%, target at most 5.0%) | 64.7% (63.4 to 66.0) | 5.3% (4.6 to 6.1) | 6.0% | 5.23 | 18,666 |
| CDC Diabetes | Random forest | enrichment: at least 30.0% with the outcome | not certified (lower bound 28.1%, target at least 30.0%) | 34.0% (32.7 to 35.3) | 32.1% (29.8 to 34.3) | 30.2% | 4.09 | 5,705 |
| Cardio-70k | XGBoost | negative outcome: at most 20.0% with the outcome | not certified (upper bound 23.1%, target at most 20.0%) | 29.1% (27.9 to 30.3) | 21.0% (18.9 to 23.1) | 22.8% | 3.66 | 1,761 |
| Cardio-70k | XGBoost | enrichment: at least 80.0% with the outcome | certified (lower bound 82.6%, target at least 80.0%) | 28.4% (27.1 to 29.6) | 84.1% (82.1 to 85.9) | 82.4% | 1.51 | 2,943 |
| Cardio-70k | LightGBM | negative outcome: at most 20.0% with the outcome | not certified (upper bound 22.2%, target at most 20.0%) | 30.2% (28.9 to 31.4) | 20.5% (18.5 to 22.6) | 22.3% | 3.62 | 1,893 |
| Cardio-70k | LightGBM | enrichment: at least 80.0% with the outcome | certified (lower bound 82.4%, target at least 80.0%) | 28.5% (27.3 to 29.8) | 84.0% (82.1 to 85.9) | 82.3% | 1.00 | 5,971 |
| Cardio-70k | Random forest | negative outcome: at most 20.0% with the outcome | not certified (upper bound 22.6%, target at most 20.0%) | 31.3% (30.1 to 32.6) | 20.9% (18.9 to 22.9) | 22.6% | 4.13 | 3,016 |
| Cardio-70k | Random forest | enrichment: at least 80.0% with the outcome | certified (lower bound 82.4%, target at least 80.0%) | 28.5% (27.2 to 29.7) | 84.1% (82.1 to 85.9) | 82.4% | 1.00 | 5,963 |
| UCI Heart | XGBoost | negative outcome: at most 5.0% with the outcome | not run | 0.0% (0.0 to 0.0) | n/a | n/a | n/a | n/a |
| UCI Heart | XGBoost | enrichment: at least 30.0% with the outcome | not run | 23.0% (20.6 to 25.5) | 90.3% (86.1 to 94.2) | n/a | 4.04 | n/a |
| UCI Heart | LightGBM | negative outcome: at most 5.0% with the outcome | not run | 0.0% (0.0 to 0.0) | n/a | n/a | n/a | n/a |
| UCI Heart | LightGBM | enrichment: at least 30.0% with the outcome | not run | 18.4% (16.1 to 20.8) | 87.9% (82.5 to 92.7) | n/a | 4.74 | n/a |
| UCI Heart | Random forest | negative outcome: at most 5.0% with the outcome | not run | 0.0% (0.0 to 0.0) | n/a | n/a | n/a | n/a |
| UCI Heart | Random forest | enrichment: at least 30.0% with the outcome | not run | 49.9% (46.9 to 53.1) | 83.1% (79.5 to 86.6) | n/a | 3.20 | n/a |

Policy: Clopper-Pearson bound on the share with the outcome among released patients of calibration split 2 at delta_2/2 per kind (upper bound for negative-outcome claims, lower bound for enrichment claims). Intervals: 95% percentile bootstrap over the explained test patients (10,000 resamples). One-sided 95% bound: Clopper-Pearson on the test patients (upper for negative-outcome claims, lower for enrichment claims). UCI Heart pools the four held-out sites.

## Table 2. Underlying models on the full test split

Source: 09b_model_metrics.csv, 09e_cardio_model_metrics.csv

| Task | Model | AUROC | AUPRC | Brier score | ECE (15 equal-mass bins) |
|---|---|---|---|---|---|
| BRFSS Heart disease | XGBoost | 0.848 | 0.381 | 0.071 | 0.005 |
| BRFSS Heart disease | LightGBM | 0.848 | 0.380 | 0.071 | 0.005 |
| BRFSS Heart disease | Random forest | 0.829 | 0.352 | 0.075 | 0.029 |
| CDC Diabetes | XGBoost | 0.830 | 0.438 | 0.097 | 0.005 |
| CDC Diabetes | LightGBM | 0.828 | 0.433 | 0.097 | 0.006 |
| CDC Diabetes | Random forest | 0.812 | 0.409 | 0.102 | 0.038 |
| Cardio-70k | XGBoost | 0.803 | 0.788 | 0.179 | 0.015 |
| Cardio-70k | LightGBM | 0.802 | 0.787 | 0.179 | 0.013 |
| Cardio-70k | Random forest | 0.795 | 0.776 | 0.187 | 0.057 |

## Table 3. UCI Heart by held-out site (frozen heart targets)

Source: 09e_uci_main.csv

| site | model | kind | alpha_claim | explained | accepted | coverage | coverage_lo | coverage_hi | errors | error_rate | error_upper_95 | refuted_share | auroc | prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cleveland | XGBoost | negative outcome | 5.0% | 303 | 0 | 0.0% | 0.0% | 1.0% | 0 | n/a | 100.0% | 0.0% | 0.796 | 45.9% |
| Cleveland | XGBoost | enrichment | 70.0% | 303 | 37 | 12.2% | 9.2% | 15.7% | 2 | 5.4% | 16.1% | 0.0% | 0.796 | 45.9% |
| Hungary | XGBoost | negative outcome | 5.0% | 292 | 0 | 0.0% | 0.0% | 1.0% | 0 | n/a | 100.0% | 0.0% | 0.868 | 36.0% |
| Hungary | XGBoost | enrichment | 70.0% | 292 | 42 | 14.4% | 11.1% | 18.2% | 4 | 9.5% | 20.5% | 0.0% | 0.868 | 36.0% |
| Switzerland | XGBoost | negative outcome | 5.0% | 116 | 0 | 0.0% | 0.0% | 2.5% | 0 | n/a | 100.0% | 0.0% | 0.767 | 93.1% |
| Switzerland | XGBoost | enrichment | 70.0% | 116 | 23 | 19.8% | 13.9% | 26.9% | 1 | 4.3% | 19.0% | 0.0% | 0.767 | 93.1% |
| VA Long Beach | XGBoost | negative outcome | 5.0% | 140 | 0 | 0.0% | 0.0% | 2.1% | 0 | n/a | 100.0% | 0.0% | 0.676 | 78.6% |
| VA Long Beach | XGBoost | enrichment | 70.0% | 140 | 94 | 67.1% | 60.0% | 73.7% | 12 | 12.8% | 19.9% | 0.0% | 0.676 | 78.6% |
| Cleveland | LightGBM | negative outcome | 5.0% | 303 | 0 | 0.0% | n/a | n/a | 0 | n/a | 100.0% | 0.0% | 0.773 | 45.9% |
| Cleveland | LightGBM | enrichment | 70.0% | 303 | 37 | 12.2% | n/a | n/a | 4 | 10.8% | 23.1% | 0.0% | 0.773 | 45.9% |
| Cleveland | random forest | negative outcome | 5.0% | 303 | 0 | 0.0% | n/a | n/a | 0 | n/a | 100.0% | 0.0% | 0.874 | 45.9% |
| Cleveland | random forest | enrichment | 70.0% | 303 | 125 | 41.3% | n/a | n/a | 25 | 20.0% | 26.8% | 0.0% | 0.874 | 45.9% |
| Hungary | LightGBM | negative outcome | 5.0% | 292 | 0 | 0.0% | n/a | n/a | 0 | n/a | 100.0% | 0.0% | 0.863 | 36.0% |
| Hungary | LightGBM | enrichment | 70.0% | 292 | 24 | 8.2% | n/a | n/a | 3 | 12.5% | 29.2% | 0.0% | 0.863 | 36.0% |
| Hungary | random forest | negative outcome | 5.0% | 292 | 0 | 0.0% | n/a | n/a | 0 | n/a | 100.0% | 0.0% | 0.870 | 36.0% |
| Hungary | random forest | enrichment | 70.0% | 292 | 105 | 36.0% | n/a | n/a | 28 | 26.7% | 34.7% | 0.0% | 0.870 | 36.0% |
| Switzerland | LightGBM | negative outcome | 5.0% | 116 | 0 | 0.0% | n/a | n/a | 0 | n/a | 100.0% | 0.0% | 0.691 | 93.1% |
| Switzerland | LightGBM | enrichment | 70.0% | 116 | 16 | 13.8% | n/a | n/a | 1 | 6.2% | 26.4% | 0.0% | 0.691 | 93.1% |
| Switzerland | random forest | negative outcome | 5.0% | 116 | 0 | 0.0% | n/a | n/a | 0 | n/a | 100.0% | 0.0% | 0.795 | 93.1% |
| Switzerland | random forest | enrichment | 70.0% | 116 | 91 | 78.4% | n/a | n/a | 2 | 2.2% | 6.8% | 0.0% | 0.795 | 93.1% |
| VA Long Beach | LightGBM | negative outcome | 5.0% | 140 | 0 | 0.0% | n/a | n/a | 0 | n/a | 100.0% | 0.0% | 0.612 | 78.6% |
| VA Long Beach | LightGBM | enrichment | 70.0% | 140 | 80 | 57.1% | n/a | n/a | 11 | 13.8% | 21.7% | 0.0% | 0.612 | 78.6% |
| VA Long Beach | random forest | negative outcome | 5.0% | 140 | 0 | 0.0% | n/a | n/a | 0 | n/a | 100.0% | 0.0% | 0.706 | 78.6% |
| VA Long Beach | random forest | enrichment | 70.0% | 140 | 104 | 74.3% | n/a | n/a | 17 | 16.3% | 23.5% | 0.0% | 0.706 | 78.6% |

## Table 4. Validity on the planted-rule population (E2)

Source: 10_validity.csv

| prior | delta1 | configurations | repetitions | low_enough | inconclusive | excess | max_violation | mean_violation | policy_excess | max_policy_violation | certification_rate | policy_certified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Per-box bound without prior | 1.0% | 156 | 500 | 36 | 4 | 116 | 100.0% | 19.74% | 0 | 0.8% | 94.3% | 95.2% |
| Per-box bound without prior | 2.5% | 156 | 500 | 36 | 0 | 120 | 100.0% | 29.78% | 0 | 0.4% | 95.5% | 94.5% |
| Per-box bound without prior | 5.0% | 156 | 500 | 36 | 0 | 120 | 100.0% | 40.22% | 0 | 0.2% | 96.3% | 93.9% |
| SCAX, Occam prior | 1.0% | 156 | 500 | 156 | 0 | 0 | 0.0% | 0.00% | 0 | 0.0% | 76.3% | 90.3% |
| SCAX, Occam prior | 2.5% | 156 | 500 | 156 | 0 | 0 | 0.0% | 0.00% | 0 | 0.0% | 77.7% | 91.4% |
| SCAX, Occam prior | 5.0% | 156 | 500 | 156 | 0 | 0 | 0.0% | 0.00% | 0 | 0.0% | 78.8% | 92.2% |
| SCAX, mixture prior (frozen) | 1.0% | 156 | 5000 | 156 | 0 | 0 | 0.5% | 0.12% | 0 | 1.0% | 88.2% | 96.3% |
| SCAX, mixture prior (frozen) | 2.5% | 156 | 5000 | 156 | 0 | 0 | 1.2% | 0.38% | 0 | 1.0% | 89.6% | 96.5% |
| SCAX, mixture prior (frozen) | 5.0% | 156 | 500 | 156 | 0 | 0 | 2.6% | 0.88% | 0 | 1.0% | 90.5% | 96.6% |

Per prior and delta_1: configurations (13 label settings x 3 calibration sizes x 4 values of M), verdicts of the three-way rule for box certificates (one-sided 95% Clopper-Pearson limits, excess test corrected for 936 configurations), and the policy check at delta_2 = 0.025.

## Table 5a. Baselines, CDC Diabetes (20 calibration re-draws)

Source: 11_baselines_summary.csv

| kind | method | coverage | coverage_lo | coverage_hi | error_rate | above_target_share | refuted_share | policy_certified | sound_share | mean_size |
|---|---|---|---|---|---|---|---|---|---|---|
| negative outcome | SCAX | 58.6% | 58.5% | 58.8% | 3.98% | 0.0% | 0.0% | 100.0% | n/a | 5.17 |
| enrichment | SCAX | 35.2% | 35.0% | 35.3% | 67.88% | 0.0% | 0.0% | 0.0% | n/a | 6.20 |
| negative outcome | Per-box bound without prior | 60.8% | 60.8% | 60.9% | 4.21% | 0.0% | 0.0% | 100.0% | n/a | 5.26 |
| enrichment | Per-box bound without prior | 37.7% | 37.6% | 37.8% | 68.77% | 0.1% | 0.0% | 0.0% | n/a | 6.43 |
| negative outcome | Empirical error rule | 61.9% | 61.9% | 61.9% | 4.39% | 0.6% | 0.0% | 100.0% | n/a | 5.34 |
| enrichment | Empirical error rule | 38.1% | 38.0% | 38.1% | 68.83% | 0.5% | 0.0% | 0.0% | n/a | 6.46 |
| negative outcome | Uncertified AXp, all patients | 61.9% | 61.9% | 61.9% | 4.39% | 14.0% | 0.0% | n/a | n/a | 5.73 |
| enrichment | Uncertified AXp, all patients | 38.1% | 38.1% | 38.1% | 68.87% | 47.1% | 0.0% | n/a | n/a | 6.99 |
| negative outcome | Uncertified AXp, matched coverage | 58.6% | 58.5% | 58.8% | 3.98% | 13.4% | 0.0% | n/a | n/a | 5.47 |
| enrichment | Uncertified AXp, matched coverage | 35.2% | 35.0% | 35.3% | 68.01% | 45.0% | 0.0% | n/a | n/a | 6.73 |
| negative outcome | Uncertified inflated box, all patients | 61.9% | 61.9% | 61.9% | 4.39% | 1.2% | 0.0% | n/a | n/a | 5.73 |
| enrichment | Uncertified inflated box, all patients | 38.1% | 38.1% | 38.1% | 68.87% | 1.4% | 0.0% | n/a | n/a | 6.99 |
| negative outcome | Uncertified inflated box, matched coverage | 58.6% | 58.5% | 58.8% | 3.98% | 0.9% | 0.0% | n/a | n/a | 5.47 |
| enrichment | Uncertified inflated box, matched coverage | 35.2% | 35.0% | 35.3% | 68.01% | 0.8% | 0.0% | n/a | n/a | 6.73 |
| negative outcome | Rule dictionary | 27.9% | 27.9% | 27.9% | 2.80% | 0.0% | 0.0% | 100.0% | 0.0% | 2.00 |
| enrichment | Rule dictionary | 35.4% | 35.1% | 36.4% | 67.78% | 0.0% | 0.0% | 0.0% | 0.0% | 1.77 |
| negative outcome | Rule dictionary, sound rules only | 0.0% | 0.0% | 0.0% | n/a | 0.0% | 0.0% | 0.0% | n/a | n/a |
| enrichment | Rule dictionary, sound rules only | 0.0% | 0.0% | 0.0% | n/a | 0.0% | 0.0% | 0.0% | n/a | n/a |
| negative outcome | SGR | 61.9% | 61.9% | 61.9% | 4.40% | n/a | n/a | n/a | n/a | n/a |
| enrichment | SGR | 34.8% | 34.3% | 35.3% | 67.88% | n/a | n/a | n/a | n/a | n/a |
| negative outcome | Learn-then-Test | 61.9% | 61.9% | 61.9% | 4.39% | n/a | n/a | n/a | n/a | n/a |
| enrichment | Learn-then-Test | 35.1% | 34.6% | 35.6% | 67.97% | n/a | n/a | n/a | n/a | n/a |
| negative outcome | Split conformal, singletons | 76.8% | 76.4% | 77.1% | 7.68% | n/a | n/a | n/a | n/a | n/a |
| enrichment | Split conformal, singletons | 0.0% | 0.0% | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| negative outcome | Mondrian conformal, singletons | 41.2% | 40.6% | 41.6% | 2.38% | n/a | n/a | n/a | n/a | n/a |
| enrichment | Mondrian conformal, singletons | 8.3% | 8.2% | 8.3% | 44.28% | n/a | n/a | n/a | n/a | n/a |

coverage_lo and coverage_hi: 2.5% and 97.5% quantiles over the re-draws; above_target_share: share of released patients whose box has a test error above its target; refuted_share: the same with the Bonferroni-corrected lower limit above the target; policy_certified: share of re-draws.

## Table 5b. Baselines, BRFSS Heart disease (20 calibration re-draws)

Source: 11_baselines_summary.csv

| kind | method | coverage | coverage_lo | coverage_hi | error_rate | above_target_share | refuted_share | policy_certified | sound_share | mean_size |
|---|---|---|---|---|---|---|---|---|---|---|
| negative outcome | SCAX | 65.2% | 65.2% | 65.3% | 2.30% | 0.0% | 0.0% | 100.0% | n/a | 5.95 |
| enrichment | SCAX | 25.9% | 24.9% | 26.5% | 72.86% | 0.0% | 0.0% | 100.0% | n/a | 5.27 |
| negative outcome | Per-box bound without prior | 66.5% | 66.5% | 66.5% | 2.38% | 0.0% | 0.0% | 100.0% | n/a | 6.01 |
| enrichment | Per-box bound without prior | 32.0% | 31.7% | 32.3% | 75.66% | 0.6% | 0.0% | 100.0% | n/a | 5.74 |
| negative outcome | Empirical error rule | 66.7% | 66.7% | 66.7% | 2.37% | 0.0% | 0.0% | 100.0% | n/a | 6.02 |
| enrichment | Empirical error rule | 33.1% | 33.0% | 33.1% | 76.24% | 1.8% | 0.0% | 100.0% | n/a | 5.81 |
| negative outcome | Uncertified AXp, all patients | 66.7% | 66.7% | 66.7% | 2.37% | 5.1% | 0.0% | n/a | n/a | 6.37 |
| enrichment | Uncertified AXp, all patients | 33.3% | 33.3% | 33.3% | 76.31% | 51.1% | 1.7% | n/a | n/a | 6.34 |
| negative outcome | Uncertified AXp, matched coverage | 65.2% | 65.2% | 65.3% | 2.21% | 4.7% | 0.0% | n/a | n/a | 6.27 |
| enrichment | Uncertified AXp, matched coverage | 25.9% | 24.9% | 26.5% | 73.10% | 42.5% | 1.3% | n/a | n/a | 5.74 |
| negative outcome | Uncertified inflated box, all patients | 66.7% | 66.7% | 66.7% | 2.37% | 0.1% | 0.0% | n/a | n/a | 6.37 |
| enrichment | Uncertified inflated box, all patients | 33.3% | 33.3% | 33.3% | 76.31% | 8.5% | 0.1% | n/a | n/a | 6.34 |
| negative outcome | Uncertified inflated box, matched coverage | 65.2% | 65.2% | 65.3% | 2.21% | 0.1% | 0.0% | n/a | n/a | 6.27 |
| enrichment | Uncertified inflated box, matched coverage | 25.9% | 24.9% | 26.5% | 73.10% | 5.7% | 0.0% | n/a | n/a | 5.74 |
| negative outcome | Rule dictionary | 66.7% | 66.7% | 66.7% | 2.37% | 0.0% | 0.0% | 100.0% | 0.0% | 1.03 |
| enrichment | Rule dictionary | 14.0% | 12.4% | 15.5% | 67.02% | 0.0% | 0.0% | 100.0% | 0.0% | 1.99 |
| negative outcome | Rule dictionary, sound rules only | 0.0% | 0.0% | 0.0% | n/a | 0.0% | 0.0% | 0.0% | n/a | n/a |
| enrichment | Rule dictionary, sound rules only | 0.0% | 0.0% | 0.0% | n/a | 0.0% | 0.0% | 0.0% | n/a | n/a |
| negative outcome | SGR | 66.7% | 66.7% | 66.7% | 2.37% | n/a | n/a | n/a | n/a | n/a |
| enrichment | SGR | 33.3% | 33.3% | 33.3% | 76.31% | n/a | n/a | n/a | n/a | n/a |
| negative outcome | Learn-then-Test | 66.7% | 66.7% | 66.7% | 2.37% | n/a | n/a | n/a | n/a | n/a |
| enrichment | Learn-then-Test | 33.3% | 33.3% | 33.3% | 76.31% | n/a | n/a | n/a | n/a | n/a |
| negative outcome | Split conformal, singletons | 88.4% | 88.2% | 88.7% | 5.57% | n/a | n/a | n/a | n/a | n/a |
| enrichment | Split conformal, singletons | 0.1% | 0.1% | 0.1% | 24.42% | n/a | n/a | n/a | n/a | n/a |
| negative outcome | Mondrian conformal, singletons | 45.5% | 44.7% | 46.0% | 0.92% | n/a | n/a | n/a | n/a | n/a |
| enrichment | Mondrian conformal, singletons | 7.7% | 7.6% | 7.8% | 57.14% | n/a | n/a | n/a | n/a | n/a |

coverage_lo and coverage_hi: 2.5% and 97.5% quantiles over the re-draws; above_target_share: share of released patients whose box has a test error above its target; refuted_share: the same with the Bonferroni-corrected lower limit above the target; policy_certified: share of re-draws.

## Table 6. Baselines on the planted rule P4, where true risk is known (200 repetitions)

Source: 11_synthetic_summary.csv

| alpha_0 | alpha_1 | method | coverage | coverage_lo | coverage_hi | claim_precision | precision_lo | precision_hi | any_false_claim | true_selective_risk | mean_size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.0% | 70.0% | SCAX | 71.4% | 51.9% | 80.3% | 100.0% | 100.0% | 100.0% | 0.0% | 4.67% | 1.04 |
| 5.0% | 70.0% | Per-box bound without prior | 79.4% | 75.1% | 84.3% | 100.0% | 99.8% | 100.0% | 44.5% | 5.61% | 1.16 |
| 5.0% | 70.0% | Empirical error rule | 87.9% | 83.3% | 90.6% | 94.5% | 87.5% | 97.5% | 100.0% | 7.26% | 1.32 |
| 5.0% | 70.0% | Rule dictionary | 67.8% | 55.1% | 72.4% | 100.0% | 100.0% | 100.0% | 0.0% | 4.32% | 1.13 |
| 5.0% | 70.0% | Rule dictionary, sound rules only | 67.7% | 55.0% | 72.1% | 100.0% | 100.0% | 100.0% | 0.0% | 4.25% | 1.13 |
| 5.0% | 70.0% | Confidence-aware AXp | 47.1% | 47.1% | 47.1% | 98.9% | 98.9% | 98.9% | 100.0% | 2.57% | 3.18 |
| 5.0% | 70.0% | Uncertified AXp | 100.0% | 100.0% | 100.0% | 41.4% | 41.4% | 41.4% | 100.0% | 9.99% | 1.41 |
| 5.0% | 70.0% | Uncertified inflated box | 100.0% | 100.0% | 100.0% | 63.3% | 63.3% | 63.3% | 100.0% | 9.99% | 1.41 |
| 2.0% | 50.0% | SCAX | 0.4% | 0.0% | 5.4% | 100.0% | 100.0% | 100.0% | 0.0% | 1.36% | 1.00 |
| 2.0% | 50.0% | Per-box bound without prior | 10.8% | 0.0% | 21.8% | 98.8% | 98.2% | 100.0% | 8.0% | 2.50% | 1.16 |
| 2.0% | 50.0% | Empirical error rule | 29.3% | 21.1% | 36.2% | 68.3% | 45.4% | 81.4% | 100.0% | 5.47% | 1.84 |
| 2.0% | 50.0% | Rule dictionary | 0.0% | 0.0% | 0.0% | n/a | n/a | n/a | 0.0% | n/a | n/a |
| 2.0% | 50.0% | Rule dictionary, sound rules only | 0.0% | 0.0% | 0.0% | n/a | n/a | n/a | 0.0% | n/a | n/a |
| 2.0% | 50.0% | Confidence-aware AXp | 23.7% | 23.7% | 23.7% | 96.7% | 96.7% | 96.7% | 100.0% | 1.81% | 3.92 |
| 2.0% | 50.0% | Uncertified AXp | 100.0% | 100.0% | 100.0% | 15.5% | 15.5% | 15.5% | 100.0% | 9.99% | 1.41 |
| 2.0% | 50.0% | Uncertified inflated box | 100.0% | 100.0% | 100.0% | 0.1% | 0.1% | 0.1% | 100.0% | 9.99% | 1.41 |

## Table 7. Ablations on the frozen models, final splits (E3)

Source: 12_ablations_summary.csv

| task | factor | level | kind | frozen | explained_share | explained_lo | explained_hi | test_error | policy_certified | U_sel |
|---|---|---|---|---|---|---|---|---|---|---|
| cdc_diabetes | prior | Occam prior only | negative outcome | no | 58.3% | 58.3% | 58.3% | 3.95% | 100.0% | 4.49% |
| cdc_diabetes | prior | Occam prior only | enrichment | no | 34.6% | 34.6% | 34.6% | 67.53% | 0.0% | 70.97% |
| cdc_diabetes | prior | mixture, lambda = 0.75 | negative outcome | no | 58.7% | 58.7% | 58.7% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | prior | mixture, lambda = 0.75 | enrichment | no | 35.0% | 35.0% | 35.0% | 67.81% | 0.0% | 71.18% |
| cdc_diabetes | prior | mixture, lambda = 0.5 | negative outcome | yes | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | prior | mixture, lambda = 0.5 | enrichment | yes | 35.0% | 35.0% | 35.0% | 67.81% | 0.0% | 71.21% |
| cdc_diabetes | prior | mixture, lambda = 0.25 | negative outcome | no | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | prior | mixture, lambda = 0.25 | enrichment | no | 35.0% | 35.0% | 35.0% | 67.79% | 0.0% | 71.11% |
| cdc_diabetes | prior | training support only | negative outcome | no | 53.5% | 53.5% | 53.5% | 3.59% | 100.0% | 3.63% |
| cdc_diabetes | prior | training support only | enrichment | no | 32.8% | 32.8% | 32.8% | 67.28% | 0.0% | 70.43% |
| cdc_diabetes | explanation construction | AXps only | negative outcome | no | 27.4% | 27.4% | 27.4% | 1.39% | 100.0% | 2.14% |
| cdc_diabetes | explanation construction | AXps only | enrichment | no | 5.0% | 5.0% | 5.0% | 50.80% | 100.0% | 56.20% |
| cdc_diabetes | explanation construction | widest inflated box of each order only | negative outcome | no | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | explanation construction | widest inflated box of each order only | enrichment | no | 35.0% | 35.0% | 35.0% | 67.79% | 0.0% | 71.21% |
| cdc_diabetes | explanation construction | every sound box on the inflation path | negative outcome | yes | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | explanation construction | every sound box on the inflation path | enrichment | yes | 35.0% | 35.0% | 35.0% | 67.81% | 0.0% | 71.21% |
| cdc_diabetes | selection rule | most succinct | negative outcome | yes | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | selection rule | most succinct | enrichment | yes | 35.0% | 35.0% | 35.0% | 67.81% | 0.0% | 71.21% |
| cdc_diabetes | selection rule | largest support | negative outcome | no | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | selection rule | largest support | enrichment | no | 35.0% | 35.0% | 35.0% | 67.81% | 0.0% | 71.21% |
| cdc_diabetes | selection rule | tightest bound | negative outcome | no | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | selection rule | tightest bound | enrichment | no | 35.0% | 35.0% | 35.0% | 67.81% | 0.0% | 71.21% |
| cdc_diabetes | candidates M | M = 1 | negative outcome | no | 50.5% | 50.5% | 50.5% | 3.49% | 100.0% | 4.38% |
| cdc_diabetes | candidates M | M = 1 | enrichment | no | 29.2% | 29.2% | 29.2% | 67.19% | 0.0% | 71.26% |
| cdc_diabetes | candidates M | M = 3 | negative outcome | no | 55.7% | 55.7% | 55.7% | 3.73% | 100.0% | 4.37% |
| cdc_diabetes | candidates M | M = 3 | enrichment | no | 34.3% | 34.3% | 34.3% | 67.56% | 0.0% | 71.00% |
| cdc_diabetes | candidates M | M = 5 | negative outcome | no | 57.3% | 57.3% | 57.3% | 3.88% | 100.0% | 4.35% |
| cdc_diabetes | candidates M | M = 5 | enrichment | no | 34.5% | 34.5% | 34.5% | 67.67% | 0.0% | 70.92% |
| cdc_diabetes | candidates M | M = 10 | negative outcome | yes | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | candidates M | M = 10 | enrichment | yes | 35.0% | 35.0% | 35.0% | 67.81% | 0.0% | 71.21% |
| cdc_diabetes | calibration size | 5,000 records | negative outcome | no | 34.3% | 30.3% | 38.7% | 3.12% | 100.0% | 3.50% |
| cdc_diabetes | calibration size | 5,000 records | enrichment | no | 20.9% | 12.3% | 26.8% | 64.69% | 70.0% | 69.58% |
| cdc_diabetes | calibration size | 10,000 records | negative outcome | no | 49.3% | 48.4% | 51.0% | 3.46% | 100.0% | 3.58% |
| cdc_diabetes | calibration size | 10,000 records | enrichment | no | 28.7% | 26.0% | 30.1% | 66.30% | 90.0% | 69.49% |
| cdc_diabetes | calibration size | 20,000 records | negative outcome | no | 54.4% | 54.2% | 54.6% | 3.72% | 100.0% | 4.00% |
| cdc_diabetes | calibration size | 20,000 records | enrichment | no | 31.9% | 31.3% | 32.7% | 67.09% | 50.0% | 69.99% |
| cdc_diabetes | calibration size | 40,000 records | negative outcome | no | 57.2% | 56.9% | 57.5% | 3.91% | 100.0% | 4.20% |
| cdc_diabetes | calibration size | 40,000 records | enrichment | no | 33.9% | 33.5% | 34.1% | 67.49% | 0.0% | 70.62% |
| cdc_diabetes | calibration size | 76,104 records | negative outcome | yes | 58.6% | 58.6% | 58.6% | 3.99% | 100.0% | 4.47% |
| cdc_diabetes | calibration size | 76,104 records | enrichment | yes | 35.0% | 35.0% | 35.0% | 67.81% | 0.0% | 71.21% |
| brfss2015_heart | prior | Occam prior only | negative outcome | no | 64.9% | 64.9% | 64.9% | 2.25% | 100.0% | 2.74% |
| brfss2015_heart | prior | Occam prior only | enrichment | no | 22.8% | 22.8% | 22.8% | 71.84% | 100.0% | 75.48% |
| brfss2015_heart | prior | mixture, lambda = 0.75 | negative outcome | no | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | prior | mixture, lambda = 0.75 | enrichment | no | 25.8% | 25.8% | 25.8% | 72.71% | 100.0% | 76.70% |
| brfss2015_heart | prior | mixture, lambda = 0.5 | negative outcome | yes | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | prior | mixture, lambda = 0.5 | enrichment | yes | 26.0% | 26.0% | 26.0% | 72.92% | 100.0% | 76.86% |
| brfss2015_heart | prior | mixture, lambda = 0.25 | negative outcome | no | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.78% |
| brfss2015_heart | prior | mixture, lambda = 0.25 | enrichment | no | 26.0% | 26.0% | 26.0% | 72.96% | 100.0% | 76.86% |
| brfss2015_heart | prior | training support only | negative outcome | no | 63.5% | 63.5% | 63.5% | 2.05% | 100.0% | 2.48% |
| brfss2015_heart | prior | training support only | enrichment | no | 25.3% | 25.3% | 25.3% | 72.93% | 100.0% | 76.37% |
| brfss2015_heart | explanation construction | AXps only | negative outcome | no | 46.0% | 46.0% | 46.0% | 1.26% | 100.0% | 1.63% |
| brfss2015_heart | explanation construction | AXps only | enrichment | no | 6.8% | 6.8% | 6.8% | 58.70% | 100.0% | 69.12% |
| brfss2015_heart | explanation construction | widest inflated box of each order only | negative outcome | no | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | explanation construction | widest inflated box of each order only | enrichment | no | 26.0% | 26.0% | 26.0% | 72.92% | 100.0% | 76.86% |
| brfss2015_heart | explanation construction | every sound box on the inflation path | negative outcome | yes | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | explanation construction | every sound box on the inflation path | enrichment | yes | 26.0% | 26.0% | 26.0% | 72.92% | 100.0% | 76.86% |
| brfss2015_heart | selection rule | most succinct | negative outcome | yes | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | selection rule | most succinct | enrichment | yes | 26.0% | 26.0% | 26.0% | 72.92% | 100.0% | 76.86% |
| brfss2015_heart | selection rule | largest support | negative outcome | no | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | selection rule | largest support | enrichment | no | 26.0% | 26.0% | 26.0% | 72.92% | 100.0% | 76.86% |
| brfss2015_heart | selection rule | tightest bound | negative outcome | no | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | selection rule | tightest bound | enrichment | no | 26.0% | 26.0% | 26.0% | 72.92% | 100.0% | 76.86% |
| brfss2015_heart | candidates M | M = 1 | negative outcome | no | 62.3% | 62.3% | 62.3% | 2.15% | 100.0% | 2.56% |
| brfss2015_heart | candidates M | M = 1 | enrichment | no | 16.7% | 16.7% | 16.7% | 69.89% | 100.0% | 74.86% |
| brfss2015_heart | candidates M | M = 3 | negative outcome | no | 64.4% | 64.4% | 64.4% | 2.27% | 100.0% | 2.68% |
| brfss2015_heart | candidates M | M = 3 | enrichment | no | 23.0% | 23.0% | 23.0% | 72.24% | 100.0% | 76.37% |
| brfss2015_heart | candidates M | M = 5 | negative outcome | no | 65.0% | 65.0% | 65.0% | 2.28% | 100.0% | 2.72% |
| brfss2015_heart | candidates M | M = 5 | enrichment | no | 24.6% | 24.6% | 24.6% | 72.14% | 100.0% | 76.53% |
| brfss2015_heart | candidates M | M = 10 | negative outcome | yes | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | candidates M | M = 10 | enrichment | yes | 26.0% | 26.0% | 26.0% | 72.92% | 100.0% | 76.86% |
| brfss2015_heart | calibration size | 5,000 records | negative outcome | no | 56.3% | 52.2% | 58.3% | 1.76% | 100.0% | 2.26% |
| brfss2015_heart | calibration size | 5,000 records | enrichment | no | 1.1% | 0.0% | 2.9% | 61.14% | 40.0% | 85.46% |
| brfss2015_heart | calibration size | 10,000 records | negative outcome | no | 62.4% | 61.7% | 63.1% | 1.98% | 100.0% | 2.49% |
| brfss2015_heart | calibration size | 10,000 records | enrichment | no | 4.0% | 0.2% | 6.6% | 60.08% | 90.0% | 72.07% |
| brfss2015_heart | calibration size | 20,000 records | negative outcome | no | 64.1% | 64.0% | 64.2% | 2.15% | 100.0% | 2.61% |
| brfss2015_heart | calibration size | 20,000 records | enrichment | no | 12.9% | 9.0% | 15.5% | 68.53% | 100.0% | 72.54% |
| brfss2015_heart | calibration size | 40,000 records | negative outcome | no | 64.7% | 64.6% | 64.8% | 2.24% | 100.0% | 2.71% |
| brfss2015_heart | calibration size | 40,000 records | enrichment | no | 22.2% | 19.8% | 24.0% | 71.90% | 100.0% | 75.33% |
| brfss2015_heart | calibration size | 76,104 records | negative outcome | yes | 65.3% | 65.3% | 65.3% | 2.30% | 100.0% | 2.76% |
| brfss2015_heart | calibration size | 76,104 records | enrichment | yes | 26.0% | 26.0% | 26.0% | 72.92% | 100.0% | 76.86% |

## Table 8. Model factors, 500 explained test patients each (E3)

Source: 12b_model_ablations.csv, 12b_model_variants.csv

| task | factor | level | kind | coverage | explained_lo | explained_hi | error_rate | mean_size | refuted_share | above_target_share |
|---|---|---|---|---|---|---|---|---|---|---|
| cdc_diabetes | frozen model | frozen (XGBoost, depth 4, 100 trees, max_bin 32) | negative outcome | 58.6% | 54.8% | 62.3% | 5.46% | 5.20 | 0.0% | 0.0% |
| cdc_diabetes | frozen model | frozen (XGBoost, depth 4, 100 trees, max_bin 32) | enrichment | 33.0% | 29.5% | 36.6% | 63.64% | 6.13 | 0.0% | 0.0% |
| cdc_diabetes | model family | LightGBM | negative outcome | 60.2% | 56.5% | 63.8% | 5.65% | 5.39 | 0.0% | 0.0% |
| cdc_diabetes | model family | LightGBM | enrichment | 32.0% | 28.6% | 35.6% | 64.38% | 6.24 | 0.0% | 0.0% |
| cdc_diabetes | model family | random forest | negative outcome | 63.4% | 59.7% | 67.0% | 6.31% | 5.25 | 0.0% | 0.0% |
| cdc_diabetes | model family | random forest | enrichment | 35.4% | 31.9% | 39.1% | 66.10% | 3.81 | 0.0% | 0.0% |
| cdc_diabetes | tree depth | depth 3 | negative outcome | 65.2% | 61.5% | 68.7% | 6.13% | 4.84 | 0.0% | 0.0% |
| cdc_diabetes | tree depth | depth 3 | enrichment | 29.6% | 26.2% | 33.1% | 62.84% | 6.47 | 0.0% | 0.0% |
| cdc_diabetes | tree depth | depth 5 | negative outcome | 61.0% | 57.3% | 64.6% | 5.90% | 4.87 | 0.0% | 0.0% |
| cdc_diabetes | tree depth | depth 5 | enrichment | 30.8% | 27.4% | 34.4% | 62.99% | 6.69 | 0.0% | 0.0% |
| cdc_diabetes | tree depth | depth 6 | negative outcome | 58.0% | 54.2% | 61.7% | 5.17% | 4.76 | 0.0% | 0.0% |
| cdc_diabetes | tree depth | depth 6 | enrichment | 32.4% | 28.9% | 36.0% | 64.81% | 6.57 | 0.0% | 0.0% |
| cdc_diabetes | number of trees | 25 trees | negative outcome | 64.6% | 60.9% | 68.1% | 5.88% | 4.47 | 0.0% | 0.0% |
| cdc_diabetes | number of trees | 25 trees | enrichment | 31.0% | 27.6% | 34.6% | 63.23% | 6.05 | 0.0% | 0.0% |
| cdc_diabetes | number of trees | 50 trees | negative outcome | 58.6% | 54.8% | 62.3% | 5.46% | 5.20 | 0.0% | 0.0% |
| cdc_diabetes | number of trees | 50 trees | enrichment | 33.0% | 29.5% | 36.6% | 63.64% | 6.13 | 0.0% | 0.0% |
| cdc_diabetes | threshold bins | max_bin 16 | negative outcome | 59.8% | 56.1% | 63.5% | 5.35% | 4.96 | 0.0% | 0.0% |
| cdc_diabetes | threshold bins | max_bin 16 | enrichment | 32.8% | 29.3% | 36.4% | 65.24% | 6.09 | 0.0% | 0.0% |
| cdc_diabetes | threshold bins | max_bin 64 | negative outcome | 61.6% | 57.9% | 65.2% | 6.17% | 5.40 | 0.0% | 0.0% |
| cdc_diabetes | threshold bins | max_bin 64 | enrichment | 31.2% | 27.8% | 34.8% | 63.46% | 6.33 | 0.0% | 0.0% |
| cdc_diabetes | threshold bins | max_bin 256 | negative outcome | 61.6% | 57.9% | 65.2% | 6.17% | 5.40 | 0.0% | 0.0% |
| cdc_diabetes | threshold bins | max_bin 256 | enrichment | 31.2% | 27.8% | 34.8% | 63.46% | 6.33 | 0.0% | 0.0% |
| cdc_diabetes | monotone constraints | clinically signed | negative outcome | 63.4% | 59.7% | 67.0% | 6.62% | 4.66 | 0.0% | 0.0% |
| cdc_diabetes | monotone constraints | clinically signed | enrichment | 33.8% | 30.3% | 37.4% | 66.27% | 6.19 | 0.0% | 0.0% |
| brfss2015_heart | frozen model | frozen (XGBoost, depth 4, 100 trees, max_bin 32) | negative outcome | 64.8% | 61.1% | 68.3% | 3.40% | 6.01 | 0.0% | 0.0% |
| brfss2015_heart | frozen model | frozen (XGBoost, depth 4, 100 trees, max_bin 32) | enrichment | 27.4% | 24.1% | 30.9% | 75.91% | 5.18 | 0.0% | 0.0% |
| brfss2015_heart | model family | LightGBM | negative outcome | 63.2% | 59.5% | 66.8% | 2.85% | 6.67 | 0.0% | 0.0% |
| brfss2015_heart | model family | LightGBM | enrichment | 25.6% | 22.4% | 29.0% | 73.44% | 5.13 | 0.0% | 0.0% |
| brfss2015_heart | model family | random forest | negative outcome | 64.2% | 60.5% | 67.8% | 3.74% | 5.88 | 0.0% | 0.0% |
| brfss2015_heart | model family | random forest | enrichment | 29.8% | 26.4% | 33.3% | 77.85% | 3.26 | 0.0% | 0.0% |
| brfss2015_heart | tree depth | depth 3 | negative outcome | 65.2% | 61.5% | 68.7% | 3.68% | 5.78 | 0.0% | 0.0% |
| brfss2015_heart | tree depth | depth 3 | enrichment | 27.4% | 24.1% | 30.9% | 75.18% | 5.43 | 0.0% | 0.0% |
| brfss2015_heart | tree depth | depth 5 | negative outcome | 63.0% | 59.3% | 66.6% | 3.17% | 5.82 | 0.0% | 0.0% |
| brfss2015_heart | tree depth | depth 5 | enrichment | 25.4% | 22.2% | 28.8% | 74.02% | 5.43 | 0.0% | 0.0% |
| brfss2015_heart | tree depth | depth 6 | negative outcome | 65.0% | 61.3% | 68.5% | 3.69% | 5.63 | 0.0% | 0.0% |
| brfss2015_heart | tree depth | depth 6 | enrichment | 23.4% | 20.3% | 26.7% | 73.50% | 5.26 | 0.0% | 0.0% |
| brfss2015_heart | number of trees | 25 trees | negative outcome | 65.8% | 62.1% | 69.3% | 3.34% | 5.41 | 0.0% | 0.0% |
| brfss2015_heart | number of trees | 25 trees | enrichment | 28.8% | 25.5% | 32.3% | 76.39% | 4.88 | 0.0% | 0.0% |
| brfss2015_heart | number of trees | 50 trees | negative outcome | 64.8% | 61.1% | 68.3% | 3.40% | 6.01 | 0.0% | 0.0% |
| brfss2015_heart | number of trees | 50 trees | enrichment | 27.4% | 24.1% | 30.9% | 75.91% | 5.18 | 0.0% | 0.0% |
| brfss2015_heart | threshold bins | max_bin 16 | negative outcome | 64.4% | 60.7% | 68.0% | 3.73% | 5.70 | 0.0% | 0.0% |
| brfss2015_heart | threshold bins | max_bin 16 | enrichment | 25.6% | 22.4% | 29.0% | 74.22% | 5.23 | 0.0% | 0.0% |
| brfss2015_heart | threshold bins | max_bin 64 | negative outcome | 64.6% | 60.9% | 68.1% | 3.41% | 6.04 | 0.0% | 0.0% |
| brfss2015_heart | threshold bins | max_bin 64 | enrichment | 27.4% | 24.1% | 30.9% | 75.91% | 5.20 | 0.0% | 0.0% |
| brfss2015_heart | threshold bins | max_bin 256 | negative outcome | 64.6% | 60.9% | 68.1% | 3.41% | 6.04 | 0.0% | 0.0% |
| brfss2015_heart | threshold bins | max_bin 256 | enrichment | 27.4% | 24.1% | 30.9% | 75.91% | 5.20 | 0.0% | 0.0% |
| brfss2015_heart | monotone constraints | clinically signed | negative outcome | 66.6% | 63.0% | 70.1% | 3.60% | 5.29 | 0.0% | 0.0% |
| brfss2015_heart | monotone constraints | clinically signed | enrichment | 28.2% | 24.9% | 31.7% | 75.89% | 5.01 | 0.0% | 0.0% |

## Table 9. Shift: source certificates transported to target domains (E4)

Source: 13_shift.csv, 13b_shift.csv

| setting | target | role | targets_set | kind_name | explained | coverage | error_rate | error_upper_95 | refuted_share | above_target_share | certificate_refuted_share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| temporal | 2015 test split | reference | frozen | negative outcome | 500 | 67.8% | 2.95% | 5.0% | 0.0% | 0.0% | 0.0% |
| temporal | 2015 test split | reference | frozen | enrichment | 500 | 24.0% | 75.00% | 81.4% | 0.0% | 0.0% | 0.0% |
| temporal | 2017 | target | frozen | negative outcome | 1000 | 67.6% | 1.63% | 2.7% | 0.0% | 0.0% | 0.0% |
| temporal | 2017 | target | frozen | enrichment | 1000 | 25.5% | 77.65% | 81.9% | 0.0% | 0.0% | 0.0% |
| temporal | 2019 | target | frozen | negative outcome | 1000 | 65.3% | 1.23% | 2.2% | 0.0% | 0.0% | 0.0% |
| temporal | 2019 | target | frozen | enrichment | 1000 | 27.1% | 74.17% | 78.5% | 0.0% | 0.0% | 0.0% |
| temporal | 2021 | target | frozen | negative outcome | 1000 | 65.6% | 3.05% | 4.4% | 0.0% | 0.0% | 0.0% |
| temporal | 2021 | target | frozen | enrichment | 1000 | 26.7% | 75.66% | 79.9% | 0.0% | 0.0% | 0.0% |
| temporal | 2023 | target | frozen | negative outcome | 1000 | 63.3% | 2.84% | 4.2% | 0.0% | 0.0% | 0.0% |
| temporal | 2023 | target | frozen | enrichment | 1000 | 29.8% | 76.51% | 80.5% | 0.0% | 0.0% | 0.0% |
| geographic | Northeast held out: source test split | reference | frozen | negative outcome | 500 | 59.6% | 2.68% | 4.8% | 0.0% | 0.0% | 0.0% |
| geographic | Northeast held out: source test split | reference | frozen | enrichment | 500 | 29.6% | 66.89% | 73.3% | 0.0% | 0.0% | 0.0% |
| geographic | Northeast | target | frozen | negative outcome | 1000 | 67.9% | 1.91% | 3.0% | 0.0% | 0.0% | 0.0% |
| geographic | Northeast | target | frozen | enrichment | 1000 | 21.8% | 74.31% | 79.1% | 0.0% | 0.0% | 0.0% |
| geographic | Midwest held out: source test split | reference | frozen | negative outcome | 500 | 61.2% | 2.61% | 4.7% | 0.0% | 0.0% | 0.0% |
| geographic | Midwest held out: source test split | reference | frozen | enrichment | 500 | 22.6% | 68.14% | 75.4% | 0.0% | 0.0% | 0.0% |
| geographic | Midwest | target | frozen | negative outcome | 1000 | 66.1% | 2.27% | 3.5% | 0.0% | 0.0% | 0.0% |
| geographic | Midwest | target | frozen | enrichment | 1000 | 19.8% | 77.27% | 82.1% | 0.0% | 0.0% | 0.0% |
| state | Kansas | target | frozen | negative outcome | 500 | 63.8% | 2.51% | 4.5% | 0.0% | 0.0% | 0.0% |
| state | Kansas | target | frozen | enrichment | 500 | 22.8% | 71.05% | 78.0% | 0.0% | 0.0% | 0.0% |
| state | Nebraska | target | frozen | negative outcome | 500 | 65.4% | 2.14% | 4.0% | 0.0% | 0.0% | 0.0% |
| state | Nebraska | target | frozen | enrichment | 500 | 20.2% | 67.33% | 75.0% | 0.0% | 0.0% | 0.0% |
| state | Minnesota | target | frozen | negative outcome | 500 | 74.8% | 1.87% | 3.5% | 0.0% | 0.0% | 0.0% |
| state | Minnesota | target | frozen | enrichment | 500 | 17.6% | 65.91% | 74.3% | 0.0% | 0.0% | 0.0% |
| geographic | South held out: source test split | reference | frozen | negative outcome | 500 | 71.8% | 0.84% | 2.1% | 0.0% | 0.0% | 0.0% |
| geographic | South held out: source test split | reference | frozen | enrichment | 500 | 18.4% | 59.78% | 68.4% | 0.0% | 0.0% | 0.0% |
| geographic | South | target | frozen | negative outcome | 1000 | 64.8% | 2.31% | 3.5% | 0.0% | 0.0% | 0.0% |
| geographic | South | target | frozen | enrichment | 1000 | 24.5% | 68.57% | 73.5% | 0.0% | 0.0% | 0.0% |
| state | Texas | target | frozen | negative outcome | 500 | 63.2% | 2.85% | 4.9% | 0.0% | 0.0% | 0.0% |
| state | Texas | target | frozen | enrichment | 500 | 21.8% | 71.56% | 78.6% | 0.0% | 0.0% | 0.0% |
| geographic | West held out: source test split | reference | frozen | negative outcome | 500 | 65.8% | 2.43% | 4.3% | 0.0% | 0.0% | 0.0% |
| geographic | West held out: source test split | reference | frozen | enrichment | 500 | 26.6% | 69.17% | 75.8% | 0.0% | 0.0% | 0.0% |
| geographic | West | target | frozen | negative outcome | 1000 | 70.1% | 2.00% | 3.1% | 0.0% | 0.0% | 0.0% |
| geographic | West | target | frozen | enrichment | 1000 | 23.3% | 77.68% | 82.1% | 0.0% | 0.0% | 0.0% |
| state | Washington | target | frozen | negative outcome | 500 | 64.4% | 1.55% | 3.2% | 0.0% | 0.0% | 0.0% |
| state | Washington | target | frozen | enrichment | 500 | 28.2% | 74.47% | 80.4% | 0.0% | 0.7% | 0.0% |
| measurement | BRFSS 2015 test split | reference | frozen | negative outcome | 500 | 69.8% | 2.58% | 4.5% | 0.0% | 0.0% | 0.0% |
| measurement | BRFSS 2015 test split | reference | frozen | enrichment | 500 | 23.8% | 77.31% | 83.5% | 0.0% | 0.0% | 0.0% |
| measurement | NHANES, self-reported | target | frozen | negative outcome | 500 | 58.8% | 0.34% | 1.6% | 0.0% | 0.0% | 0.0% |
| measurement | NHANES, self-reported | target | frozen | enrichment | 500 | 34.6% | 82.08% | 86.7% | 0.0% | 0.6% | 0.0% |
| measurement | NHANES, measured BMI and blood pressure | target | frozen | negative outcome | 500 | 64.0% | 0.62% | 2.0% | 0.0% | 0.0% | 0.0% |
| measurement | NHANES, measured BMI and blood pressure | target | frozen | enrichment | 500 | 30.2% | 80.13% | 85.3% | 0.0% | 8.6% | 0.0% |
| site | Cleveland test split | reference | frozen | negative outcome | 46 | 0.0% | n/a | 100.0% | 0.0% | 0.0% | 0.0% |
| site | Cleveland test split | reference | frozen | enrichment | 46 | 32.6% | 13.33% | 36.3% | 0.0% | 0.0% | 0.0% |
| site | Hungary | target | frozen | negative outcome | 292 | 0.0% | n/a | 100.0% | 0.0% | 0.0% | 0.0% |
| site | Hungary | target | frozen | enrichment | 292 | 25.0% | 16.44% | 25.3% | 0.0% | 0.0% | 0.0% |
| site | Switzerland | target | frozen | negative outcome | 116 | 0.0% | n/a | 100.0% | 0.0% | 0.0% | 0.0% |
| site | Switzerland | target | frozen | enrichment | 116 | 34.5% | 5.00% | 14.9% | 0.0% | 0.0% | 0.0% |
| site | VA Long Beach | target | frozen | negative outcome | 140 | 0.0% | n/a | 100.0% | 0.0% | 0.0% | 0.0% |
| site | VA Long Beach | target | frozen | enrichment | 140 | 56.4% | 10.13% | 17.5% | 0.0% | 0.0% | 0.0% |
| site | Cleveland test split | reference | lenient | negative outcome | 46 | 0.0% | n/a | 100.0% | 0.0% | 0.0% | 0.0% |
| site | Cleveland test split | reference | lenient | enrichment | 46 | 37.0% | 11.76% | 32.6% | 0.0% | 0.0% | 0.0% |
| site | Hungary | target | lenient | negative outcome | 292 | 0.0% | n/a | 100.0% | 0.0% | 0.0% | 0.0% |
| site | Hungary | target | lenient | enrichment | 292 | 27.7% | 18.52% | 27.1% | 0.0% | 0.0% | 0.0% |
| site | Switzerland | target | lenient | negative outcome | 116 | 0.0% | n/a | 100.0% | 0.0% | 0.0% | 0.0% |
| site | Switzerland | target | lenient | enrichment | 116 | 38.8% | 4.44% | 13.3% | 0.0% | 0.0% | 0.0% |
| site | VA Long Beach | target | lenient | negative outcome | 140 | 0.0% | n/a | 100.0% | 0.0% | 0.0% | 0.0% |
| site | VA Long Beach | target | lenient | enrichment | 140 | 61.4% | 11.63% | 18.9% | 0.0% | 0.0% | 0.0% |

## Table 10. Subgroups: marginal and group-conditional SCAX, SGR (E5)

Source: 14_fairness_summary.csv, 14_fairness.csv

| task | attribute | method | kind | certified_share | coverage_cost_vs_marginal | worst_group | worst_group_error_rate | worst_group_error_lower_95 | worst_group_error_upper_95 | groups_above_claim_target | groups_policy_certified | groups |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cdc_diabetes | Sex | SCAX, marginal | negative outcome | 58.6% | 0.0% | Male | 4.3% | 3.4% | 5.4% | 0 | 1 | 2 |
| cdc_diabetes | Sex | SCAX, marginal | enrichment | 35.0% | 0.0% | Female | 68.8% | 66.2% | 71.4% | 0 | 0 | 2 |
| cdc_diabetes | Sex | SCAX, group-conditional | negative outcome | 56.7% | 1.9% | Male | 4.0% | 3.1% | 5.1% | 0 | 1 | 2 |
| cdc_diabetes | Sex | SCAX, group-conditional | enrichment | 34.3% | 0.7% | Female | 68.7% | 66.1% | 71.3% | 0 | 0 | 2 |
| cdc_diabetes | Sex | SCAX, group-conditional, strict | negative outcome | 47.7% | 11.0% | Male | 3.6% | 2.7% | 4.7% | 0 | 1 | 2 |
| cdc_diabetes | Sex | SCAX, group-conditional, strict | enrichment | 29.6% | 5.4% | Female | 67.5% | 64.6% | 70.2% | 0 | 0 | 2 |
| cdc_diabetes | Sex | SGR | negative outcome | 61.9% | -3.2% | Male | 4.9% | 3.9% | 6.0% | 0 | n/a | 2 |
| cdc_diabetes | Sex | SGR | enrichment | 34.8% | 0.2% | Female | 69.2% | 66.6% | 71.8% | 0 | n/a | 2 |
| cdc_diabetes | Sex | SGR, per group | negative outcome | 60.7% | -2.1% | Male | 4.6% | 3.7% | 5.7% | 0 | n/a | 2 |
| cdc_diabetes | Sex | SGR, per group | enrichment | 33.9% | 1.1% | Female | 68.6% | 66.0% | 71.2% | 0 | n/a | 2 |
| cdc_diabetes | Age band | SCAX, marginal | negative outcome | 58.6% | 0.0% | Age 65 or older | 5.8% | 4.3% | 7.6% | 1 | 1 | 3 |
| cdc_diabetes | Age band | SCAX, marginal | enrichment | 35.0% | 0.0% | Age 18 to 44 | 80.6% | 70.9% | 88.1% | 2 | 0 | 3 |
| cdc_diabetes | Age band | SCAX, group-conditional | negative outcome | 50.4% | 8.3% | Age 45 to 64 | 4.5% | 3.5% | 5.5% | 0 | 1 | 3 |
| cdc_diabetes | Age band | SCAX, group-conditional | enrichment | 32.9% | 2.1% | Age 45 to 64 | 69.3% | 66.3% | 72.2% | 0 | 0 | 3 |
| cdc_diabetes | Age band | SCAX, group-conditional, strict | negative outcome | 47.7% | 11.0% | Age 45 to 64 | 4.3% | 3.4% | 5.4% | 0 | 1 | 3 |
| cdc_diabetes | Age band | SCAX, group-conditional, strict | enrichment | 29.6% | 5.4% | Age 45 to 64 | 68.8% | 65.7% | 71.7% | 0 | 1 | 3 |
| cdc_diabetes | Age band | SGR | negative outcome | 61.9% | -3.2% | Age 65 or older | 6.9% | 5.4% | 8.7% | 1 | n/a | 3 |
| cdc_diabetes | Age band | SGR | enrichment | 34.8% | 0.2% | Age 18 to 44 | 80.3% | 70.5% | 87.9% | 1 | n/a | 3 |
| cdc_diabetes | Age band | SGR, per group | negative outcome | 56.9% | 1.7% | Age 45 to 64 | 4.8% | 3.9% | 5.8% | 0 | n/a | 3 |
| cdc_diabetes | Age band | SGR, per group | enrichment | 33.3% | 1.6% | Age 45 to 64 | 69.2% | 66.2% | 72.0% | 0 | n/a | 3 |
| cdc_diabetes | Income tertile | SCAX, marginal | negative outcome | 58.6% | 0.0% | Income under $35,000 | 5.8% | 4.3% | 7.6% | 1 | 0 | 3 |
| cdc_diabetes | Income tertile | SCAX, marginal | enrichment | 35.0% | 0.0% | Income $75,000 or more | 73.5% | 69.1% | 77.5% | 1 | 1 | 3 |
| cdc_diabetes | Income tertile | SCAX, group-conditional | negative outcome | 53.8% | 4.8% | Income under $35,000 | 4.9% | 3.3% | 7.0% | 0 | 0 | 3 |
| cdc_diabetes | Income tertile | SCAX, group-conditional | enrichment | 31.2% | 3.7% | Income $75,000 or more | 68.9% | 63.4% | 74.0% | 0 | 1 | 3 |
| cdc_diabetes | Income tertile | SCAX, group-conditional, strict | negative outcome | 47.7% | 11.0% | Income under $35,000 | 4.7% | 3.1% | 6.9% | 0 | 0 | 3 |
| cdc_diabetes | Income tertile | SCAX, group-conditional, strict | enrichment | 29.6% | 5.4% | Income $75,000 or more | 68.5% | 62.9% | 73.7% | 0 | 1 | 3 |
| cdc_diabetes | Income tertile | SGR | negative outcome | 61.9% | -3.2% | Income under $35,000 | 6.5% | 5.0% | 8.3% | 1 | n/a | 3 |
| cdc_diabetes | Income tertile | SGR | enrichment | 34.8% | 0.2% | Income $75,000 or more | 71.7% | 67.1% | 76.0% | 1 | n/a | 3 |
| cdc_diabetes | Income tertile | SGR, per group | negative outcome | 59.2% | -0.5% | Income under $35,000 | 4.8% | 3.4% | 6.6% | 0 | n/a | 3 |
| cdc_diabetes | Income tertile | SGR, per group | enrichment | 31.4% | 3.6% | Income $35,000 to $74,999 | 67.2% | 63.1% | 71.1% | 0 | n/a | 3 |
| brfss2015_heart | Sex | SCAX, marginal | negative outcome | 65.3% | 0.0% | Female | 2.4% | 1.8% | 3.0% | 0 | 2 | 2 |
| brfss2015_heart | Sex | SCAX, marginal | enrichment | 26.0% | 0.0% | Female | 75.0% | 71.8% | 78.0% | 2 | 1 | 2 |
| brfss2015_heart | Sex | SCAX, group-conditional | negative outcome | 64.9% | 0.3% | Female | 2.4% | 1.8% | 3.0% | 0 | 2 | 2 |
| brfss2015_heart | Sex | SCAX, group-conditional | enrichment | 21.1% | 4.9% | Male | 71.2% | 68.4% | 74.0% | 1 | 2 | 2 |
| brfss2015_heart | Sex | SCAX, group-conditional, strict | negative outcome | 58.9% | 6.3% | Male | 1.7% | 1.1% | 2.5% | 0 | 2 | 2 |
| brfss2015_heart | Sex | SCAX, group-conditional, strict | enrichment | 10.1% | 15.9% | Female | 67.2% | 61.3% | 72.6% | 0 | 2 | 2 |
| brfss2015_heart | Sex | SGR | negative outcome | 66.7% | -1.5% | Female | 2.4% | 1.9% | 3.0% | 0 | n/a | 2 |
| brfss2015_heart | Sex | SGR | enrichment | 33.3% | -7.3% | Female | 78.4% | 75.7% | 80.9% | 2 | n/a | 2 |
| brfss2015_heart | Sex | SGR, per group | negative outcome | 66.7% | -1.5% | Female | 2.4% | 1.9% | 3.0% | 0 | n/a | 2 |
| brfss2015_heart | Sex | SGR, per group | enrichment | 33.2% | -7.2% | Female | 78.3% | 75.6% | 80.8% | 2 | n/a | 2 |
| brfss2015_heart | Age band | SCAX, marginal | negative outcome | 65.3% | 0.0% | Age 65 or older | 4.4% | 3.1% | 6.1% | 0 | 2 | 3 |
| brfss2015_heart | Age band | SCAX, marginal | enrichment | 26.0% | 0.0% | Age 18 to 44 | 84.6% | 59.0% | 97.2% | 3 | 1 | 3 |
| brfss2015_heart | Age band | SCAX, group-conditional | negative outcome | 60.2% | 5.1% | Age 45 to 64 | 2.2% | 1.6% | 2.9% | 0 | 2 | 3 |
| brfss2015_heart | Age band | SCAX, group-conditional | enrichment | 19.7% | 6.3% | Age 65 or older | 71.7% | 69.1% | 74.3% | 1 | 2 | 3 |
| brfss2015_heart | Age band | SCAX, group-conditional, strict | negative outcome | 58.9% | 6.3% | Age 45 to 64 | 1.9% | 1.4% | 2.6% | 0 | 2 | 3 |
| brfss2015_heart | Age band | SCAX, group-conditional, strict | enrichment | 10.1% | 15.9% | Age 65 or older | 65.7% | 61.6% | 69.7% | 0 | 2 | 3 |
| brfss2015_heart | Age band | SGR | negative outcome | 66.7% | -1.5% | Age 65 or older | 4.6% | 3.3% | 6.2% | 0 | n/a | 3 |
| brfss2015_heart | Age band | SGR | enrichment | 33.3% | -7.3% | Age 18 to 44 | 90.0% | 71.7% | 98.2% | 3 | n/a | 3 |
| brfss2015_heart | Age band | SGR, per group | negative outcome | 65.8% | -0.6% | Age 65 or older | 4.2% | 3.0% | 5.9% | 0 | n/a | 3 |
| brfss2015_heart | Age band | SGR, per group | enrichment | 31.8% | -5.8% | Age 18 to 44 | 85.7% | 47.9% | 99.3% | 3 | n/a | 3 |
| brfss2015_heart | Income tertile | SCAX, marginal | negative outcome | 65.3% | 0.0% | Income under $35,000 | 4.2% | 3.0% | 5.6% | 0 | 3 | 3 |
| brfss2015_heart | Income tertile | SCAX, marginal | enrichment | 26.0% | 0.0% | Income $75,000 or more | 78.5% | 73.4% | 83.0% | 3 | 2 | 3 |
| brfss2015_heart | Income tertile | SCAX, group-conditional | negative outcome | 63.6% | 1.7% | Income under $35,000 | 3.4% | 2.3% | 4.8% | 0 | 3 | 3 |
| brfss2015_heart | Income tertile | SCAX, group-conditional | enrichment | 16.8% | 9.2% | Income under $35,000 | 71.8% | 68.6% | 74.8% | 1 | 3 | 3 |
| brfss2015_heart | Income tertile | SCAX, group-conditional, strict | negative outcome | 58.9% | 6.3% | Income under $35,000 | 2.9% | 1.9% | 4.3% | 0 | 3 | 3 |
| brfss2015_heart | Income tertile | SCAX, group-conditional, strict | enrichment | 10.1% | 15.9% | Income under $35,000 | 66.4% | 62.1% | 70.5% | 0 | 2 | 3 |
| brfss2015_heart | Income tertile | SGR | negative outcome | 66.7% | -1.5% | Income under $35,000 | 4.2% | 3.0% | 5.5% | 0 | n/a | 3 |
| brfss2015_heart | Income tertile | SGR | enrichment | 33.3% | -7.3% | Income $75,000 or more | 80.9% | 76.6% | 84.7% | 3 | n/a | 3 |
| brfss2015_heart | Income tertile | SGR, per group | negative outcome | 66.7% | -1.5% | Income under $35,000 | 4.2% | 3.0% | 5.5% | 0 | n/a | 3 |
| brfss2015_heart | Income tertile | SGR, per group | enrichment | 32.3% | -6.3% | Income $75,000 or more | 79.0% | 74.1% | 83.4% | 3 | n/a | 3 |
| cardio70k | Sex | SCAX, marginal | negative outcome | 29.1% | 0.0% | Male | 23.8% | 20.6% | 27.2% | 1 | 0 | 2 |
| cardio70k | Sex | SCAX, marginal | enrichment | 28.4% | 0.0% | Female | 16.0% | 14.0% | 18.2% | 0 | 2 | 2 |
| cardio70k | Sex | SCAX, group-conditional | negative outcome | 22.6% | 6.5% | Female | 19.4% | 17.3% | 21.7% | 0 | 0 | 2 |
| cardio70k | Sex | SCAX, group-conditional | enrichment | 17.3% | 11.1% | Female | 16.0% | 14.0% | 18.2% | 0 | 1 | 2 |
| cardio70k | Sex | SCAX, group-conditional, strict | negative outcome | 10.5% | 18.6% | Female | 14.0% | 11.1% | 17.3% | 0 | 1 | 2 |
| cardio70k | Sex | SCAX, group-conditional, strict | enrichment | 16.1% | 12.3% | Female | 16.2% | 14.1% | 18.5% | 0 | 1 | 2 |
| cardio70k | Sex | SGR | negative outcome | 28.6% | 0.5% | Male | 19.4% | 16.5% | 22.7% | 0 | n/a | 2 |
| cardio70k | Sex | SGR | enrichment | 35.5% | -7.2% | Female | 19.0% | 17.1% | 21.1% | 0 | n/a | 2 |
| cardio70k | Sex | SGR, per group | negative outcome | 27.1% | 2.0% | Male | 18.8% | 15.5% | 22.3% | 0 | n/a | 2 |
| cardio70k | Sex | SGR, per group | enrichment | 33.9% | -5.5% | Female | 18.8% | 16.9% | 20.8% | 0 | n/a | 2 |
| cardio70k | Age band | SCAX, marginal | negative outcome | 29.1% | 0.0% | Age 45 to 64 | 25.2% | 22.8% | 27.7% | 1 | 1 | 2 |
| cardio70k | Age band | SCAX, marginal | enrichment | 28.4% | 0.0% | Age 45 to 64 | 16.3% | 14.6% | 18.1% | 0 | 1 | 2 |
| cardio70k | Age band | SCAX, group-conditional | negative outcome | 11.7% | 17.4% | Age 18 to 44 | 14.7% | 12.3% | 17.3% | 0 | 1 | 2 |
| cardio70k | Age band | SCAX, group-conditional | enrichment | 26.0% | 2.4% | Age 45 to 64 | 16.3% | 14.6% | 18.1% | 0 | 1 | 2 |
| cardio70k | Age band | SCAX, group-conditional, strict | negative outcome | 10.5% | 18.6% | Age 18 to 44 | 13.9% | 11.5% | 16.6% | 0 | 1 | 2 |
| cardio70k | Age band | SCAX, group-conditional, strict | enrichment | 16.1% | 12.3% | Age 45 to 64 | 16.2% | 14.1% | 18.5% | 0 | 1 | 2 |
| cardio70k | Age band | SGR | negative outcome | 28.6% | 0.5% | Age 45 to 64 | 21.1% | 18.8% | 23.5% | 1 | n/a | 2 |
| cardio70k | Age band | SGR | enrichment | 35.5% | -7.2% | Age 45 to 64 | 18.9% | 17.3% | 20.5% | 0 | n/a | 2 |
| cardio70k | Age band | SGR, per group | negative outcome | 20.5% | 8.6% | Age 45 to 64 | 18.2% | 15.1% | 21.5% | 0 | n/a | 2 |
| cardio70k | Age band | SGR, per group | enrichment | 32.4% | -4.0% | Age 45 to 64 | 18.6% | 17.0% | 20.2% | 0 | n/a | 2 |

## Table 11. Runtime of the main results, seconds per distinct cell (E6)

Source: 15_runtime_production.csv

| task | family | explained_patients | distinct_cells | trees_with_a_split | internal_nodes | median_seconds | p95_seconds | max_seconds | memo_hit_rate | search_unknown |
|---|---|---|---|---|---|---|---|---|---|---|
| cdc_diabetes | xgboost | 7874 | 7775 | 33 | 420 | 0.55 | 1.24 | 3.12 | 25.9% | 0 |
| cdc_diabetes | lightgbm | 7874 | 7767 | 31 | 358 | 0.56 | 1.21 | 3.91 | 24.4% | 0 |
| cdc_diabetes | sklearn_rf | 7874 | 7679 | 50 | 747 | 0.39 | 0.95 | 2.70 | 34.8% | 0 |
| brfss2015_heart | xgboost | 14346 | 13788 | 31 | 391 | 0.44 | 0.94 | 2.77 | 29.4% | 0 |
| brfss2015_heart | lightgbm | 14346 | 13798 | 27 | 327 | 0.39 | 0.81 | 3.10 | 27.3% | 0 |
| brfss2015_heart | sklearn_rf | 14346 | 13508 | 50 | 747 | 0.16 | 0.55 | 1.82 | 48.3% | 0 |
| cardio70k | xgboost | 15500 | 13356 | 20 | 220 | 0.04 | 0.25 | 0.64 | 64.4% | 0 |

## Table 12. Runtime scaling, timing study (E6)

Source: 15_runtime.csv

| task | series | n_trees | max_depth | trees_with_a_split | internal_nodes | oof_auroc_training_portion | cells | censored_cells | median_seconds | median_is_lower_bound | p95_seconds | search_share | inflation_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cdc_diabetes | frozen settings | 100 | 4 | 33 | 420 | 0.829 | 100 | 0 | 0.75 | no | 1.69 | 93.2% | 69.1% |
| brfss2015_heart | frozen settings | 25 | 3 | 25 | 171 | 0.849 | 100 | 0 | 0.15 | no | 0.26 | 83.9% | 50.5% |
| brfss2015_heart | frozen settings | 50 | 3 | 39 | 262 | 0.849 | 100 | 0 | 0.50 | no | 0.82 | 91.9% | 63.7% |
| brfss2015_heart | frozen settings | 100 | 3 | 39 | 262 | 0.849 | 100 | 0 | 0.51 | no | 0.89 | 92.4% | 63.2% |
| brfss2015_heart | frozen settings | 200 | 3 | 39 | 262 | 0.849 | 100 | 0 | 0.59 | no | 1.06 | 92.3% | 62.6% |
| brfss2015_heart | frozen settings | 25 | 4 | 25 | 332 | 0.849 | 100 | 0 | 0.33 | no | 0.69 | 88.9% | 56.0% |
| brfss2015_heart | frozen settings | 50 | 4 | 31 | 391 | 0.849 | 100 | 0 | 0.55 | no | 0.97 | 91.9% | 62.9% |
| brfss2015_heart | frozen settings | 100 | 4 | 31 | 391 | 0.849 | 100 | 0 | 0.60 | no | 1.12 | 92.6% | 63.3% |
| brfss2015_heart | frozen settings | 200 | 4 | 31 | 391 | 0.849 | 100 | 0 | 0.71 | no | 1.26 | 92.4% | 62.9% |
| brfss2015_heart | frozen settings | 25 | 5 | 25 | 567 | 0.849 | 100 | 0 | 0.55 | no | 1.46 | 92.4% | 64.4% |
| brfss2015_heart | frozen settings | 50 | 5 | 28 | 606 | 0.849 | 100 | 0 | 0.76 | no | 1.52 | 92.8% | 66.3% |
| brfss2015_heart | frozen settings | 100 | 5 | 28 | 606 | 0.849 | 100 | 0 | 0.78 | no | 1.83 | 93.1% | 65.9% |
| brfss2015_heart | frozen settings | 200 | 5 | 28 | 606 | 0.849 | 100 | 0 | 0.90 | no | 1.94 | 93.5% | 66.2% |
| brfss2015_heart | frozen settings | 25 | 6 | 24 | 828 | 0.849 | 100 | 0 | 0.80 | no | 2.04 | 93.5% | 67.7% |
| brfss2015_heart | frozen settings | 50 | 6 | 24 | 828 | 0.849 | 100 | 0 | 0.83 | no | 2.10 | 93.2% | 67.3% |
| brfss2015_heart | frozen settings | 100 | 6 | 24 | 828 | 0.849 | 100 | 0 | 0.89 | no | 2.19 | 93.5% | 67.4% |
| brfss2015_heart | frozen settings | 200 | 6 | 24 | 828 | 0.849 | 100 | 0 | 0.97 | no | 2.42 | 93.7% | 67.2% |
| cardio70k | frozen settings | 25 | 3 | 24 | 152 | 0.799 | 100 | 0 | 0.09 | no | 0.23 | 83.5% | 65.9% |
| cardio70k | frozen settings | 50 | 3 | 24 | 152 | 0.799 | 100 | 0 | 0.07 | no | 0.19 | 85.9% | 65.0% |
| cardio70k | frozen settings | 100 | 3 | 24 | 152 | 0.799 | 100 | 0 | 0.08 | no | 0.24 | 86.3% | 65.2% |
| cardio70k | frozen settings | 200 | 3 | 24 | 152 | 0.799 | 100 | 0 | 0.10 | no | 0.25 | 84.7% | 63.7% |
| cardio70k | frozen settings | 25 | 4 | 20 | 220 | 0.799 | 100 | 0 | 0.10 | no | 0.30 | 88.4% | 72.1% |
| cardio70k | frozen settings | 50 | 4 | 20 | 220 | 0.799 | 100 | 0 | 0.11 | no | 0.33 | 88.6% | 72.5% |
| cardio70k | frozen settings | 100 | 4 | 20 | 220 | 0.799 | 100 | 0 | 0.11 | no | 0.36 | 88.4% | 71.8% |
| cardio70k | frozen settings | 200 | 4 | 20 | 220 | 0.799 | 100 | 0 | 0.14 | no | 0.43 | 88.9% | 72.3% |
| cardio70k | frozen settings | 25 | 5 | 17 | 285 | 0.798 | 100 | 0 | 0.11 | no | 0.34 | 86.4% | 69.2% |
| cardio70k | frozen settings | 50 | 5 | 17 | 285 | 0.798 | 100 | 0 | 0.12 | no | 0.37 | 88.3% | 70.6% |
| cardio70k | frozen settings | 100 | 5 | 17 | 285 | 0.798 | 100 | 0 | 0.11 | no | 0.41 | 88.5% | 70.7% |
| cardio70k | frozen settings | 200 | 5 | 17 | 285 | 0.798 | 100 | 0 | 0.13 | no | 0.45 | 87.6% | 70.0% |
| cardio70k | frozen settings | 25 | 6 | 15 | 347 | 0.798 | 100 | 0 | 0.11 | no | 0.36 | 88.2% | 72.7% |
| cardio70k | frozen settings | 50 | 6 | 15 | 347 | 0.798 | 100 | 0 | 0.11 | no | 0.36 | 88.3% | 73.0% |
| cardio70k | frozen settings | 100 | 6 | 15 | 347 | 0.798 | 100 | 0 | 0.14 | no | 0.39 | 87.4% | 73.3% |
| cardio70k | frozen settings | 200 | 6 | 15 | 347 | 0.798 | 100 | 0 | 0.14 | no | 0.44 | 89.0% | 73.5% |
| brfss2015_heart | no regularisation | 25 | 3 | 25 | 175 | 0.849 | 100 | 0 | 0.14 | no | 0.33 | 85.2% | 50.8% |
| brfss2015_heart | no regularisation | 50 | 3 | 50 | 350 | 0.849 | 100 | 0 | 1.25 | no | 2.56 | 96.0% | 73.0% |
| brfss2015_heart | no regularisation | 100 | 3 | 100 | 696 | 0.849 | 20 | 0 | 10.97 | no | 29.08 | 99.3% | 90.0% |
| brfss2015_heart | no regularisation | 200 | 3 | 200 | 1390 | 0.847 | 4 | 0 | 94.56 | no | 112.26 | 99.8% | 92.1% |
| brfss2015_heart | no regularisation | 25 | 4 | 25 | 373 | 0.850 | 100 | 0 | 0.52 | no | 0.96 | 91.2% | 65.8% |
| brfss2015_heart | no regularisation | 50 | 4 | 50 | 735 | 0.849 | 28 | 0 | 7.44 | no | 22.41 | 99.0% | 86.4% |
| brfss2015_heart | no regularisation | 100 | 4 | 100 | 1470 | 0.846 | 5 | 0 | 42.94 | no | 71.81 | 99.7% | 91.7% |
| brfss2015_heart | no regularisation | 200 | 4 | 200 | 2928 | 0.843 | 3 | 2 | 120.00 | yes | 120.00 | 99.1% | 93.9% |
| brfss2015_heart | no regularisation | 25 | 6 | 25 | 1475 | 0.848 | 33 | 0 | 6.43 | no | 16.23 | 98.1% | 84.4% |
| brfss2015_heart | no regularisation | 50 | 6 | 50 | 2824 | 0.844 | 4 | 1 | 56.81 | no | 112.01 | 99.5% | 92.4% |
| brfss2015_heart | no regularisation | 100 | 6 | 100 | 5384 | 0.837 | 3 | 2 | 120.00 | yes | 120.00 | 99.3% | 94.9% |
| brfss2015_heart | no regularisation | 200 | 6 | 200 | 10683 | 0.827 | 2 | 2 | 120.00 | yes | 120.00 | n/a | n/a |
| cardio70k | no regularisation | 25 | 3 | 25 | 175 | 0.799 | 100 | 0 | 0.15 | no | 0.37 | 87.7% | 73.6% |
| cardio70k | no regularisation | 50 | 3 | 50 | 350 | 0.799 | 100 | 0 | 0.57 | no | 1.28 | 95.7% | 84.9% |
| cardio70k | no regularisation | 100 | 3 | 100 | 697 | 0.798 | 92 | 0 | 2.49 | no | 5.31 | 98.4% | 90.7% |
| cardio70k | no regularisation | 200 | 3 | 200 | 1397 | 0.796 | 21 | 0 | 11.68 | no | 21.80 | 99.4% | 96.1% |
| cardio70k | no regularisation | 25 | 4 | 25 | 375 | 0.799 | 100 | 0 | 0.40 | no | 1.04 | 95.0% | 82.1% |
| cardio70k | no regularisation | 50 | 4 | 50 | 745 | 0.797 | 100 | 0 | 2.22 | no | 4.68 | 98.2% | 90.1% |
| cardio70k | no regularisation | 100 | 4 | 100 | 1487 | 0.794 | 35 | 0 | 6.86 | no | 11.46 | 99.0% | 93.5% |
| cardio70k | no regularisation | 200 | 4 | 200 | 2967 | 0.790 | 16 | 0 | 17.70 | no | 26.19 | 99.4% | 93.4% |
| cardio70k | no regularisation | 25 | 6 | 25 | 1420 | 0.794 | 100 | 0 | 1.81 | no | 4.06 | 97.2% | 89.8% |
| cardio70k | no regularisation | 50 | 6 | 50 | 2769 | 0.789 | 43 | 0 | 5.18 | no | 9.18 | 98.3% | 92.0% |
| cardio70k | no regularisation | 100 | 6 | 100 | 5429 | 0.782 | 17 | 0 | 12.37 | no | 23.34 | 98.9% | 91.6% |
| cardio70k | no regularisation | 200 | 6 | 200 | 10813 | 0.772 | 8 | 0 | 31.68 | no | 45.56 | 99.1% | 95.0% |

## Table 14. Survey-weighted sensitivity analysis, frozen XGBoost (plan Section 6.2)

Source: 16_weighted_sensitivity.csv

| task_name | kind | certified_share | Share | outcome_share | Outcome | prevalence_explained_test | prevalence_explained_test_weighted | prevalence_table | prevalence_table_weighted |
|---|---|---|---|---|---|---|---|---|---|
| BRFSS Heart disease | negative outcome | 65.3% | 72.0% (69.8 to 74.2) | 2.3% | 1.6% (1.0 to 2.2) | 9.5% | 6.9% | 9.4% | 7.4% |
| BRFSS Heart disease | enrichment | 26.0% | 20.6% (18.7 to 22.7) | 27.1% | 24.5% (20.5 to 28.9) | 9.5% | 6.9% | 9.4% | 7.4% |
| CDC Diabetes | negative outcome | 58.6% | 64.5% (62.0 to 66.9) | 4.0% | 3.9% (2.5 to 5.7) | 14.6% | 13.7% | 13.9% | 12.3% |
| CDC Diabetes | enrichment | 35.0% | 29.8% (27.6 to 32.2) | 32.2% | 33.4% (29.1 to 37.8) | 14.6% | 13.7% | 13.9% | 12.3% |

Weighted columns use the BRFSS 2015 final survey weight (_LLCPWT) of each explained test patient; intervals: 95% percentile bootstrap over patients with the weights carried along. Descriptive only: the certificates are not weighted.

## Table 15. Explanation cards: the box released to the most explained test patients

Source: 16_example_cards.csv, 09_test_releases.csv, 09d_test_releases.csv

| task_name | kind | explanation | features | calibration_records | bound | claim_target | claim | explained_test_patients_given_it | outcome_share_among_them |
|---|---|---|---|---|---|---|---|---|---|
| BRFSS Heart disease | negative outcome | HighBP < 1 and Stroke < 1 and GenHlth < 3 and Age < 5 | 4 | 6661 | 0.6% | 5.0% | at most 0.6% have the outcome (target at most 5.0%) | 441 | 0.2% |
| BRFSS Heart disease | enrichment | 1 <= HighBP and 1 <= HighChol and 4 <= GenHlth and 10 <= Age | 4 | 2950 | 63.4% | 70.0% | at least 36.6% have the outcome (target at least 30.0%) | 94 | 34.0% |
| CDC Diabetes | negative outcome | HighBP < 1 and Stroke < 1 and GenHlth < 2 and Age < 10 | 4 | 8467 | 1.2% | 5.0% | at most 1.2% have the outcome (target at most 5.0%) | 198 | 1.0% |
| CDC Diabetes | enrichment | 1 <= CholCheck and 31 <= BMI and HvyAlcoholConsump < 1 and 3 <= GenHlth and 9 <= Age | 5 | 6405 | 60.6% | 70.0% | at least 39.4% have the outcome (target at least 30.0%) | 226 | 37.6% |
| Cardio-70k | negative outcome | age < 16059 and ap_hi < 130 and cholesterol < 2 | 3 | 1761 | 15.6% | 20.0% | at most 15.6% have the outcome (target at most 20.0%) | 436 | 12.2% |
| Cardio-70k | enrichment | 141 <= ap_hi | 1 | 2943 | 17.1% | 20.0% | at least 82.9% have the outcome (target at least 80.0%) | 712 | 88.1% |

## Table 13. Candidate explanations of the frozen models

Source: 16_candidate_stats.csv

| statistic | CDC Diabetes | BRFSS Heart disease | Cardio-70k |
|---|---|---|---|
| distinct_cells | 7775 | 1.379e+04 | 1.336e+04 |
| cells_without_box_rows | 0 | 0 | 0 |
| candidates_median | 65 | 51 | 46 |
| candidates_q25 | 49 | 36 | 24 |
| candidates_q75 | 81 | 68 | 77 |
| axp_share_of_candidates | 0.09749 | 0.1395 | 0.06661 |
| smallest_size_median | 5 | 6 | 4 |
| largest_size_median | 9 | 9 | 5 |
| largest_support_median | 5503 | 6661 | 2943 |
| cells_with_releasable_box | 0.9336 | 0.9171 | 0.5761 |
| distinct_boxes | 2.547e+05 | 3.016e+05 | 1.493e+05 |
| negative_outcome_boxes | 1.487e+05 | 1.589e+05 | 6.818e+04 |
| enrichment_boxes | 1.06e+05 | 1.427e+05 | 8.11e+04 |
| box_support_median | 63 | 45 | 13 |
| second_check_boxes | 6.378e+04 | 6.236e+04 | 2.719e+04 |
| second_check_sound | 6.378e+04 | 6.236e+04 | 2.719e+04 |
| second_check_unknown | 0 | 0 | 0 |
| seconds_median | 0.5462 | 0.439 | 0.03537 |
| seconds_p95 | 1.242 | 0.9356 | 0.2548 |
| checks | 3.117e+06 | 5.103e+06 | 3.469e+06 |
| search_unknown | 0 | 0 | 0 |
| memo_hit_rate | 0.2592 | 0.2935 | 0.6435 |

## Figures

| paper_figure | file | sources |
|---|---|---|
| Figure 1 | figures/fig_framework.png | schematic (no data) |
| Figure 2 | figures/fig_mechanism.png | schematic (simulated points, illustrative numbers) |
| Figure 3 | figures/fig_validity_naive_vs_scax.png | 10_validity.csv |
| Figure 5 | figures/fig_validity_overlap_p4.png | 10_validity.csv |
| Figure 4 | figures/fig_validity_delta.png | 10_validity.csv, 10b_validity_more_repetitions.csv |
| Figure 6 | figures/fig_certified_share_curves.png | 09b_main.csv, 09e_cardio_main.csv |
| not in the manuscript (covered by Figure 6) | figures/fig_certified_share_vs_target.png | 09_main.csv |
| not in the manuscript (covered by Figure 6) | figures/fig_certified_share_by_family.png | 09b_main.csv |
| not in the manuscript (covered by Figure 6) | figures/fig_cardio_certified_share.png | 09d_main.csv |
| Figure A1 | figures/fig_families_cardio_uci.png | 09e_cardio_main.csv, 09e_uci_main.csv |
| Figure 7 | figures/fig_coverage_vs_true_claim_precision.png | 11_synthetic_summary.csv |
| Figure A2 | figures/fig_baselines_claims_real.png | 11_baselines_summary.csv |
| Figure 8 | figures/fig_shift_recertification.png | 13_shift_recertification.csv, 13_shift_targets.csv, 13_shift.csv |
| Figure A6 | figures/fig_shift_measurement.png | 13b_shift_recertification.csv, 13b_shift.csv |
| Figure A7 | figures/fig_shift_box_diagnostics.png | 13_shift_boxes.csv |
| Figure 9 | figures/fig_subgroups_negative_outcome.png | 14_fairness.csv, 14_group_sizes.csv |
| Figure A5 | figures/fig_subgroups_enrichment.png | 14_fairness.csv, 14_group_sizes.csv |
| Figure 10 | figures/fig_runtime.png | 15_runtime.csv, 15_runtime.json |
| Figure A3 | figures/fig_ablations_frozen_model.png | 12_ablations_summary.csv |
| Figure A4 | figures/fig_ablations_model_factors.png | 12b_model_ablations.csv |
| Figure A8 | figures/fig_nhanes_measurement.png | 16_nhanes_measurement_summary.csv (from the NHANES table of notebook 03) |
