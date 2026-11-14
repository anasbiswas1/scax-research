# Claims check

Every numerical claim planned for the paper, written by notebook 16 from the results file it cites, with a check computed from that file. A claim whose check does not hold is not used in the paper as stated.

53 of 53 checks hold.

## Soundness (Theorem 1 in practice, gate G2)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| S1 | No disagreement was recorded between the encodings, brute force, the two exact checks, or exported and native predictions, in any check recorded in the cited files. | 06_encoding_tests.json, configs/frozen/*.json, 08_certificates.json, 09d_results.json, 09b_families.json, 09e_families.json, 13_shift.json, 13b_shift.json, 12b_model_variants.csv, 10_settings.csv | yes |  |
| S2 | Every box that could be released at the loosest grid target for the frozen XGBoost models (153,329 boxes over three tasks) was confirmed sound by the second exact implementation; none was undecided. | 08_certificates.json, 09d_results.json | yes |  |
| S3 | Brute-force enumeration on small ensembles: 0 errors in 2,400 checks. | 06_encoding_tests.json | yes |  |

## Validity (E2, RQ1, gate G4)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| V1 | No SCAX configuration showed excess box-certificate violations (936 configurations, Occam and mixture priors); the largest violation frequency was 0.026. | 10_validity.csv, 10b_validity_more_repetitions.csv | yes |  |
| V2 | No configuration showed excess policy violations; the largest frequency was 0.010 against delta_2 = 0.025. | 10_validity.csv, 10b_validity_more_repetitions.csv | yes |  |
| V3 | Per-box bounds without the prior showed excess violations in 356 of 468 configurations (Proposition 3). | 10_validity.csv | yes |  |
| V4 | With the Occam prior every configuration was 'low enough' (468 of 468); with the mixture prior every configuration was 'low enough' (468 of 468), 35 of them after 5,000 repetitions because 500 had left them inconclusive. | 10_validity.csv, 10b_validity_more_repetitions.csv | yes |  |
| V5 | Overlap problem on P4 (10,000 calibration records, delta_1 = 0.025): every released box was within target while the released patients were above alpha_sel in up to 52% of repetitions (mixture prior), and the policy test certified such a policy in at most 1.0%. | 10_validity.csv | yes |  |

## Main results (E1, RQ2, RQ3)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| M1 | The release policy was certified in 10 of 18 task, model and claim units with a policy test: BRFSS Heart disease 6 of 6; CDC Diabetes 1 of 6; Cardio-70k 3 of 6. | 09b_main.csv, 09e_cardio_main.csv | yes |  |
| M2 | In every unit whose policy was certified, the share with the outcome among released test patients met the policy target (at most alpha_sel for negative-outcome claims, at least 1 - alpha_sel for enrichment claims). | 09b_main.csv, 09e_cardio_main.csv | yes |  |
| M-brfss2015_heart-negative-outcome | BRFSS Heart disease, frozen XGBoost, negative outcome: at most 5.0% with the outcome: policy certified (upper bound 2.8%, target at most 5.0%); 65.3% (63.9 to 66.6) of explained test patients received a certified claim, and 2.3% (1.8 to 2.8) of them had the outcome. | 09b_main.csv, 09b_model_metrics.csv, 09_test_releases.csv | yes |  |
| M-brfss2015_heart-enrichment | BRFSS Heart disease, frozen XGBoost, enrichment: at least 30.0% with the outcome: policy certified (lower bound 23.1%, target at least 20.0%); 26.0% (24.8 to 27.2) of explained test patients received a certified claim, and 27.1% (24.7 to 29.5) of them had the outcome. | 09b_main.csv, 09b_model_metrics.csv, 09_test_releases.csv | yes |  |
| M-cdc_diabetes-negative-outcome | CDC Diabetes, frozen XGBoost, negative outcome: at most 5.0% with the outcome: policy certified (upper bound 4.5%, target at most 5.0%); 58.6% (57.3 to 60.0) of explained test patients received a certified claim, and 4.0% (3.3 to 4.7) of them had the outcome. | 09b_main.csv, 09b_model_metrics.csv, 09_test_releases.csv | yes |  |
| M-cdc_diabetes-enrichment | CDC Diabetes, frozen XGBoost, enrichment: at least 30.0% with the outcome: policy not certified (lower bound 28.8%, target at least 30.0%); 35.0% (33.7 to 36.3) of explained test patients received a certified claim, and 32.2% (30.1 to 34.4) of them had the outcome. | 09b_main.csv, 09b_model_metrics.csv, 09_test_releases.csv | yes |  |
| M-cardio70k-negative-outcome | Cardio-70k, frozen XGBoost, negative outcome: at most 20.0% with the outcome: policy not certified (upper bound 23.1%, target at most 20.0%); 29.1% (27.9 to 30.3) of explained test patients received a certified claim, and 21.0% (18.9 to 23.1) of them had the outcome. | 09e_cardio_main.csv, 09e_cardio_model_metrics.csv, 09d_test_releases.csv | yes |  |
| M-cardio70k-enrichment | Cardio-70k, frozen XGBoost, enrichment: at least 80.0% with the outcome: policy certified (lower bound 82.6%, target at least 80.0%); 28.4% (27.1 to 29.6) of explained test patients received a certified claim, and 84.1% (82.1 to 85.9) of them had the outcome. | 09e_cardio_main.csv, 09e_cardio_model_metrics.csv, 09d_test_releases.csv | yes |  |
| M3 | No released box had a test error significantly above its target on the full test split (Bonferroni-corrected lower limits; 0 flags over all models). | 09_box_consistency.csv, 09d_box_consistency.csv, 09b_box_consistency.csv, 09e_cardio_box_consistency.csv | yes |  |
| M4 | UCI Heart (each site held out, three models): no negative-outcome claim was certified at alpha_0 = 5.0%; enrichment claims were certified at every held-out site for every model (8.2% to 78.4% of patients), with 73% to 98% of them having angiographic disease and no claim refuted. | 09e_uci_main.csv | yes |  |
| M5 | Test AUROC of the underlying models: BRFSS Heart disease XGBoost 0.848; BRFSS Heart disease LightGBM 0.848; BRFSS Heart disease Random forest 0.829; CDC Diabetes XGBoost 0.830; CDC Diabetes LightGBM 0.828; CDC Diabetes Random forest 0.812; Cardio-70k XGBoost 0.803; Cardio-70k LightGBM 0.802; Cardio-70k Random forest 0.795. | 09b_model_metrics.csv, 09e_cardio_model_metrics.csv | yes |  |
| M6 | Bootstrap intervals from the saved releases and from the saved counts agree for the 3 XGBoost models, and the saved releases reproduce the saved counts. | 09_test_releases.csv, 09d_test_releases.csv, 16_bootstrap_check.csv | yes |  |

## Baselines (E1, E3)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| B1 | SCAX released no box whose test error was above its target in either task or claim kind (mean over 20 calibration re-draws). | 11_baselines_summary.csv | yes |  |
| B2 | Uncertified AXps at SCAX's coverage released boxes above their target for 4.7% to 13.4% of negative-outcome and 42.5% to 45.0% of enrichment patients. | 11_baselines_summary.csv | yes |  |
| B3 | None of the rules of the training rule dictionary used for releases is sound for the deployed model (sound share 0.0%). | 11_baselines_summary.csv | yes |  |
| B4 | On the planted rule P4 SCAX made no false claim in any repetition at either target pair; per-box bounds without the prior made at least one false claim in 44.5% and 8.0% of repetitions, the empirical rule in 100.0%. | 11_synthetic_summary.csv | yes |  |

## Ablations (E3)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| A1 | Keeping every sound box on the inflation path, rather than the AXps only, raised the certified share from 27.4% to 58.6% (CDC Diabetes, negative outcome); 5.0% to 35.0% (CDC Diabetes, enrichment); 46.0% to 65.3% (BRFSS Heart disease, negative outcome); 6.8% to 26.0% (BRFSS Heart disease, enrichment). | 12_ablations_summary.csv | yes |  |
| A2 | The three selection rules gave the same certified share for both tasks and kinds (they change which box a patient receives, not whether one is certified). | 12_ablations_summary.csv | yes |  |
| A3 | The certified share grew with the number of candidate orders M (1 to 10) for both tasks and kinds. | 12_ablations_summary.csv | yes |  |
| A4 | No model variant (family, depth, trees, bins, monotone constraints) released a box refuted on the test split. | 12b_model_ablations.csv | yes |  |

## Shift (E4, RQ4)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| H1 | Source certificates transported to 18 target domains (later years, held-out regions, states, NHANES with self-reported and measured values, UCI sites) had no refuted claim and no refuted certificate. | 13_shift.csv, 13b_shift.csv | yes |  |
| H2 | The share of released patients whose box had a target error estimate above its target was at most 8.6% (NHANES, measured BMI and blood pressure, enrichment). | 13_shift.csv, 13b_shift.csv | yes |  |

## Subgroups (E5, RQ5)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| SG1 | Marginal SCAX left at least one group whose observed share with the outcome missed the claim target in 9 of 16 task, attribute and kind combinations. | 14_fairness_summary.csv | yes |  |
| SG2 | Group-conditional SCAX (standard) left no group missing the negative-outcome claim target. | 14_fairness_summary.csv | yes |  |
| SG3 | The strict version left no group missing the claim target, for either kind. | 14_fairness_summary.csv | yes |  |
| SG4 | Cost in certified share against marginal SCAX: standard 0.3 to 17.4 percentage points, strict 5.4 to 18.6 percentage points of explained test patients. | 14_fairness_summary.csv | yes |  |

## Runtime (E6, RQ6, gate G5)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| R1 | Production runtime of the main results: median 0.04 to 0.56 seconds per distinct cell, 95th percentile at most 1.24 s, maximum 3.9 s, no search stopped by the node budget (7 models). | 15_runtime_production.csv | yes |  |
| R2 | With the frozen settings every model up to 100 trees and depth 6 had a median of at most 0.89 s (RQ6 criterion 5 s); without regularisation 9 of 18 models up to 100 trees were above 5 s. | 15_runtime.json, 15_runtime.csv | yes |  |
| R3 | The frozen 100-round models have CDC Diabetes 33 trees with a split (420 internal nodes); BRFSS Heart disease 31 trees with a split (391 internal nodes); Cardio-70k 20 trees with a split (220 internal nodes). | 15_runtime.csv | yes |  |

## Survey-weighted sensitivity (plan Section 6.2)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| W-brfss2015_heart-negative-outcome | BRFSS Heart disease, negative-outcome claims, frozen XGBoost: with the BRFSS survey weights, 72.0% (69.8 to 74.2) of explained test patients received a certified claim (unweighted 65.3%), and 1.6% (1.0 to 2.2) of them had the outcome (unweighted 2.3%). | 16_weighted_sensitivity.csv, 02_brfss_rebuild.json | yes | descriptive; certificates are unweighted |
| W-brfss2015_heart-enrichment | BRFSS Heart disease, enrichment claims, frozen XGBoost: with the BRFSS survey weights, 20.6% (18.7 to 22.7) of explained test patients received a certified claim (unweighted 26.0%), and 24.5% (20.5 to 28.9) of them had the outcome (unweighted 27.1%). | 16_weighted_sensitivity.csv, 02_brfss_rebuild.json | yes | descriptive; certificates are unweighted |
| W-cdc_diabetes-negative-outcome | CDC Diabetes, negative-outcome claims, frozen XGBoost: with the BRFSS survey weights, 64.5% (62.0 to 66.9) of explained test patients received a certified claim (unweighted 58.6%), and 3.9% (2.5 to 5.7) of them had the outcome (unweighted 4.0%). | 16_weighted_sensitivity.csv, 02_brfss_rebuild.json | yes | descriptive; certificates are unweighted |
| W-cdc_diabetes-enrichment | CDC Diabetes, enrichment claims, frozen XGBoost: with the BRFSS survey weights, 29.8% (27.6 to 32.2) of explained test patients received a certified claim (unweighted 35.0%), and 33.4% (29.1 to 37.8) of them had the outcome (unweighted 32.2%). | 16_weighted_sensitivity.csv, 02_brfss_rebuild.json | yes | descriptive; certificates are unweighted |

## Explanation cards (introduction)

| id | claim | sources | holds | note |
|---|---|---|---|---|
| C-brfss2015_heart-negative-outcome | BRFSS Heart disease, negative-outcome claim given to 441 explained test patients: 'HighBP < 1 and Stroke < 1 and GenHlth < 3 and Age < 5'; 6,661 calibration records, at most 0.6% have the outcome (target at most 5.0%). | 16_example_cards.csv, 09_test_releases.csv | yes |  |
| C-brfss2015_heart-enrichment | BRFSS Heart disease, enrichment claim given to 94 explained test patients: '1 <= HighBP and 1 <= HighChol and 4 <= GenHlth and 10 <= Age'; 2,950 calibration records, at least 36.6% have the outcome (target at least 30.0%). | 16_example_cards.csv, 09_test_releases.csv | yes |  |
| C-cdc_diabetes-negative-outcome | CDC Diabetes, negative-outcome claim given to 198 explained test patients: 'HighBP < 1 and Stroke < 1 and GenHlth < 2 and Age < 10'; 8,467 calibration records, at most 1.2% have the outcome (target at most 5.0%). | 16_example_cards.csv, 09_test_releases.csv | yes |  |
| C-cdc_diabetes-enrichment | CDC Diabetes, enrichment claim given to 226 explained test patients: '1 <= CholCheck and 31 <= BMI and HvyAlcoholConsump < 1 and 3 <= GenHlth and 9 <= Age'; 6,405 calibration records, at least 39.4% have the outcome (target at least 30.0%). | 16_example_cards.csv, 09_test_releases.csv | yes |  |
| C-cardio70k-negative-outcome | Cardio-70k, negative-outcome claim given to 436 explained test patients: 'age < 16059 and ap_hi < 130 and cholesterol < 2'; 1,761 calibration records, at most 15.6% have the outcome (target at most 20.0%). | 16_example_cards.csv, 09d_test_releases.csv | yes |  |
| C-cardio70k-enrichment | Cardio-70k, enrichment claim given to 712 explained test patients: '141 <= ap_hi'; 2,943 calibration records, at least 82.9% have the outcome (target at least 80.0%). | 16_example_cards.csv, 09d_test_releases.csv | yes |  |

## Gates

| id | claim | sources | holds | note |
|---|---|---|---|---|
| Gate-G0 | G0 data: passed. | 02_brfss_rebuild.json | yes |  |
| Gate-G1 | G1 model: passed. | 05_dev_configs.csv, 09c_dev_configs.csv | yes |  |
| Gate-G2 | G2 soundness: passed. | 06_encoding_tests.json, configs/frozen/*.json, 08_certificates.json, 09d_results.json, 09b_families.json, 09e_families.json, 13_shift.json, 13b_shift.json, 12b_model_variants.csv, 10_settings.csv | yes |  |
| Gate-G3 | G3 utility (development only): passed for the other tasks; Cardio-70k: cascade followed, shortfall recorded before calibration split 1 was opened (docs/PLAN.md). | configs/frozen/*.json | yes |  |
| Gate-G4 | G4 validity: passed. | 10_validity.csv, 10b_validity_more_repetitions.csv | yes |  |
| Gate-G5 | G5 runtime: passed. | 15_runtime.json | yes |  |
