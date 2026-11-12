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

## Implementation decisions recorded during the work

- 7 Nov 2026, notebook 01: in the CDC Diabetes table prediabetes is coded 0, so task B is
  diagnosed diabetes against no diabetes or prediabetes (plan v2 Sections 3 and 6.1 said
  "prediabetes or diabetes"; corrected). The heart and diabetes tables hold the same 253,680
  respondents in the same row order: they are two outcomes on one population, not two datasets.
- Stage 2, soundness: Z3 on the path-literal or real-input encoding takes about 0.1 to 1 s per query
  on 100-tree models and times out on some large boxes, too slow for the thousands of checks each
  explanation needs. Soundness is therefore decided by an exact leaf branch-and-bound over grid
  cells in integer arithmetic (encode/cell_search.py; complete, so its answer is exact), after the
  interval pre-check. Every candidate box is re-checked by a second exact implementation on real
  thresholds and fractions that shares no data structure with the first (encode/real_search.py); a
  disagreement stops the run, a budget overrun removes the box. Z3 checks a random sample of
  candidate boxes, and brute-force enumeration checks small ensembles exhaustively (notebook 06 and
  the unit tests). PySAT is not used for boosted trees, whose rational leaf values would need a
  pseudo-Boolean library.
- Stage 2, synthetic study model: XGBoost with 100 trees of depth 4, min_child_weight 50, gamma 2
  and max_bin 32, so that candidate sets for every one of the ~18,000 population cells can be
  computed; an unregularised model makes every population row its own cell and takes several
  seconds per cell.

- Notebook 05, policy certification: negative-outcome releases (target alpha_0) and enrichment
  releases (target alpha_1 = 1 - beta) are certified as two separate policies, each at delta_2 / 2,
  with their own alpha_sel; one error rate over both kinds would mix a 5% claim with a 70% claim.
  By the union bound both statements hold together with probability at least 1 - delta_2.
- Notebook 05, freezing: only training-portion rows are kept in memory; the explained subsamples of
  calibration split 2 and the test split are chosen from row indices alone. alpha_sel for each kind
  is the smallest value on a fixed grid that the development selective error would pass at the
  expected final number of accepted patients.

- Notebook 11, units: the plan's paired comparisons use 5 model seeds x 20 calibration re-draws.
  Notebook 11 uses 20 calibration re-draws of the frozen model (design below) and reuses the
  candidate sets of notebook 08. Extra model seeds need new candidate sets for every explained
  patient and are run with a smaller explained sample in notebook 12b.
- Notebook 11, re-draw design: the explained patients of calibration split 2 stay in split 2 in
  every re-draw, and split 1 is drawn at random from the other records of splits 1 and 2. This is a
  random draw of the original design (the explained patients are a uniform random subset of the
  pooled records) and keeps the policy test at the size used in notebook 09. The first run of
  notebook 11 re-partitioned all records, so its policy tests used only the explained patients that
  fell into the new split 2, about half of them; it was rerun with this design.
- Notebook 12, ablations on the frozen model, one factor at a time on the final splits: prior
  (Occam only, mixture with lambda 0.25, 0.5, 0.75, training support only), explanation construction
  (AXps only, widest inflated box of each order only, every sound box on the inflation path),
  selection rule (most succinct, largest support, tightest bound), candidates M (1, 3, 5, 10) and
  subsamples of calibration split 1 (5,000 to 40,000 records, 10 draws each). The most general
  explanation is not run (as for B3). Development results for M and lambda are those of notebook 05.
  Model factors (family, depth, trees, monotone constraints, max_bin) and extra model seeds need new
  candidate sets and are run in notebook 12b on smaller explained samples.
- Notebook 12b, model factors one at a time against the frozen XGBoost: family (LightGBM and random
  forest with the closest equivalent settings: same trees, depth and bins, min_child_weight 50 read
  as 50 records per leaf, gamma 2 as LightGBM's minimum split gain), depth 3, 5, 6, 25 and 50 trees,
  max_bin 16, 64, 256, and clinically signed monotone constraints (risk rises with HighBP, HighChol,
  BMI, Smoker, Stroke, the other disease, GenHlth, PhysHlth, DiffWalk and Age, and falls with
  PhysActivity, Income and Education; no direction imposed elsewhere). Each variant is trained like
  the frozen model (training portion, seed 42, Youden threshold on 5-fold out-of-fold scores); the
  frozen model refitted this way must reproduce the frozen hash. All variants and the frozen model as
  reference use the same 500 training patients for the prior and the same 500 explained test
  patients; certificates on the full calibration split 1. The policy test on split 2 is not run for
  variants. Model seeds are not varied: the frozen XGBoost uses no row or column sampling, and seeds
  42, 43 and 44 give the identical model.
- Notebook 02, raw BRFSS years: only odd years are usable. CDC's variable layouts show that even
  years (2016 checked; 2018 to 2022 follow the same rotation) do not ask about blood pressure,
  cholesterol or fruit and vegetables, and 2023 does not ask about fruit and vegetables. Years used:
  2015 (rebuild and gate G0), 2017, 2019, 2021, 2023. Raw names per year, checked against the
  layouts: CholCheck is _CHOLCHK (2015), _CHOLCH1 (2017), _CHOLCH2 (2019), _CHOLCH3 (2021, 2023);
  Fruits and Veggies are _FRTLT1A and _VEGLT1A from 2017; Diabetes is DIABETE4 from 2019; heavy
  drinking is _RFDRHV7 (2019, 2021) and _RFDRHV8 (2023); Sex is SEXVAR from 2019; HighBP and
  HighChol are _RFHYPE6 and TOLDHI3 from 2021; AnyHealthcare is _HLTHPLN (2021) and _HLTHPL1
  (2023); NoDocbcCost is MEDCOST1 and Income is INCOME3 (11 levels collapsed to 8) from 2021. Any
  raw code outside the expected set stops the build. The harmonised set H for the temporal shift is
  the 19 predictors without Fruits and Veggies.
- Notebook 11, equal calibration budget: every method uses the same labelled records, split 1 of the
  re-draw plus the explained patients of its split 2. SCAX releases only boxes that passed the second
  exact check; boxes that become releasable in a re-draw are checked then. The per-box-bound and
  empirical baselines use the same candidate boxes, all proven sound by the primary check.
- Notebook 11, baselines as run: B3 (most general explanation) is not run, because no maintained
  implementation exists for boosted trees (RFxpl handles random forests); the widest inflated box
  stands in. B6 is RuleFit (imodels; lasso path without cross-validation, rules only, at most 200
  rules) fitted on 20,000 training rows, every rule claimed for both classes, prior uniform over the
  dictionary. B8 runs on the synthetic population, where outcome risk is known, with confidence
  floors 1 - alpha_y. The synthetic comparison uses rule P4 only: under P1 to P3 every box that
  follows the rule has true risk equal to the noise level, so every method's claims are true there.
  B9 and B10 run per kind at the frozen alpha_sel and delta_2 / 2; B11 at epsilon = 0.05. B12 uses anchor-exp (the Anchors authors' implementation) because alibi does not
  install (its spaCy dependency fails to build), and SHAP values from the native XGBoost model
  retrained with the frozen settings, used only when its export hash equals the frozen model hash.
- Notebook 13, shift (temporal and geographic): one fixed model per source, trained like the frozen
  heart model (frozen XGBoost settings, seed 42, Youden threshold on 5-fold out-of-fold scores of the
  source training rows), with its own prior (mixture with the frozen lambda, candidate sets of 1,000
  source training patients) and its own box certificates (the source records of calibration split 1).
  Its hash is recorded when it is first fitted and checked before every target evaluation. Temporal:
  source 2015 with the 19 predictors of H, targets 2017, 2019, 2021 and 2023; the frozen splits apply
  to the rebuilt 2015 table because it equals the public table row by row. Geographic: the frozen
  splits restricted to three census regions of 2015, target the fourth (four models, all 21
  predictors). The "five largest states" are the five states with the most 2015 records (not the most
  populous), each a target of the model trained without its region, so that re-certification can be
  run up to 10,000 records; Guam and Puerto Rico have no census region and are left out. The model
  settings were chosen in development on all regions; only the fit excludes the target region. Per
  target 1,000 explained patients (500 per state and for the no-shift reference, the source test
  split); the other target records check claims. Re-certification: labelled target samples of 500 to
  10,000 records, 20 draws each, with at least 1,000 target records left to check claims (larger
  sizes are skipped for small targets). The policy test on a second split is not run in target
  domains. Site (UCI) and measurement (NHANES) shift follow in notebook 13b after notebook 03.
- Notebook 03, NHANES and UCI preparation: NHANES 2015-2016 and 2017-2018, adults aged 20 and over,
  outcome MCQ160C or MCQ160E built like _MICHD. NHANES has no days of poor physical or mental health
  (HSQ470 and HSQ480 were dropped after 2012), no "could not see a doctor because of cost" item and no
  comparable fruit and vegetable items, so the common predictor set has 16 predictors; education is
  coarsened to 5 levels and income to 7 ($35k to $75k merged, because NHANES has a $45k to $55k band),
  and the same coarsening is applied to BRFSS. HighChol is defined only for people whose cholesterol was
  ever checked, as BRFSS asks it. Each item's rule and comparability is in
  results/03_nhanes_variable_map.csv. Measured values: examination BMI; high blood pressure from the
  mean of the examination readings (systolic 140 or more, or diastolic 90 or more, diastolic 0 left out);
  total cholesterol 240 mg/dL or more. Complete cases on the 16 predictors, the outcome, measured BMI
  and measured blood pressure, so the self-reported and measured versions describe the same people.
  Survey weights are kept but not used, as for BRFSS. The files are downloaded by the notebook when
  Colab can reach CDC, otherwise by hand. UCI: the eight predictors recorded at every site (age, sex,
  chest pain type, resting blood pressure, resting ECG, maximum heart rate, exercise angina, ST
  depression); cholesterol, fasting blood sugar, slope, vessels and thalassemia are left out; a zero
  blood pressure or cholesterol is missing; complete cases.
- Notebook 13b, measurement and site shift. Measurement: one XGBoost trained like the frozen heart
  model on the BRFSS 2015 training portion with the 16 predictors NHANES shares (education and
  income coarsened as in notebook 03) and diabetes coded yes or no: NHANES asks about prediabetes as
  its own question and about 17% say yes, against about 2% in BRFSS, so prediabetes is coded as no in
  both surveys (decided after notebook 03 and before any model was fitted). Prior from 1,000 training
  patients, certificates on calibration split 1. Targets: NHANES with self-reported values, then the
  same people with measured BMI and measured high blood pressure in place of the self-reported items;
  both use the same 500 explained patients and the same records for checking claims. Re-certification
  at 500 to 5,000 records (NHANES has 6,287 adults), 20 draws, at least 500 records left to check
  claims. Site: UCI Cleveland split 50/35/15 (stratified, fixed seed per model) into training, calibration and test
  rows; XGBoost settings (100 trees, 32 bins, depth 2 to 4, minimum child weight 1 or 5) chosen on the
  Cleveland training rows only by 5-fold out-of-fold AUROC; prior from every training patient. Targets
  Hungary, Switzerland and VA, every record explained and also used to check claims. Re-certification
  is not run: each target site has fewer than 300 records.
- Gate G2, exported against native at the threshold (notebook 13b onwards): XGBoost sums leaf values in
  float32, so a row whose exact score lies within that rounding error of the threshold can be
  classified either way by the library. Such rows are counted and reported, and the exported exact
  sum decides them; a disagreement farther from the threshold than the bound 2 (T + 1) eps32 max(1, S)
  is an export error and stops the pipeline. Found while testing notebook 13b: the 16-predictor
  measurement model puts 2 calibration rows 9e-7 from its threshold (bound 2e-4); all earlier models
  had no disagreements.
- Gate G2, XGBoost input conversion (found while testing notebook 13b): XGBoost converts inputs to
  float32 before comparing them with its float32 thresholds, while the export compared float64 inputs.
  The two agree on every table used so far (BRFSS, CDC Diabetes, NHANES and the synthetic population
  are integer-coded, and float32 represents those values exactly), so no committed result changes and
  every frozen model keeps its hash. With decimal inputs they can differ: in testing on UCI, where ST
  depression has one decimal, records with ST depression exactly at a threshold went the other way.
  XGBoost models on such data are now exported with input_cast float32 (XGBoost's own conversion),
  and a float64 export refuses any input whose cell would change under float32 conversion, so the
  difference cannot pass silently.
- UCI Heart main result (E1): the plan's leave-one-site-out design, in notebook 13b. For each site, a
  model built the same way on the other three sites (split 50/35/15) is evaluated on the held-out site.
  UCI uses the frozen heart targets (alpha_0 = 0.05, alpha_1 = 0.70) and the most lenient grid point
  (alpha_0 = 0.10, beta = 0.2, so alpha_1 = 0.80). The policy test is not run: there is no second
  calibration split at this size.
- E1 scope still open (recorded 11 November 2026): the plan asks for main results on four tasks (BRFSS
  Heart, CDC Diabetes, Cardio-70k, UCI Heart) with three model families. Done: BRFSS Heart and CDC
  Diabetes with XGBoost (notebook 09) and UCI Heart (notebook 13b); LightGBM and random forest only as
  ablations (notebook 12b, without the policy test). The LightGBM and random forest main rows with the
  policy test are notebook 09b; Cardio-70k (development, freezing, certificates, release) is notebook
  09c.
- Notebook 09b, other model families (E1) on the frozen design of each task: same training portion and
  seed, the same 2,000 prior training patients and lambda, the same M, feature-order seed and search
  budget, the same explained subsamples of calibration split 2 and of the test split, calibration
  split 1 for certificates, the frozen target grids and alpha_sel. Settings closest to the frozen
  XGBoost, as in notebook 12b: LightGBM with 100 trees of depth 4 (15 leaves), 32 bins, learning rate
  0.3, at least 50 records per leaf and minimum split gain 2; random forest with the plan's 50 trees
  of depth 4 and at least 50 records per leaf. Each model is fitted on the training portion with a
  Youden threshold on 5-fold out-of-fold scores, saved and hashed before calibration split 1 is read.
  Prior boxes are not re-verified (release still requires the second exact check, as in notebooks 12b
  and 13). Results use the layout of notebook 09, with the XGBoost rows taken from results/09_main.csv.

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
| 09b | Main results for LightGBM and random forest on the frozen design, with the policy test |
| 09c | Main results for Cardio-70k: development, freezing, certificates, release |
| 10 | Validity study on the synthetic population |
| 11 | Baselines |
| 12 | Ablations |
| 13 | Shift, BRFSS: temporal and geographic; transport, re-certification sample sizes, within-box shift |
| 13b | Shift: measurement (NHANES) and site (UCI); UCI Heart main result, leave-one-site-out |
| 14 | Subgroups and group-conditional certificates |
| 15 | Runtime |
| 16 | Paper results and claims check |

## Conventions

- Drive root `/content/drive/MyDrive/SCAX_Research/`; datasets in `datasets/` outside the repository;
  `commit_console.ipynb` at the Drive root.
- Every notebook starts with the session bootstrap cell and ends with the standard commit cell;
  `tools/commit_cell.py` is the only commit mechanism.
- Seeds in every results file; figures at 600 dpi with paper names; notebooks tested before delivery.
