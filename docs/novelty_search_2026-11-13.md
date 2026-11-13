# Novelty search, 13 November 2026

Plan v2.1, Section 9.1 step 9 and Section 1.11: the novelty search is repeated on Scopus and Web of
Science before writing. This file records what was searched, what was found, and what changes in
the related work.

## Question

Does any published or preprint work certify the recorded-outcome rate inside formally sound
explanation regions (boxes inside which the model's prediction cannot change), with a correction for
selecting among many candidate explanations, and release a prediction only with such a certified
explanation?

## Searches

| Source | Date | Years | Records | Screened |
|---|---|---|---|---|
| Open web search (arXiv, proceedings, publisher pages, GitHub) | 13 Nov 2026 | any | about 15 queries | titles and abstracts of every relevant hit |
| Scopus, documents | 13 Nov 2026 | 2022 to 2026 | 84 | every title; full abstract of the 24 records on explanations, guarantees or abstention; keyword scan of all abstracts |
| Scopus, preprints (arXiv, Research Square, SSRN, ChemRxiv, medRxiv) | 13 Nov 2026 | 2022 to 2026 | 69 | every title and abstract |
| Web of Science Core Collection | 13 Nov 2026 | 2022 to 2026 | 50 (46 also in Scopus) | every title; abstracts of the 4 new records |

Scopus query (Advanced search):

```
( TITLE-ABS-KEY ( "abductive explanation*" AND ( guarantee* OR certif* OR conformal OR calibrat* OR "confidence bound" OR "Clopper-Pearson" ) )
OR TITLE-ABS-KEY ( ( "formal explanation*" OR "logic-based explanation*" OR "sufficient reason*" OR "prime implicant*" ) AND ( "tree ensemble*" OR "random forest*" OR "boosted tree*" OR xgboost ) AND ( probab* OR confidence OR risk OR calibrat* ) )
OR TITLE-ABS-KEY ( explanation* AND ( abstain* OR abstention OR "reject option" OR "selective prediction" OR "selective classification" ) AND ( guarantee* OR certif* OR "finite-sample" ) )
OR TITLE-ABS-KEY ( ( rule* OR box* OR hyperrectangle* OR anchor* ) AND explanation* AND ( "Clopper-Pearson" OR pac OR "finite-sample" OR "union bound" OR conformal ) )
OR TITLE-ABS-KEY ( "certified explanation*" OR "explanation-conditional" OR "conditional on the explanation" OR "statistically certified" )
OR TITLE-ABS-KEY ( ( "formal explanation*" OR "abductive explanation*" OR "certified explanation*" ) AND ( disease* OR clinical OR medical OR patient* ) ) )
AND PUBYEAR > 2021
```

Web of Science query (Advanced search, Query Builder): the same six blocks as `TS=( ... )` joined by
`OR`, followed by `AND PY=(2022-2026)`.

## Result

No record certifies the recorded-outcome rate of formally sound explanation regions with selection
validity, and none releases predictions only together with such certified explanations. The novelty
statement of the plan (Section 1.11) stands.

## Nearest neighbours, to be set apart in the gap table

- Bressan, M., Brukhim, N., Cesa-Bianchi, N., Esposito, E., Mansour, Y., Moran, S. and Thiessen, M.
  (2026). Learning Conditional Averages. arXiv:2602.11920. PAC learning of the average label over a
  neighbourhood containing each point; explainability is one motivation. A learning problem with
  sample-complexity bounds, not a per-patient certificate for a model-sound region.
- Cheng, N., Spector, A. and Janson, L. (2025). Chiseling: Powerful and Valid Subgroup Selection via
  Interactive Machine Learning. arXiv:2509.19490. Valid selection of one subgroup whose average
  outcome exceeds a threshold. Not tied to a model's predictions; one subgroup per analysis.
- Frost, N., Lipton, Z. C., Mansour, Y. and Moshkovitz, M. (2024). Partially Interpretable Models with
  Guarantees on Coverage and Accuracy. PMLR 237:590-613. A decision list decides or abstains
  to a black box; guarantees relative to the best list, not statistical bounds on outcome rates from
  held-out data.
- Lafifi, S. (2026). Constraint-Anchored Attribution: Feasibility-Certified Counterfactuals and
  Bonferroni-PAC Sufficient Subsets for Neural CO Policies. arXiv:2605.25235 (Scopus lists the authors
  as Lafifi, S. and Afifi, S.). A Bonferroni-corrected Hoeffding test for sufficient subsets; the claim
  is agreement with a neural combinatorial-optimisation policy, not a recorded outcome.
- Dhurandhar, A., Haldar, S., Wei, D. and Ramamurthy, K. N. (2024). Trust Regions for Explanations via
  Black-Box Probabilistic Certification. arXiv:2402.11168. Certifies a hypercube in which an
  explanation stays faithful to the model, from model queries, not outcome labels.

## Further citations

- Audemard, G., Lagniez, J.-M., Marquis, P. and Szczepanski, N. (2024). On the Computation of
  Example-Based Abductive Explanations for Random Forests. IJCAI 2024, 3679-3687. Sound explanations
  that cover many reference examples; coverage, no outcome bound.
- Cooper, M. and Amgoud, L. (2023). Abductive Explanations of Classifiers Under Constraints:
  Complexity and Properties. ECAI 2023, FAIA 372:469-476. DOI 10.3233/FAIA230305.
- Koriche, F., Lagniez, J.-M., Mengel, S. and Tran, C. (2024). Learning Model Agnostic Explanations via
  Constraint Programming. ECML PKDD 2024, LNCS 14944:437-453. DOI 10.1007/978-3-031-70359-1_26.
  PAC-style guarantee of agreement with the black box.
- Koriche, F., Lagniez, J.-M. and Tran, C. (2026). Probabilistic Linear Explanations. arXiv:2609.19077.
- Izza, Y., Meel, K. S. and Marques-Silva, J. (2024). Locally-Minimal Probabilistic Explanations.
  ECAI 2024, FAIA 392:1092-1099. DOI 10.3233/FAIA240601.
- Bounia, L. and Koriche, F. (2023). Approximating Probabilistic Explanations via Supermodular
  Minimization. UAI 2023, PMLR 216:216-225.
- Bounia, L. (2026). k-Pseudo-Probable Path Explanations for Weighted Tree-Based Ensembles. ECML PKDD
  2026.
- Murtovi, A., Schlüter, M. and Steffen, B. (2025). Computing Inflated Explanations for Boosted Trees:
  A Compilation-Based Approach. LNCS 15240:183-201. DOI 10.1007/978-3-031-73887-6_14. The first
  algorithm for inflated explanations of gradient boosted trees (relevant to Algorithm 3 and baseline B2).
- Rocha, F. M., Rocha, T. A., Ribeiro, R. P. F. and Rocha, A. R. (2023). Logic-Based Explanations for
  Linear Support Vector Classifiers with Reject Option. BRACIS 2023, LNCS 14195:144-159.
  DOI 10.1007/978-3-031-45368-7_10.
- Rigorous Explanations for Tree Ensembles (2026). arXiv:2603.29361. Logical soundness only.
- Shakhovska, N., Izonin, I. and Mitoulis, S.-A. (2026). Prediction certification cannot replace
  explanation certification: a competence envelope for trustworthy AI under compound stress.
  arXiv:2608.20825.
- Medical uses of formal explanations, without outcome guarantees: Chagas, V. P., Viana, L. H. T.,
  Carlos, M. M. da S., Madeiro, J. P. V., Pedrosa, R. C., Rocha, T. A. and Cavalcante, C. H. L. (2026).
  Reliable XAI Explanations in Sudden Cardiac Death Prediction for Chagas Cardiomyopathy.
  arXiv:2602.22288. Lens, C., Majed, B., Marquis, P., Tabia, K. and Wallon, R. (2027). Predicting
  Critical Deterioration of Patients in Emergency Units Using Administrative Health Data. LNCS
  16749:161-166. DOI 10.1007/978-3-032-30813-9_30.

## Datasets and limitations

- Tareaf, R. B., Al-Rajab, M., Loucif, S., Ellaham, S. and Schmitz, C. (2026). Target leakage, not
  model class, explains reported accuracy in survey-based cardiovascular screening: a leakage-tiered
  audit of glass-box and tabular foundation models. arXiv:2609.11838. On BRFSS 2022, two
  post-diagnostic features raise AUROC by about 0.05; frozen models transport to 2023; marginal
  conformal coverage is lower for adults over 60 and Mondrian calibration repairs it. The BRFSS 2015
  heart table used here has none of those post-diagnostic items (21 features: HighBP, HighChol,
  CholCheck, BMI, Smoker, Stroke, Diabetes, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
  AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income).

## Off-topic records

The remaining Scopus and Web of Science records concern other fields (battery and energy forecasting,
astronomy, physics, sport, process monitoring, LLM verification, attribution methods with conformal
intervals) or are already in the plan's review (Chagas et al. 2026 on confidence, MINABRO, Bridging AI
and Clinical Reasoning, Learning Conformal Explainers, Calibrated Explanations, Conformal Decision
Rules, CONFIDERAI, Most General Explanations of Tree Ensembles).
