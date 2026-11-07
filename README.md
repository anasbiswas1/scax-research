
# SCAX: Statistically Certified Abductive Explanations for Disease Classification

SCAX is a certify-or-abstain framework for disease classification on public health data. A tree
ensemble is trained; a logic solver proves, for each patient, explanations (boxes of feature ranges)
inside which the model's prediction cannot change; each box receives a statistical certificate on how
often the recorded outcome disagrees with the prediction among people inside the box; and a release
policy is certified on held-out data before deployment.

What SCAX certifies:

1. Each released explanation is sound for the deployed classifier (exact, by proof).
2. Each released explanation's claim about the recorded outcome holds, for all boxes at once, with
   probability at least 1 - delta_1.
3. If the release policy is certified, its error rate among released patients is at most alpha_sel,
   with probability at least 1 - delta_2.

SCAX differs from the authors' FG-LTT project (faithfulness-gated selective ECG diagnosis with a
Learn-then-Test guarantee) and XIDS-Certify (adversarial robustness certificates for intrusion
detection): it certifies the recorded-outcome risk of formally sound explanation regions on
population-health tables.

## Layout

```
notebooks/   numbered notebooks, 00 to 16 (docs/PLAN.md has the map)
src/scax/    package: data, models, encode, explain, certify, baselines, metrics, plots
tests/       unit tests run inside the notebooks that need them
tools/       commit_cell.py, the single commit step
docs/        PLAN.md, the fixed plan
configs/     frozen configurations with hashes
results/     small CSV and JSON outputs and figures
```

Data live on Google Drive under `SCAX_Research/datasets/`, outside the repository. Raw CDC BRFSS
files are not redistributed; Kaggle and UCI terms apply to the public tables.

## Authors

Md Anas Biswas and Alif Laila, School of Computing, University of Portsmouth.
