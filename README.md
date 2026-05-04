# ML Anti-Cancer Activity Prediction

Pan-drug anticancer drug sensitivity prediction with multi-source, multi-omics, mechanism-aware machine learning.

This repository contains a research pipeline for predicting cancer cell-line drug response across many anticancer compounds rather than training one model per drug. The current validated model is an identity-free XGBoost tabular model trained on harmonised GDSC, CTRPv2, and CCLE response data with DepMap cell-line features and transferable drug descriptors. A PyTorch two-tower interaction model is included as an experimental scaffold for PRISM-scale drug-cell interaction learning.

The key result is not just random-split accuracy. The project now evaluates pan-drug generalisation under repeated unseen-drug, unseen-scaffold, unseen-target, unseen-pathway, leave-cell-line-out, leave-drug-out, and leave-both-out regimes.

## What This Is

This is a multi-modal tabular machine learning framework for anticancer drug response prediction.

It integrates:

- Drug response screens: GDSC, CTRPv2, CCLE, and PRISM experiments for neural/scaffold experiments.
- Cell-line transcriptomics: DepMap Log2(TPM + 1) gene expression.
- Cell-line multi-omics: somatic mutation, copy-number, fusion, RPPA/protein abundance, and tissue annotations.
- Drug chemistry: Morgan fingerprints and RDKit molecular descriptors.
- Drug mechanism: target, pathway, and mechanism multi-hot annotations.
- Explainability: true XGBoost TreeSHAP prediction contributions.

The model predicts a harmonised response score where higher values correspond to greater resistance and lower values correspond to greater sensitivity.

## Final Validation Summary

The final pan-drug stress test used the multi-source multi-omics identity-free XGBoost model across 20 full runs.

| Generalisation regime | Runs | Mean R2 | SD | Mean Pearson | Mean Spearman | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Repeated leave-drug-out | 5 | 0.233 | 0.069 | 0.489 | 0.494 | Supports unseen-drug transfer |
| Scaffold holdout | 5 | 0.229 | 0.029 | 0.487 | 0.526 | Supports unseen chemical-scaffold transfer |
| Target holdout | 5 | 0.180 | 0.040 | 0.468 | 0.465 | Supports unseen target-group transfer |
| Pathway holdout | 5 | -0.002 | 0.052 | 0.204 | 0.187 | Weak transfer to unseen pathway classes |

The strongest honest conclusion is:

> The model demonstrates meaningful pan-drug generalisation to unseen drugs, unseen chemical scaffolds, and unseen target groups, while pathway-level generalisation remains weak.

See the full results, figures, and linked CSV outputs in [docs/RESULTS.md](docs/RESULTS.md).

## Key Figures

![Pan-drug generalisation stress tests](analysis_results/manuscript_figures/fig7_pan_drug_generalization_stress.png)

![True TreeSHAP modality importance](analysis_results/manuscript_figures/fig3_shap_modality.png)

## Repository Structure

```text
.
|-- app.py                              # Gradio research interface
|-- pipeline.py                         # Original GDSC/DepMap integration pipeline
|-- benchmarking.py                     # Strict splits, ablations, TreeSHAP, baselines
|-- external_validation.py              # CTRPv2/CCLE endpoint harmonisation and validation
|-- multisource_inductive_training.py   # Final multi-source XGBoost training
|-- pan_drug_generalization.py          # Repeated drug/scaffold/target/pathway stress tests
|-- drug_cell_interaction_model.py      # PyTorch two-tower interaction model
|-- lazy_interaction_training.py        # Lazy indexed PRISM-scale interaction training
|-- generate_manuscript_figures.py      # Rebuild manuscript figures from result CSVs
|-- docs/RESULTS.md                     # Detailed result summary with CSV links
`-- analysis_results/                   # CSV summaries and manuscript figures
```

Raw datasets are intentionally excluded from Git and should be placed under `data/` and `data/new/`.

## How We Built It

The final modelling workflow has five main stages.

1. Harmonise drug response datasets.

GDSC is used as the primary response dataset. CTRPv2 and CCLE are transformed onto a GDSC-like harmonised response scale and added to training. PRISM is used for separate large-scale neural interaction experiments because its interaction matrix is much larger.

2. Build transferable cell and drug features.

Cell-line features include expression, mutation, copy-number, fusion, RPPA/protein abundance, and tissue annotations. Drug features include target/pathway annotations, mechanism tokens, Morgan fingerprints, RDKit descriptors, and a structure-missing flag. Drug identity is excluded from the final model to reduce memorisation.

3. Train the primary model.

The main model is XGBoost with CUDA acceleration. Hyperparameters were selected under leave-drug-out validation, not random splits, so tuning aligns with unseen-drug generalisation.

4. Stress-test pan-drug generalisation.

The final model is evaluated with repeated leave-drug-out, scaffold holdout, target holdout, and pathway holdout splits. This tests whether performance survives different definitions of unseen drug novelty.

5. Interpret with true TreeSHAP.

TreeSHAP values are computed from XGBoost native prediction contributions. Global interpretation uses mean absolute SHAP values by feature and modality.

## Installation

Create and activate a Python environment, then install dependencies.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU runs require a CUDA-compatible NVIDIA driver plus compatible XGBoost/PyTorch builds.

## Data Layout

Expected local layout:

```text
data/
├── GDSC1_fitted_dose_response_27Oct23.xlsx
├── DepMap/
│   ├── Model.csv
│   ├── OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
│   ├── OmicsSomaticMutations.csv
│   ├── PortalOmicsCNGeneLog2.csv
│   ├── OmicsFusionFiltered.csv
│   └── CCLE_RPPA_20180123.csv
└── new/
    ├── CTRPv2 files
    ├── CCLE files
    └── PRISM/
```

The repository does not include raw datasets because they are large and often have source-specific redistribution terms.

## Reproducing The Main Runs

Run the final multi-source multi-omics XGBoost training:

```powershell
.\venv\Scripts\python.exe multisource_inductive_training.py `
  --output-dir analysis_results\multisource_multiomics_full `
  --include-external ctrp,ccle `
  --split leave_drug_out `
  --include-omics `
  --tune-xgboost `
  --source-weighting sqrt_inverse_dataset `
  --device cuda
```

Run the pan-drug generalisation stress test:

```powershell
.\venv\Scripts\python.exe pan_drug_generalization.py `
  --output-dir analysis_results\pan_drug_generalization_full `
  --include-external ctrp,ccle `
  --min-samples-per-drug 100 `
  --regimes all `
  --repeats 5 `
  --bootstrap 200 `
  --include-omics `
  --device cuda `
  --resume
```

Generate manuscript figures:

```powershell
.\venv\Scripts\python.exe generate_manuscript_figures.py
```

Run the research interface:

```powershell
.\venv\Scripts\python.exe app.py
```

## Result Artifacts

Important committed result artifacts:

- [analysis_results/pan_drug_generalization_full/aggregate.csv](analysis_results/pan_drug_generalization_full/aggregate.csv)
- [analysis_results/pan_drug_generalization_full/runs.csv](analysis_results/pan_drug_generalization_full/runs.csv)
- [analysis_results/pan_drug_generalization_full/per_dataset.csv](analysis_results/pan_drug_generalization_full/per_dataset.csv)
- [analysis_results/pan_drug_generalization_full/summary.json](analysis_results/pan_drug_generalization_full/summary.json)
- [analysis_results/manuscript_figures](analysis_results/manuscript_figures)

Detailed interpretation is maintained in [docs/RESULTS.md](docs/RESULTS.md).

## Research Use Only

This project is for research and hypothesis generation. It is not a clinical decision system and should not be used to select patient therapy without prospective validation in patient-derived models or clinical cohorts.

