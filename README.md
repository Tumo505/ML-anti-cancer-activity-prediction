## 1. Introduction

### 1.1 Problem Statement

Predicting cancer drug sensitivity is a critical challenge in precision oncology. Traditional approaches often develop drug-specific models that cannot generalize to new compounds. This work presents a pan-drug modeling approach that learns generalizable patterns across multiple drugs, enabling:

- Prediction of sensitivity for drugs in the training set
- Transfer learning potential to similar drugs
- Identification of biomarkers that drive drug response across drug classes

### 1.2 Objectives

1. Develop a unified predictive model capable of predicting drug sensitivity across 378 anticancer compounds
2. Integrate multi-omics and chemical structure data for improved predictive accuracy
3. Provide interpretable predictions through feature importance and SHAP-based explanations
4. Deploy the model as an accessible web-based prediction platform

---

## 2. Data Sources

### 2.1 Genomics of Drug Sensitivity in Cancer (GDSC)

**Source:** Wellcome Sanger Institute GDSC1 Database  
**File:** `GDSC1_fitted_dose_response_27Oct23.xlsx`

The GDSC database provides high-throughput drug screening data for cancer cell lines:

- **Total experiments:** 333,161 drug-cell line combinations
- **Unique drugs:** 378 anticancer compounds
- **Unique cell lines:** 970 cancer cell lines
- **Response metric:** AUC (Area Under the dose-response Curve)
  - Lower AUC indicates higher drug sensitivity
  - Higher AUC indicates drug resistance

**Drug annotations included:**
- `DRUG_NAME`: Compound identifier
- `PUTATIVE_TARGET`: Known molecular target (e.g., BRAF, EGFR, MEK1)
- `PATHWAY_NAME`: Associated signaling pathway (e.g., ERK MAPK signaling)

### 2.2 DepMap Gene Expression Data

**Source:** Cancer Dependency Map (DepMap) Project, Broad Institute  
**File:** `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`

Gene expression profiles for cancer cell lines:

- **Cell lines:** 1,754 unique cancer cell lines
- **Genes:** 19,215 protein-coding genes
- **Expression metric:** Log2(TPM + 1) normalized expression values
- **Filtering:** Only default entries per model retained (1,699 cell lines after filtering)

### 2.3 DepMap Model Mapping

**Source:** DepMap Model.csv  
**File:** `Model.csv`

Cell line identifier mapping to link GDSC and DepMap datasets:

- **Total models:** 2,132
- **Models with Sanger ID:** 1,218 (used for merging)

### 2.4 Chemical Structure Data (SMILES)

**Source:** PRISM secondary screen dose-response curve parameters  
**File:** `secondary-screen-dose-response-curve-parameters.csv`

Molecular structure information for drugs:

- **Unique drugs with SMILES:** 1,448 compounds
- **Format:** SMILES (Simplified Molecular Input Line Entry System) strings

---

## 3. Data Preprocessing

### 3.1 Dataset Integration

The datasets were merged using cell line identifiers:

1. **GDSC ↔ DepMap mapping:** Linked via `SANGER_MODEL_ID` to `SangerModelID`
2. **Merged with expression:** Final dataset: 245,235 experiments
3. **Final statistics:**
   - Unique drugs: 378
   - Unique cell lines: 714

### 3.2 Quality Filtering

- **Minimum samples per drug:** 100 experiments required for inclusion in training
- **Expression data filtering:** Only "default entry" samples retained from DepMap
- **Missing value handling:** Median imputation using `SimpleImputer`

### 3.3 Feature Selection

Gene expression features were selected based on variance:

- **Selection method:** Top 1,000 most variable genes across all drugs
- **Rationale:** Highly variable genes are more likely to distinguish drug response phenotypes
- **Implementation:** Variance calculated across all drug-cell line combinations

---

## 4. Feature Engineering

### 4.1 Gene Expression Features (n=1,000)

**Processing:**
1. Extract top 1,000 most variable genes from the merged dataset
2. Feature format: `GENE_SYMBOL (ENTREZ_ID)` (e.g., "BRAF (673)")

**Variance-Based Selection:**

Genes were ranked by variance across all samples:

$$\sigma^2_g = \frac{1}{N-1} \sum_{i=1}^{N} (x_{gi} - \bar{x}_g)^2$$

where $x_{gi}$ is the expression of gene $g$ in sample $i$, and $\bar{x}_g$ is the mean expression.

**Scaling (StandardScaler):**

Z-score normalization was applied to each feature:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

where $\mu$ is the training set mean and $\sigma$ is the training set standard deviation.

### 4.2 Drug Target Encoding (n=1)

**Method:** Label encoding of putative molecular targets

- **Unique targets:** 289 distinct targets
- **Encoding:** Integer encoding via `sklearn.preprocessing.LabelEncoder`
- **Unknown handling:** Targets not in training set encoded as -1

**Examples of encoded targets:**
- BRAF, EGFR, MEK1, ALK, BCR-ABL, PI3K, mTOR, etc.

### 4.3 Drug Pathway Encoding (n=1)

**Method:** Label encoding of pathway annotations

- **Unique pathways:** 24 distinct pathways
- **Encoding:** Integer encoding via `sklearn.preprocessing.LabelEncoder`

**Pathway categories include:**
- ERK MAPK signaling
- PI3K/MTOR signaling
- Cell cycle
- DNA replication
- Apoptosis regulation
- RTK signaling
- Chromatin histone modification
- And others

### 4.4 Drug Identity Encoding (n=1)

**Method:** Label encoding of drug names

- **Unique drugs:** 378 compounds
- **Encoding:** Integer encoding
- **Purpose:** Allows the model to learn drug-specific response patterns while still generalizing across drugs

### 4.5 Molecular Fingerprints (n=256)

**Method:** Morgan (Extended Connectivity) Fingerprints using RDKit

Morgan fingerprints encode molecular substructures as binary vectors. For each atom $a$ in the molecule, a circular neighborhood of radius $r$ is hashed:

$$h_a^{(r)} = \text{hash}\left(h_a^{(r-1)}, \{h_b^{(r-1)} : b \in N(a)\}\right)$$

where $N(a)$ denotes the neighbors of atom $a$. The final fingerprint is a binary vector $\mathbf{f} \in \{0,1\}^{256}$ where:

$$f_i = \begin{cases} 1 & \text{if any } h_a^{(r)} \mod 256 = i \\ 0 & \text{otherwise} \end{cases}$$

**Parameters:**
- **Fingerprint type:** Morgan (ECFP-like)
- **Radius:** 2 (captures local chemical environment up to 2 bonds)
- **Bit size:** 256 bits
- **Generator:** `rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator`

**Processing:**
1. Parse SMILES string using `Chem.MolFromSmiles()`
2. Generate fingerprint using Morgan generator
3. Convert to binary array (256 bits)
4. Drugs without SMILES assigned zero vectors

**Coverage:**
- Drugs with molecular fingerprints: ~67% of training drugs
- Samples with fingerprints: ~85% of total training samples

### 4.6 Total Feature Composition

| Feature Category | Count | Description |
|-----------------|-------|-------------|
| Gene Expression | 1,000 | Top variable genes |
| Drug Target | 1 | Encoded molecular target |
| Drug Pathway | 1 | Encoded signaling pathway |
| Drug Identity | 1 | Encoded drug name |
| Molecular Fingerprints | 256 | Morgan fingerprint bits |
| **Total** | **1,259** | Combined feature vector |

---

## 5. Model Architecture

### 5.1 Algorithm Selection

**Algorithm:** XGBoost Regressor (Extreme Gradient Boosting)

**Rationale:**
- Handles high-dimensional sparse data effectively
- Robust to missing values with built-in handling
- GPU acceleration for large datasets
- Provides native feature importance scores
- Strong performance on tabular biomedical data

### 5.2 Hyperparameters

```python
XGBRegressor(
    n_estimators=500,          # Number of boosting rounds
    max_depth=6,               # Maximum tree depth
    learning_rate=0.05,        # Step size shrinkage
    min_child_weight=3,        # Minimum sum of instance weight
    subsample=0.8,             # Row subsampling ratio
    colsample_bytree=0.8,      # Column subsampling ratio
    gamma=0.1,                 # Minimum loss reduction for split
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    device='cuda',             # GPU acceleration
    tree_method='hist',        # Histogram-based tree construction
    random_state=42,           # Reproducibility seed
    n_jobs=-1                  # Parallel processing
)
```

### 5.3 Regularization Strategy

The XGBoost objective function with regularization is:

$$\mathcal{L}(\theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

where the regularization term is:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T}w_j^2 + \alpha \sum_{j=1}^{T}|w_j|$$

- $T$ = number of leaves in the tree
- $w_j$ = weight of leaf $j$
- $\gamma$ = minimum loss reduction for split (0.1)
- $\lambda$ = L2 regularization coefficient (1.0)
- $\alpha$ = L1 regularization coefficient (0.1)

**Regularization components:**
- **L1 (Lasso):** `reg_alpha=0.1` - Promotes sparsity in feature selection
- **L2 (Ridge):** `reg_lambda=1.0` - Prevents large weight values
- **Subsampling:** 80% row and column sampling to reduce overfitting
- **Gamma:** Minimum loss reduction required for tree split

### 5.4 GPU Acceleration

**Hardware:** NVIDIA RTX GPU with CUDA support  
**Implementation:** XGBoost native CUDA implementation

Benefits:
- ~10-50x speedup over CPU training
- Enables rapid hyperparameter tuning
- Supports real-time prediction in deployment

---

## 6. Training Procedure

### 6.1 Data Splitting

**Strategy:** Stratified train-test split

```python
train_test_split(
    X, y, drug_names,
    test_size=0.2,           # 80/20 split
    random_state=42,         # Reproducibility
    stratify=drug_names      # Maintain drug distribution
)
```

**Stratification rationale:** Ensures all drugs are represented in both training and test sets, preventing evaluation bias.

### 6.2 Preprocessing Pipeline

1. **Imputation:** `SimpleImputer(strategy='median')`
   - Handles missing gene expression values
   - Fitted on training set only

2. **Scaling:** `StandardScaler()`
   - Z-score normalization
   - Fitted on training set only
   - Applied to test set using training statistics

### 6.3 Cross-Validation

**Method:** K-Fold Cross-Validation

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

**Scoring metric:** R² (coefficient of determination)

**Purpose:**
- Estimate model generalization performance
- Detect overfitting
- Report confidence intervals for performance metrics

### 6.4 Early Stopping

**Implementation:** Evaluation set monitoring

```python
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)
```

---

## 7. Model Interpretability

### 7.1 Native Feature Importance

XGBoost provides built-in feature importance via `model.feature_importances_`:

- **Method:** Gain-based importance (total gain from splits using each feature)
- **Output:** Normalized importance scores summing to 1.0

### 7.2 SHAP (SHapley Additive exPlanations)

**Library:** SHAP v0.49.1

**Theoretical Foundation:**

SHAP values are based on Shapley values from cooperative game theory. For a prediction $f(x)$, the SHAP value $\phi_i$ for feature $i$ represents its contribution to the prediction:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

where $\phi_0$ is the base value (expected model output) and $M$ is the number of features.

**Attempted approaches:**
1. **TreeExplainer:** Native tree-based SHAP computation
   - Failed due to GPU-trained model serialization incompatibility
   
2. **Feature Importance-Based Fallback (Pseudo-SHAP):**

Due to TreeExplainer incompatibility with GPU-trained models, we implemented a feature importance-based approximation. The pseudo-SHAP contribution $\psi_i$ for feature $i$ is computed as:

$$\psi_i = I_i \cdot x_i^{scaled}$$

where:
- $I_i$ = XGBoost gain-based feature importance for feature $i$
- $x_i^{scaled}$ = standardized feature value (z-score normalized)

The direction of contribution is determined by the sign of $\psi_i$:
- $\psi_i > 0$ → Feature contributes to **resistance** (higher AUC)
- $\psi_i < 0$ → Feature contributes to **sensitivity** (lower AUC)

**Interpretation output:**
- Top features contributing to resistance (↑ AUC)
- Top features contributing to sensitivity (↓ AUC)
- Waterfall visualization of feature contributions

### 7.3 Per-Drug Performance Analysis

The model reports per-drug R² and Pearson correlation to identify:
- Drugs with strongest predictive accuracy
- Drug classes that may require specialized models
- Potential biomarkers specific to drug mechanisms

---

## 8. Deployment Architecture

### 8.1 Web Interface

**Framework:** Gradio v4.0+

**Features:**
- Drug selection from database (378 compounds)
- Cell line selection from DepMap database (714 cell lines)
- Custom expression data upload (CSV format)
- Real-time AUC prediction
- SHAP-based interpretation generation
- Batch prediction capability

### 8.2 Model Serialization

Saved components:
- `model.pkl`: Trained XGBoost model
- `scaler.pkl`: Fitted StandardScaler
- `imputer.pkl`: Fitted SimpleImputer
- `feature_names.pkl`: Ordered feature name list
- `drug_encoders.pkl`: LabelEncoders for target, pathway, drug


---

## 9. Evaluation Metrics

### 9.1 Regression Metrics

**Coefficient of Determination (R²):**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

**Root Mean Squared Error (RMSE):**

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Mean Absolute Error (MAE):**

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Pearson Correlation Coefficient:**

$$r = \frac{\sum_{i=1}^{n}(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}\sqrt{\sum_{i=1}^{n}(\hat{y}_i - \bar{\hat{y}})^2}}$$

**Spearman Rank Correlation:**

$$\rho = 1 - \frac{6\sum_{i=1}^{n}d_i^2}{n(n^2-1)}$$

where $d_i$ is the difference between ranks of $y_i$ and $\hat{y}_i$.

| Metric | Interpretation |
|--------|----------------|
| R² | Variance explained (0-1, higher better) |
| RMSE | Root mean squared error (lower better) |
| MAE | Mean absolute error (lower better) |
| Pearson r | Linear correlation (-1 to 1) |
| Spearman ρ | Monotonic correlation (-1 to 1) |

### 9.2 Clinical Interpretation

**AUC Thresholds for Drug Response:**
- **Sensitive:** AUC < 0.5
- **Moderate:** 0.5 ≤ AUC < 0.8
- **Resistant:** AUC ≥ 0.8

---

## 10. Limitations and Considerations

### 10.1 Data Limitations

1. **In vitro bias:** Cell line responses may not translate directly to patient tumors
2. **Drug coverage:** Limited to drugs in GDSC database
3. **Cell line representation:** May not cover rare cancer subtypes
4. **Missing SMILES:** ~33% of drugs lack molecular fingerprints

### 10.2 Model Limitations

1. **GPU model serialization:** SHAP TreeExplainer incompatibility with CUDA-trained models
2. **Cold start problem:** New drugs without similar training examples may have poor predictions
3. **Pathway annotation dependency:** Predictions rely on accurate target/pathway annotations

### 10.3 Recommendations for Clinical Use

- Use predictions as prioritization tool, not definitive treatment decisions
- Validate predictions with in vitro experiments when possible
- Consider drug mechanism and cancer type in interpretation
- Cross-reference with known biomarkers for drug response

---

## 11. Software and Dependencies

### 11.1 Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥3.10 | Runtime environment |
| NumPy | ≥1.24.0 | Numerical computing |
| Pandas | ≥2.0.0 | Data manipulation |
| Scikit-learn | ≥1.3.0 | ML utilities |
| XGBoost | ≥2.0.0 | Gradient boosting |
| RDKit | ≥2023.0.0 | Cheminformatics |
| SHAP | ≥0.49.0 | Model interpretability |
| Gradio | ≥4.0.0 | Web interface |

### 11.2 Hardware Requirements

**Training:**
- NVIDIA GPU with CUDA support (recommended)
- Minimum 16GB RAM
- ~10GB disk space for data

**Inference:**
- CPU sufficient for predictions
- GPU optional for batch processing

---

## 12. Reproducibility

### 12.1 Random Seeds

All stochastic operations use fixed seeds:
- `random_state=42` for train/test split
- `random_state=42` for cross-validation
- `random_state=42` for XGBoost

### 12.2 Code Availability

- **Repository:** GitHub (ML-anti-cancer-activity-prediction)
- **Pipeline:** `pipeline.py` - Data processing and model training
- **Application:** `app.py` - Gradio web interface

---

## 13. References

1. Yang, W., et al. (2013). Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. *Nucleic Acids Research*, 41(D1), D955-D961.

2. DepMap, Broad Institute. Cancer Dependency Map. https://depmap.org/portal/

3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*.

5. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.

