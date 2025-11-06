# ML-Based Anti-Cancer Activity Prediction

## Project Overview

This project implements a **CNN + Genetic Algorithm hybrid system** for predicting anti-cancer drug activity and generating optimal drug combinations. The system integrates multi-modal cancer data (mutations, gene expression, drug sensitivity) to:

1. **Predict drug efficacy** for cancer cell lines
2. **Predict toxicity** of drug combinations
3. **Generate optimal drug combinations** using genetic algorithms

## Dataset

The project uses real experimental data from multiple sources:

- **Drug Sensitivity Screens**: ~10,836 cell line-drug pairs from GDSC1, GDSC2, CTD², and PRISM
- **Mutation Data**: 1,955 cell lines with hotspot mutations in cancer driver genes (KRAS, TP53, PIK3CA, etc.)
- **Gene Expression**: 1,699 cell lines with RNA-seq expression data
- **Compound Library**: 6,790 FDA-approved drugs + 876 herbal compounds

## Architecture

### CNN Model
- **Input**: Multi-modal features (mutations + expression + metadata)
- **Architecture**: 
  - 1D Convolutional layers (64 → 128 → 256 filters)
  - Global Average Pooling
  - Dense layers (256 → 128 units)
  - Dropout and Batch Normalization for regularization
- **Output**: Drug response classification

### Genetic Algorithm
- **Purpose**: Generate optimal drug combinations
- **Fitness Function**: Maximize efficacy, minimize toxicity
- **Operators**: Tournament selection, two-point crossover, flip mutation
- **Constraints**: 2-5 drugs per combination

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.13+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/Tumo505/ML-anti-cancer-activity-prediction.git
cd ML-anti-cancer-activity-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
ML-anti-cancer-activity-prediction/
├── data/                          # Data directory (gitignored)
│   ├── GENES AND HERBAL COMPOUNDS/
│   │   └── DRUG SENSITIVITY AND MUTATIONS/
│   │       ├── ACTB/
│   │       ├── BRCA/
│   │       ├── KRAS/
│   │       ├── TP53/
│   │       └── ...
│   └── synthetic_anticancer_dataset_1000_samples.csv
│
├── src/                           # Source code
│   ├── data_loader.py            # Multi-modal data loading
│   ├── data_preprocessing.py     # Feature engineering & scaling
│   ├── cnn_model.py              # CNN architecture
│   ├── genetic_algorithm.py      # GA optimizer
│   └── train_pipeline.py         # Complete training pipeline
│
├── notebooks/                     # Jupyter notebooks (for exploration)
├── models/                        # Saved models (gitignored)
├── results/                       # Training outputs (gitignored)
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python src/train_pipeline.py
```

### Step-by-Step Usage

```python
from src.train_pipeline import CNNGATrainingPipeline

# Initialize pipeline
pipeline = CNNGATrainingPipeline(
    data_path="data",
    output_dir="results"
)

# Run complete pipeline
pipeline.run_complete_pipeline(
    cnn_config={
        'conv_filters': [64, 128, 256],
        'dense_units': [256, 128],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    },
    ga_config={
        'n_drugs': 100,
        'min_size': 2,
        'max_size': 5,
        'population_size': 100,
        'n_generations': 50
    }
)
```

### Individual Components

#### 1. Load Data

```python
from src.data_loader import CancerDataLoader

loader = CancerDataLoader(base_path="data")
drug_sens = loader.load_drug_sensitivity()
mutations = loader.load_mutation_data()
expressions = loader.load_expression_data()
merged = loader.merge_all_data()
```

#### 2. Preprocess Data

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
data_splits, feature_info = preprocessor.prepare_for_training(
    df=merged_df,
    feature_cols=feature_columns,
    categorical_cols=['lineage_1', 'lineage_2', 'lineage_3'],
    test_size=0.2,
    val_size=0.1
)
```

#### 3. Train CNN

```python
from src.cnn_model import DrugResponseCNN

model = DrugResponseCNN(
    input_dim=n_features,
    n_classes=n_classes
)
model.build_model()
model.compile_model()
history = model.train(X_train, y_train, X_val, y_val)
```

#### 4. Optimize Combinations

```python
from src.genetic_algorithm import GeneticAlgorithmOptimizer

ga = GeneticAlgorithmOptimizer(n_drugs=100)
ga.set_predictor(predictor_function)
best_combos = ga.optimize()
```

## Results

After training, results are saved to `results/run_YYYYMMDD_HHMMSS/`:

- `data_summary.txt` - Dataset statistics
- `best_cnn_model.h5` - Trained CNN model
- `training_history.png` - Training curves
- `test_metrics.txt` - Test set performance
- `optimized_combinations.csv` - Top drug combinations
- `preprocessor.pkl` - Saved preprocessor

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Loss**: Cross-entropy loss
- **AUC-ROC**: Area under ROC curve (for binary classification)

### Combination Optimization
- **Efficacy Score**: Predicted drug effectiveness (0-1)
- **Toxicity Score**: Predicted toxicity level (0-1)
- **Fitness Score**: Combined score (efficacy - toxicity)

## References

This project is based on the following research:

1. **Deep Learning for Drug Discovery**: https://doi.org/10.3389/fbinf.2023.1225149
2. **Multi-target Drug Prediction**: https://doi.org/10.3389/fmed.2023.1218496
3. **Genetic Algorithm Optimization**: https://doi.org/10.3390/molecules27134098
4. **DeepCancerMap**: https://doi.org/10.1016/j.ejmech.2023.115401

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is for research and educational purposes.


## Acknowledgments

- Cancer Cell Line Encyclopedia (CCLE)
- Genomics of Drug Sensitivity in Cancer (GDSC)
- PRISM Repurposing Screen
- DepMap Portal

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

Built for advancing cancer research and precision medicine.
