"""
Drug Sensitivity Prediction Pipeline
Uses GDSC drug response data + DepMap gene expression
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class DrugSensitivityPipeline:
    """Complete pipeline for drug sensitivity prediction"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.gdsc_file = self.data_dir / "DRUG SENSITIVITY AND MUTATIONS" / "GDSC1_fitted_dose_response_27Oct23.xlsx"
        self.depmap_expr_file = self.data_dir / "DepMap" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
        self.depmap_model_file = self.data_dir / "DepMap" / "Model.csv"
        
        self.gdsc_data = None
        self.expression_data = None
        self.model_mapping = None
        self.merged_data = None
        
    def load_gdsc_data(self):
        """Load GDSC drug sensitivity data"""
        print("-"*50)
        print("Loading GDSC Drug Sensitivity Data")
        print("-"*50)
        
        self.gdsc_data = pd.read_excel(self.gdsc_file)
        print(f"Loaded {len(self.gdsc_data):,} drug response experiments")
        print(f"Unique drugs: {self.gdsc_data['DRUG_NAME'].nunique()}")
        print(f"Unique cell lines: {self.gdsc_data['CELL_LINE_NAME'].nunique()}")
        
        return self.gdsc_data
    
    def load_depmap_expression(self):
        """Load DepMap gene expression data"""
        print("\n" + "-"*50)
        print("Loading DepMap Gene Expression Data")
        print("-"*50)
        
        # Load expression data
        expr_df = pd.read_csv(self.depmap_expr_file)
        print(f"Expression data shape: {expr_df.shape}")
        print(f"Cell lines: {len(expr_df)}")
        print(f"Genes: {expr_df.shape[1] - 6}")
        
        # Keep only default entries for each model
        expr_df = expr_df[expr_df['IsDefaultEntryForModel'] == 'Yes'].copy()
        print(f"After filtering for default entries: {len(expr_df)} cell lines")
        
        # Set ModelID as index and keep only gene columns
        gene_cols = [col for col in expr_df.columns if '(' in col]  # Gene columns have format "GENE (ID)"
        expr_df = expr_df[['ModelID'] + gene_cols].set_index('ModelID')
        
        print(f"Gene expression matrix: {expr_df.shape}")
        self.expression_data = expr_df
        
        return self.expression_data
    
    def load_model_mapping(self):
        """Load DepMap model mapping to link to GDSC"""
        print("\n" + "-"*50)
        print("Loading Model Mapping")
        print("-"*50)
        
        self.model_mapping = pd.read_csv(self.depmap_model_file)
        print(f"Total models: {len(self.model_mapping)}")
        
        # Filter to those with SangerModelID
        self.model_mapping = self.model_mapping[self.model_mapping['SangerModelID'].notna()].copy()
        print(f"Models with SangerModelID: {len(self.model_mapping)}")
        
        return self.model_mapping
    
    def merge_datasets(self):
        """Merge GDSC drug data with DepMap expression"""
        print("\n" + "-"*50)
        print("Merging Datasets")
        print("-"*50)
        
        # Merge GDSC with model mapping
        merged = self.gdsc_data.merge(
            self.model_mapping[['ModelID', 'SangerModelID']],
            left_on='SANGER_MODEL_ID',
            right_on='SangerModelID',
            how='inner'
        )
        print(f"After merging GDSC with model mapping: {len(merged):,} experiments")
        
        # Merge with expression data
        merged = merged.merge(
            self.expression_data,
            left_on='ModelID',
            right_index=True,
            how='inner'
        )
        print(f"After merging with expression data: {len(merged):,} experiments")
        print(f"Unique drugs: {merged['DRUG_NAME'].nunique()}")
        print(f"Unique cell lines: {merged['ModelID'].nunique()}")
        
        self.merged_data = merged
        return self.merged_data
    
    def get_drug_statistics(self):
        """Get statistics on drug coverage"""
        drug_stats = self.merged_data.groupby('DRUG_NAME').agg({
            'ModelID': 'count',
            'AUC': ['mean', 'std'],
            'PUTATIVE_TARGET': 'first',
            'PATHWAY_NAME': 'first'
        }).round(3)
        
        drug_stats.columns = ['num_experiments', 'mean_auc', 'std_auc', 'target', 'pathway']
        drug_stats = drug_stats.sort_values('num_experiments', ascending=False)
        
        return drug_stats
    
    def prepare_features_for_drug(self, drug_name, use_top_genes=1000):
        """Prepare features for a specific drug"""
        print(f"\n{'='*50}")
        print(f"Preparing Features for: {drug_name}")
        print(f"{'='*50}")
        
        # Filter for specific drug
        drug_data = self.merged_data[self.merged_data['DRUG_NAME'] == drug_name].copy()
        print(f"Experiments: {len(drug_data)}")
        
        if len(drug_data) < 50:
            print(f"Warning: Only {len(drug_data)} samples - need at least 50")
            return None, None, None
        
        # Get gene columns
        gene_cols = [col for col in drug_data.columns if '(' in col]
        
        # Feature selection: use most variable genes
        if use_top_genes and use_top_genes < len(gene_cols):
            gene_variances = drug_data[gene_cols].var()
            top_genes = gene_variances.nlargest(use_top_genes).index.tolist()
            print(f"Using top {use_top_genes} most variable genes")
        else:
            top_genes = gene_cols
            print(f"Using all {len(top_genes)} genes")
        
        # Prepare features and target
        X = drug_data[top_genes].copy()
        y = drug_data['AUC'].copy()
        
        print(f"Features shape: {X.shape}")
        print(f"Target (AUC) - mean: {y.mean():.3f}, std: {y.std():.3f}, range: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, y, top_genes
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train Random Forest model"""
        print(f"\n{'='*50}")
        print("Training Model")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Train Random Forest with regularization to prevent overfitting
        print("\nTraining Random Forest...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Metrics
        results = self.calculate_metrics(y_train, train_pred, y_test, test_pred)
        
        # Feature importance
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance_df
        results['model'] = model
        results['scaler'] = scaler
        results['imputer'] = imputer
        
        return results
    
    def calculate_metrics(self, y_train, train_pred, y_test, test_pred):
        """Calculate performance metrics"""
        # R-squared
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # MAE
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Correlation
        train_pearson, _ = pearsonr(y_train, train_pred)
        test_pearson, _ = pearsonr(y_test, test_pred)
        train_spearman, _ = spearmanr(y_train, train_pred)
        test_spearman, _ = spearmanr(y_test, test_pred)
        
        print(f"\n{'='*50}")
        print("Performance Metrics")
        print(f"{'='*50}")
        print(f"{'Metric':<20} {'Train':<12} {'Test':<12}")
        print("-" * 80)
        print(f"{'R-squared':<20} {train_r2:<12.4f} {test_r2:<12.4f}")
        print(f"{'RMSE':<20} {train_rmse:<12.4f} {test_rmse:<12.4f}")
        print(f"{'MAE':<20} {train_mae:<12.4f} {test_mae:<12.4f}")
        print(f"{'Pearson r':<20} {train_pearson:<12.4f} {test_pearson:<12.4f}")
        print(f"{'Spearman rho':<20} {train_spearman:<12.4f} {test_spearman:<12.4f}")
        print("-" * 80)
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_pearson': train_pearson,
            'test_pearson': test_pearson,
            'train_spearman': train_spearman,
            'test_spearman': test_spearman,
            'test_predictions': test_pred
        }


def main():
    """Main pipeline execution"""
    print("-"*50)
    print("GDSC Drug Sensitivity Prediction with DepMap Expression")
    print("-"*50)
    
    # Initialize pipeline
    pipeline = DrugSensitivityPipeline()
    
    # Load all data
    pipeline.load_gdsc_data()
    pipeline.load_depmap_expression()
    pipeline.load_model_mapping()
    pipeline.merge_datasets()
    
    # Get drug statistics
    print("\n" + "-"*50)
    print("Drug Statistics")
    print("-"*50)
    drug_stats = pipeline.get_drug_statistics()
    print("\nTop 20 Drugs by Number of Experiments:")
    print(drug_stats.head(20).to_string())
    
    # Select a well-tested drug
    top_drug = drug_stats.index[0]
    print(f"\n{'='*50}")
    print(f"Training Model for: {top_drug}")
    print(f"Target: {drug_stats.loc[top_drug, 'target']}")
    print(f"Pathway: {drug_stats.loc[top_drug, 'pathway']}")
    print(f"{'='*50}")
    
    # Prepare features
    X, y, feature_names = pipeline.prepare_features_for_drug(top_drug, use_top_genes=1000)
    
    if X is not None:
        # Train model
        results = pipeline.train_model(X, y)
        
        # Display top features
        print(f"\nTop 15 Most Important Genes:")
        print(results['feature_importance'].head(15).to_string(index=False))
        
        # Summary
        print(f"\n{'='*50}")
        print("Summary")
        print(f"{'='*50}")
        print(f"Drug: {top_drug}")
        print(f"Training samples: {len(X)}")
        print(f"Features: {X.shape[1]} genes")
        print(f"Test R-squared: {results['test_r2']:.4f}")
        print(f"Test RMSE: {results['test_rmse']:.4f}")
        print(f"Test Pearson correlation: {results['test_pearson']:.4f}")
        
        if results['test_r2'] > 0.5:
            print("\nExcellent performance! The model has strong predictive power.")
        elif results['test_r2'] > 0.3:
            print("\nGood performance! The model captures meaningful patterns.")
        elif results['test_r2'] > 0.1:
            print("\nModerate performance. Consider tuning hyperparameters.")
        else:
            print("\nLow performance. Drug may have limited biomarkers.")
        
        print("-"*50)
    else:
        print("Insufficient data for this drug")


if __name__ == "__main__":
    main()
