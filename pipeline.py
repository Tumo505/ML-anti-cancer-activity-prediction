"""
Drug Sensitivity Prediction Pipeline
Pan-Drug Model: Trains on multiple drugs simultaneously and can generalize to unseen drugs
Uses GDSC drug response data + DepMap gene expression + Drug features + Chemical structures
GPU-accelerated with XGBoost
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator
import warnings
warnings.filterwarnings('ignore')

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class DrugSensitivityPipeline:
    """Pan-Drug prediction pipeline - trains on multiple drugs, generalizes to unseen drugs"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.gdsc_file = self.data_dir / "DRUG SENSITIVITY AND MUTATIONS" / "GDSC1_fitted_dose_response_27Oct23.xlsx"
        self.depmap_expr_file = self.data_dir / "DepMap" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
        self.depmap_model_file = self.data_dir / "DepMap" / "Model.csv"
        self.smiles_file = self.data_dir / "DRUG SENSITIVITY AND MUTATIONS" / "secondary-screen-dose-response-curve-parameters.csv"
        
        self.gdsc_data = None
        self.expression_data = None
        self.model_mapping = None
        self.merged_data = None
        self.drug_encoders = {}
        self.feature_names = None
        self.smiles_data = None
        self.molecular_fingerprints = None
        
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
    
    def load_smiles_data(self):
        """Load SMILES molecular structures"""
        print("\n" + "-"*50)
        print("Loading Chemical Structure Data (SMILES)")
        print("-"*50)
        
        smiles_df = pd.read_csv(self.smiles_file, low_memory=False)
        
        # Clean SMILES: remove trailing commas and whitespace
        smiles_df = smiles_df[['name', 'smiles']].dropna()
        smiles_df['smiles'] = smiles_df['smiles'].str.strip().str.rstrip(',')
        smiles_df = smiles_df.drop_duplicates()
        smiles_df.columns = ['DRUG_NAME', 'SMILES']
        
        print(f"Loaded SMILES for {len(smiles_df)} unique drugs")
        self.smiles_data = smiles_df
        
        return self.smiles_data
    
    def generate_molecular_fingerprints(self, fp_type='morgan', radius=2, n_bits=512):
        """
        Generate molecular fingerprints from SMILES using modern RDKit API
        
        Parameters:
        -----------
        fp_type : str
            'morgan' (ECFP), 'maccs', or 'rdkit'
        radius : int
            Radius for Morgan fingerprints
        n_bits : int
            Number of bits for Morgan/RDKit fingerprints
        """
        print("\n" + "-"*50)
        print(f"Generating {fp_type.upper()} Molecular Fingerprints")
        print("-"*50)
        
        if self.smiles_data is None:
            self.load_smiles_data()
        
        fingerprints = []
        valid_drugs = []
        failed_count = 0
        
        # Create generator for Morgan fingerprints (modern API)
        if fp_type == 'morgan':
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        
        for idx, row in self.smiles_data.iterrows():
            drug_name = row['DRUG_NAME']
            smiles = row['SMILES']
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_count += 1
                    continue
                
                if fp_type == 'morgan':
                    # Use modern MorganGenerator API
                    fp = morgan_gen.GetFingerprint(mol)
                    fp_array = np.array(fp)
                elif fp_type == 'maccs':
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    fp_array = np.array(fp)
                elif fp_type == 'rdkit':
                    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
                    fp_array = np.array(fp)
                else:
                    raise ValueError(f"Unknown fingerprint type: {fp_type}")
                
                fingerprints.append(fp_array)
                valid_drugs.append(drug_name)
                
            except Exception as e:
                failed_count += 1
                continue
        
        fp_df = pd.DataFrame(
            fingerprints,
            index=valid_drugs,
            columns=[f'fp_{i}' for i in range(len(fingerprints[0]))]
        )
        
        print(f"Generated fingerprints for {len(fp_df)} drugs")
        print(f"Failed to parse {failed_count} SMILES strings")
        print(f"Fingerprint dimension: {fp_df.shape[1]}")
        
        self.molecular_fingerprints = fp_df
        return fp_df
    
    def encode_drug_features(self):
        """Encode drug properties as numerical features"""
        print("\n" + "-"*50)
        print("Encoding Drug Features")
        print("-"*50)

        # Extract drug metadata
        drug_info = self.merged_data[['DRUG_NAME', 'PUTATIVE_TARGET', 'PATHWAY_NAME']].drop_duplicates()

        # Encode target proteins
        target_encoder = LabelEncoder()
        self.merged_data['target_encoded'] = target_encoder.fit_transform(
            self.merged_data['PUTATIVE_TARGET'].fillna('Unknown')
        )
        self.drug_encoders['target'] = target_encoder

        # Encode pathways
        pathway_encoder = LabelEncoder()
        self.merged_data['pathway_encoded'] = pathway_encoder.fit_transform(
            self.merged_data['PATHWAY_NAME'].fillna('Unknown')
        )
        self.drug_encoders['pathway'] = pathway_encoder

        # Encode drug names (for complete drug representation)
        drug_encoder = LabelEncoder()
        self.merged_data['drug_encoded'] = drug_encoder.fit_transform(
            self.merged_data['DRUG_NAME']
        )
        self.drug_encoders['drug'] = drug_encoder
        
        print(f"Unique targets: {len(target_encoder.classes_)}")
        print(f"Unique pathways: {len(pathway_encoder.classes_)}")
        print(f"Unique drugs: {len(drug_encoder.classes_)}")
        
        return self.merged_data
    
    def prepare_pan_drug_features(self, min_samples_per_drug=100, use_top_genes=1000, 
                                   include_drug_identity=True, include_molecular_fp=True,
                                   fp_type='morgan', fp_radius=2, fp_bits=256):
        """Prepare features for pan-drug model (all drugs combined)"""
        print(f"\n{'='*50}")
        print("Preparing Pan-Drug Features")
        print(f"{'='*50}")
        
        # Filter drugs with sufficient samples
        drug_counts = self.merged_data['DRUG_NAME'].value_counts()
        valid_drugs = drug_counts[drug_counts >= min_samples_per_drug].index
        
        filtered_data = self.merged_data[self.merged_data['DRUG_NAME'].isin(valid_drugs)].copy()
        print(f"Drugs with >={min_samples_per_drug} samples: {len(valid_drugs)}")
        print(f"Total experiments: {len(filtered_data):,}")
        
        # Get gene columns
        gene_cols = [col for col in filtered_data.columns if '(' in col]
        
        # Feature selection: use most variable genes across ALL drugs
        if use_top_genes and use_top_genes < len(gene_cols):
            gene_variances = filtered_data[gene_cols].var()
            top_genes = gene_variances.nlargest(use_top_genes).index.tolist()
            print(f"Using top {use_top_genes} most variable genes (across all drugs)")
        else:
            top_genes = gene_cols
            print(f"Using all {len(top_genes)} genes")
        
        # Build feature matrix
        feature_cols = top_genes.copy()
        
        # Add drug features
        feature_cols.extend(['target_encoded', 'pathway_encoded'])
        
        # Optionally include drug identity (allows drug-specific patterns)
        if include_drug_identity:
            feature_cols.append('drug_encoded')
            print("Including drug identity as feature (enables drug-specific patterns)")
        else:
            print("Excluding drug identity (pure generalization mode)")
        
        X = filtered_data[feature_cols].copy()
        y = filtered_data['AUC'].copy()
        drug_names = filtered_data['DRUG_NAME'].copy()
        
        # Add molecular fingerprints if requested
        if include_molecular_fp:
            print(f"\nAdding molecular fingerprints ({fp_type.upper()}, {fp_bits} bits)...")
            
            if self.molecular_fingerprints is None or len(self.molecular_fingerprints.columns) != fp_bits:
                self.generate_molecular_fingerprints(fp_type=fp_type, radius=fp_radius, n_bits=fp_bits)
            
            fp_features = []
            drugs_with_fp = 0
            
            for drug in filtered_data['DRUG_NAME']:
                if drug in self.molecular_fingerprints.index:
                    fp_features.append(self.molecular_fingerprints.loc[drug].values)
                    drugs_with_fp += 1
                else:
                    fp_features.append(np.zeros(fp_bits))
            
            fp_df = pd.DataFrame(
                fp_features,
                index=X.index,
                columns=[f'fp_{i}' for i in range(fp_bits)]
            )
            
            X = pd.concat([X, fp_df], axis=1)
            feature_cols.extend(fp_df.columns.tolist())
            
            # Calculate correct percentage: unique drugs with fingerprints
            unique_drugs_in_data = filtered_data['DRUG_NAME'].nunique()
            unique_drugs_with_fp = filtered_data['DRUG_NAME'].unique()
            drugs_with_fp_count = sum(1 for d in unique_drugs_with_fp if d in self.molecular_fingerprints.index)
            
            print(f"Drugs with molecular fingerprints: {drugs_with_fp_count}/{unique_drugs_in_data} ({drugs_with_fp_count/unique_drugs_in_data*100:.1f}%)")
            print(f"Total samples with fingerprints: {drugs_with_fp}/{len(filtered_data)} ({drugs_with_fp/len(filtered_data)*100:.1f}%)")
        
        self.feature_names = feature_cols
        
        print(f"\nFeature composition:")
        print(f"  - Gene expression features: {len(top_genes)}")
        print(f"  - Drug target (encoded): 1")
        print(f"  - Drug pathway (encoded): 1")
        if include_drug_identity:
            print(f"  - Drug identity (encoded): 1")
        if include_molecular_fp:
            print(f"  - Molecular fingerprints: {fp_bits}")
        print(f"  - Total features: {X.shape[1]}")
        print(f"\nTarget (AUC) - mean: {y.mean():.3f}, std: {y.std():.3f}, range: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, y, drug_names, valid_drugs.tolist()
    

    
    def train_pan_drug_model(self, X, y, drug_names, test_size=0.2, random_state=42, 
                            use_cv=True, n_folds=5):
        """Train pan-drug model with GPU-accelerated XGBoost"""
        print(f"\n{'='*50}")
        print("Training Pan-Drug Model (GPU-Accelerated)")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_test, y_train, y_test, drugs_train, drugs_test = train_test_split(
            X, y, drug_names, test_size=test_size, random_state=random_state, stratify=drug_names
        )
        
        print(f"Train set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Drugs in train: {drugs_train.nunique()}")
        print(f"Drugs in test: {drugs_test.nunique()}")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Train XGBoost with GPU acceleration
        print("\nTraining XGBoost Regressor on GPU (RTX 5070 Ti)...")
        print("Device: CUDA")
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            device='cuda',
            tree_method='hist',
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Cross-validation on training set
        if use_cv:
            print(f"\nPerforming {n_folds}-fold cross-validation...")
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                       cv=kfold, scoring='r2', n_jobs=-1)
            print(f"CV R² scores: {cv_scores}")
            print(f"CV R² mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Metrics
        results = self.calculate_metrics(y_train, train_pred, y_test, test_pred)
        
        # Per-drug performance
        print("\n" + "="*50)
        print("Per-Drug Performance on Test Set")
        print("="*50)
        test_df = pd.DataFrame({
            'drug': drugs_test.values,
            'true': y_test.values,
            'pred': test_pred
        })
        
        per_drug_results = []
        for drug in test_df['drug'].unique():
            drug_data = test_df[test_df['drug'] == drug]
            if len(drug_data) >= 10:  # Only report if enough samples
                drug_r2 = r2_score(drug_data['true'], drug_data['pred'])
                drug_pearson, _ = pearsonr(drug_data['true'], drug_data['pred'])
                per_drug_results.append({
                    'drug': drug,
                    'n_samples': len(drug_data),
                    'r2': drug_r2,
                    'pearson': drug_pearson
                })
        
        per_drug_df = pd.DataFrame(per_drug_results).sort_values('r2', ascending=False)
        print(f"\nTop 10 Best Predicted Drugs:")
        print(per_drug_df.head(10).to_string(index=False))
        print(f"\nBottom 10 Worst Predicted Drugs:")
        print(per_drug_df.tail(10).to_string(index=False))
        
        # Feature importance
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance_df
        results['per_drug_performance'] = per_drug_df
        results['model'] = model
        results['scaler'] = scaler
        results['imputer'] = imputer
        if use_cv:
            results['cv_scores'] = cv_scores
        
        return results
    

    
    def predict_new_drug(self, model, scaler, imputer, cell_line_expression, 
                        drug_target, drug_pathway, drug_name=None):
        """
        Predict drug sensitivity for a new/unseen drug
        
        Parameters:
        -----------
        model : trained model
        scaler : fitted StandardScaler
        imputer : fitted SimpleImputer
        cell_line_expression : array-like, gene expression for cell line(s)
        drug_target : str, target protein (e.g., "BRAF", "EGFR")
        drug_pathway : str, pathway name (e.g., "ERK MAPK signaling")
        drug_name : str, optional drug name for drug_encoded feature
        
        Returns:
        --------
        predictions : array of predicted AUC values
        """
        print(f"\nPredicting sensitivity for new drug:")
        print(f"  Target: {drug_target}")
        print(f"  Pathway: {drug_pathway}")
        
        # Encode drug features
        try:
            target_encoded = self.drug_encoders['target'].transform([drug_target])[0]
        except ValueError:
            print(f"  Warning: Unknown target '{drug_target}', using default encoding")
            target_encoded = -1
        
        try:
            pathway_encoded = self.drug_encoders['pathway'].transform([drug_pathway])[0]
        except ValueError:
            print(f"  Warning: Unknown pathway '{drug_pathway}', using default encoding")
            pathway_encoded = -1
        
        # Handle drug identity encoding
        if 'drug_encoded' in self.feature_names:
            if drug_name and drug_name in self.drug_encoders['drug'].classes_:
                drug_encoded = self.drug_encoders['drug'].transform([drug_name])[0]
            else:
                # Use average encoding for unseen drugs
                drug_encoded = len(self.drug_encoders['drug'].classes_) // 2
                print(f"  Using neutral drug encoding for unseen drug")
        
        # Build feature vector
        n_samples = cell_line_expression.shape[0] if len(cell_line_expression.shape) > 1 else 1
        
        # Add drug features to each sample
        drug_features = np.array([[target_encoded, pathway_encoded]])
        if 'drug_encoded' in self.feature_names:
            drug_features = np.hstack([drug_features, [[drug_encoded]]])
        
        # Combine with expression
        if len(cell_line_expression.shape) == 1:
            cell_line_expression = cell_line_expression.reshape(1, -1)
        
        X = np.hstack([cell_line_expression, np.repeat(drug_features, n_samples, axis=0)])
        
        # Apply preprocessing
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        
        # Predict
        predictions = model.predict(X_scaled)
        
        print(f"  Predictions: {predictions}")
        return predictions
    
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


def save_model(pipeline, results, save_dir="saved_model"):
    """Save trained model and preprocessing objects"""
    import pickle
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Saving model to {save_path}...")
    print(f"{'='*50}")
    
    # Save model components
    with open(save_path / "model.pkl", "wb") as f:
        pickle.dump(results['model'], f)
    
    with open(save_path / "scaler.pkl", "wb") as f:
        pickle.dump(results['scaler'], f)
    
    with open(save_path / "imputer.pkl", "wb") as f:
        pickle.dump(results['imputer'], f)
    
    with open(save_path / "feature_names.pkl", "wb") as f:
        pickle.dump(pipeline.feature_names, f)
    
    print(f"Model saved successfully!")
    print(f"  - model.pkl")
    print(f"  - scaler.pkl")
    print(f"  - imputer.pkl")
    print(f"  - feature_names.pkl")


def main():
    """Train unified pan-drug model with GPU acceleration"""
    print("-"*50)
    print("Pan-Drug Training Mode (GPU-Accelerated)")
    print("Training unified model across all drugs")
    print("-"*50)
    
    # Initialize pipeline
    pipeline = DrugSensitivityPipeline()
    
    # Load all data
    pipeline.load_gdsc_data()
    pipeline.load_depmap_expression()
    pipeline.load_model_mapping()
    pipeline.merge_datasets()
    
    # Encode drug features
    pipeline.encode_drug_features()
    
    # Get drug statistics
    print("\n" + "-"*50)
    print("Drug Statistics")
    print("-"*50)
    drug_stats = pipeline.get_drug_statistics()
    print(f"Total drugs: {len(drug_stats)}")
    print(f"Total experiments: {len(pipeline.merged_data):,}")
    print(f"\nTop 15 Drugs by Number of Experiments:")
    print(drug_stats.head(15).to_string())
    
    # Load SMILES and generate fingerprints
    pipeline.load_smiles_data()
    
    # Prepare pan-drug features (with molecular fingerprints)
    X, y, drug_names, valid_drugs = pipeline.prepare_pan_drug_features(
        min_samples_per_drug=100,
        use_top_genes=1000,
        include_drug_identity=True,
        include_molecular_fp=True,
        fp_type='morgan',
        fp_radius=2,
        fp_bits=256
    )
    
    # Train pan-drug model
    results = pipeline.train_pan_drug_model(X, y, drug_names, use_cv=True, n_folds=5)
    
    # Save model for Gradio app
    save_model(pipeline, results)
    
    # Display feature importance
    print(f"\n{'='*50}")
    print("Top 20 Most Important Features")
    print(f"{'='*50}")
    top_features = results['feature_importance'].head(20)
    
    # Categorize features
    gene_features = top_features[top_features['feature'].str.contains(r'\(', regex=True)]
    drug_features = top_features[~top_features['feature'].str.contains(r'\(', regex=True)]
    
    print("\nTop Drug/Pathway Features:")
    if len(drug_features) > 0:
        for idx, row in drug_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.6f}")
    
    print(f"\nTop Gene Expression Features:")
    print(gene_features.head(15).to_string(index=False))
    
    # Summary
    print(f"\n{'='*50}")
    print("Pan-Drug Model Summary")
    print(f"{'='*50}")
    print(f"Training drugs: {len(valid_drugs)}")
    print(f"Training samples: {len(X):,}")
    print(f"Features: {X.shape[1]} (genes + drug metadata)")
    print(f"\nOverall Performance:")
    print(f"  Test R²: {results['test_r2']:.4f}")
    print(f"  Test RMSE: {results['test_rmse']:.4f}")
    print(f"  Test Pearson: {results['test_pearson']:.4f}")
    
    if 'cv_scores' in results:
        print(f"  CV R² (mean): {results['cv_scores'].mean():.4f}")
    
    # Performance interpretation
    print(f"\n{'='*50}")
    print("Model Capabilities")
    print(f"{'='*50}")
    if results['test_r2'] > 0.4:
        print("EXCELLENT: Strong pan-drug predictive power")
        print("  - Model learns generalizable patterns across drugs")
        print("  - Can make reliable predictions for seen drugs")
        print("  - Transfer learning to similar drugs is promising")
    elif results['test_r2'] > 0.25:
        print("GOOD: Model captures meaningful drug-gene relationships")
        print("  - Useful for drug prioritization and biomarker discovery")
        print("  - Performance varies by drug class and mechanism")
    elif results['test_r2'] > 0.15:
        print("MODERATE: Model shows some predictive signal")
        print("  - Better than random, but room for improvement")
        print("  - Consider: more features, deep learning, or drug embeddings")
    else:
        print("LIMITED: Weak generalization across drugs")
        print("  - Drug response may be highly drug-specific")
        print("  - Consider: drug-specific models or more complex architectures")
    
    print("\nTo predict on new/unseen drugs:")
    print("  1. Ensure drug has: target protein + pathway annotation")
    print("  2. Encode using same encoders (target_encoder, pathway_encoder)")
    print("  3. Combine with cell line gene expression")
    print("  4. Apply imputer + scaler + model")
    
    print("-"*50)
    
    return pipeline, results


if __name__ == "__main__":
    main()
