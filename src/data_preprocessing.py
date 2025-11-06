"""
Data preprocessing module for cancer drug response prediction.
Handles feature engineering, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import pickle


class DataPreprocessor:
    """Preprocess multi-modal cancer data for ML models."""
    
    def __init__(self):
        """Initialize preprocessor with encoders and scalers."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.feature_names = {}
        
    def encode_categorical_features(
        self, 
        df: pd.DataFrame, 
        categorical_cols: List[str],
        encoding_type: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical column names
            encoding_type: 'onehot' or 'label'
        
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            # Fill missing values with 'Unknown'
            df[col] = df[col].fillna('Unknown')
            
            if encoding_type == 'onehot':
                # One-hot encoding
                if col not in self.onehot_encoders:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    self.onehot_encoders[col] = encoder
                else:
                    encoded = self.onehot_encoders[col].transform(df[[col]])
                
                # Create new column names
                feature_names = [f"{col}_{cat}" for cat in self.onehot_encoders[col].categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                
                # Drop original column and add encoded columns
                df = df.drop(columns=[col])
                df = pd.concat([df, encoded_df], axis=1)
                
            elif encoding_type == 'label':
                # Label encoding
                if col not in self.label_encoders:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])
                    self.label_encoders[col] = encoder
                else:
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'median'
    ) -> pd.DataFrame:
        """
        Handle missing values in numerical features.
        
        Args:
            df: Input dataframe
            strategy: 'mean', 'median', or 'zero'
        
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].isnull().any():
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'zero':
                    df[col].fillna(0, inplace=True)
        
        return df
    
    def scale_features(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input dataframe
            feature_cols: List of columns to scale
            fit: Whether to fit the scaler (True for training, False for test)
        
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if fit:
            scaled_values = self.scaler.fit_transform(df[feature_cols])
        else:
            scaled_values = self.scaler.transform(df[feature_cols])
        
        df[feature_cols] = scaled_values
        
        return df
    
    def create_drug_response_targets(
        self, 
        df: pd.DataFrame, 
        sensitivity_col: str = 'target_gene'
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Create multiple target variables for multi-task learning.
        
        Args:
            df: Input dataframe
            sensitivity_col: Column containing drug sensitivity values
        
        Returns:
            Tuple of (dataframe, dict of target arrays)
        """
        targets = {}
        
        # For now, we'll use target_gene as a classification target
        # In a real scenario, you would have actual AUC or IC50 values
        if sensitivity_col in df.columns:
            # Encode target gene as classification
            if 'target_label_encoder' not in self.label_encoders:
                encoder = LabelEncoder()
                targets['gene_class'] = encoder.fit_transform(df[sensitivity_col])
                self.label_encoders['target_label_encoder'] = encoder
            else:
                targets['gene_class'] = self.label_encoders['target_label_encoder'].transform(df[sensitivity_col])
        
        return df, targets
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str],
        target_col: str = 'target_gene',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
        """
        Complete preprocessing pipeline for training.
        
        Args:
            df: Input dataframe
            feature_cols: List of feature columns
            categorical_cols: List of categorical columns
            target_col: Target column name
            test_size: Test set proportion
            val_size: Validation set proportion
            random_state: Random seed
        
        Returns:
            Tuple of (data_splits dict, feature_info dict)
        """
        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE")
        print("="*80)
        
        # 1. Handle missing values in numerical features
        print("\n1. Handling missing values...")
        numerical_cols = [col for col in feature_cols if col not in categorical_cols]
        df = self.handle_missing_values(df)
        print(f"   Imputed {len(numerical_cols)} numerical features")
        
        # 2. Encode categorical features
        print("\n2. Encoding categorical features...")
        df_encoded = self.encode_categorical_features(
            df, 
            categorical_cols, 
            encoding_type='onehot'
        )
        print(f"   Encoded {len(categorical_cols)} categorical features")
        
        # 3. Get final feature columns (after encoding)
        # Get all encoded categorical columns
        encoded_cat_cols = [col for col in df_encoded.columns 
                           if any(col.startswith(f"{cat}_") for cat in categorical_cols)]
        
        # Combine numerical features and encoded categorical features
        all_feature_cols = numerical_cols + encoded_cat_cols
        
        # 4. Extract features and target
        X = df_encoded[all_feature_cols].values
        y, targets_dict = self.create_drug_response_targets(df_encoded, target_col)
        y = targets_dict['gene_class']
        
        print(f"\n3. Feature matrix shape: {X.shape}")
        print(f"   {X.shape[0]} samples, {X.shape[1]} features")
        
        # Validate data before splitting
        print(f"\n   Checking data quality...")
        print(f"   - NaN in features: {np.isnan(X).sum()}")
        print(f"   - Inf in features: {np.isinf(X).sum()}")
        print(f"   - Feature range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"   - Target classes: {np.unique(y)}")
        print(f"   - Target range: [{y.min()}, {y.max()}]")
        
        # Replace any remaining NaN or Inf with 0
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"   WARNING: Replacing {np.isnan(X).sum()} NaN and {np.isinf(X).sum()} Inf values with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 5. Split data
        print("\n4. Splitting data...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
        )
        
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
        )
        
        print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # 6. Scale features
        print("\n5. Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        print("   Features standardized (mean=0, std=1)")
        
        # Validate scaled data
        print(f"\n   Post-scaling validation:")
        print(f"   - NaN in train: {np.isnan(X_train_scaled).sum()}")
        print(f"   - Inf in train: {np.isinf(X_train_scaled).sum()}")
        print(f"   - Train range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        
        # Replace any NaN/Inf after scaling
        if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
            print(f"   WARNING: Fixing NaN/Inf in scaled data")
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prepare return data
        data_splits = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        feature_info = {
            'feature_names': all_feature_cols,
            'numerical_features': numerical_cols,
            'categorical_features': categorical_cols,
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }
        
        self.feature_names = all_feature_cols
        
        print("\nPREPROCESSING COMPLETE")
        print("="*80)
        
        return data_splits, feature_info
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor objects to disk."""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'onehot_encoders': self.onehot_encoders,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor objects from disk."""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.onehot_encoders = preprocessor_data['onehot_encoders']
        self.feature_names = preprocessor_data['feature_names']
        
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("DATA PREPROCESSOR - TEST")
    print("="*80)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'cancer_type': np.random.choice(['Lung', 'Breast', 'Colon'], 100),
        'target_gene': np.random.choice(['KRAS', 'TP53', 'PIK3CA'], 100)
    })
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Test encoding
    print("\n1. Testing categorical encoding...")
    encoded_df = preprocessor.encode_categorical_features(
        sample_data, 
        ['cancer_type'], 
        encoding_type='onehot'
    )
    print(f"   Columns after encoding: {encoded_df.columns.tolist()}")
    
    # Test scaling
    print("\n2. Testing feature scaling...")
    scaled_df = preprocessor.scale_features(
        encoded_df, 
        ['feature1', 'feature2']
    )
    print(f"   Feature1 mean: {scaled_df['feature1'].mean():.6f}")
    print(f"   Feature1 std: {scaled_df['feature1'].std():.6f}")
    
    print("\nPreprocessing test complete!")
