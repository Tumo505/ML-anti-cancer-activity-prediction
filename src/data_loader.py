"""
Data loader module for integrating drug sensitivity, mutation, and expression data.
Handles multiple datasets from GDSC, PRISM, CTD^2 screens.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CancerDataLoader:
    """Load and integrate multi-modal cancer drug response data."""
    
    def __init__(self, base_path: str):
        """
        Initialize data loader.
        
        Args:
            base_path: Path to the data directory
        """
        self.base_path = Path(base_path)
        self.drug_sensitivity_path = self.base_path / "GENES AND HERBAL COMPOUNDS" / "DRUG SENSITIVITY AND MUTATIONS"
        
        # Gene targets to include
        self.genes = ['ACTB', 'BRCA', 'CDK12', 'GADPH', 'KRAS', 'PIK3CA', 'SLC7A11', 'TP53']
        
        # Data containers
        self.drug_sensitivity_data = None
        self.mutation_data = None
        self.expression_data = None
        self.merged_data = None
        
    def load_drug_sensitivity(self, screen_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load drug sensitivity data from multiple screening platforms.
        
        Args:
            screen_types: List of screen types to load. Options:
                        ['GDSC1', 'GDSC2', 'CTD2', 'PRISM']
                        If None, loads all available screens.

        Returns:
            Combined drug sensitivity dataframe
        """
        if screen_types is None:
            screen_types = ['GDSC1', 'GDSC2', 'CTD2', 'PRISM']
        
        all_sensitivity_data = []
        
        for gene in self.genes:
            gene_path = self.drug_sensitivity_path / gene
            if not gene_path.exists():
                continue
            
            # Load different screening platforms
            for file in gene_path.glob("Drug_sensitivity*.csv"):
                try:
                    df = pd.read_csv(file)
                    
                    # Determine screen type from filename
                    if 'GDSC1' in file.name and 'GDSC1' in screen_types:
                        df['screen_type'] = 'GDSC1'
                    elif 'GDSC2' in file.name and 'GDSC2' in screen_types:
                        df['screen_type'] = 'GDSC2'
                    elif 'CTD' in file.name and 'CTD2' in screen_types:
                        df['screen_type'] = 'CTD2'
                    else:
                        continue
                    
                    df['target_gene'] = gene
                    all_sensitivity_data.append(df)
                    print(f"Loaded {file.name} for {gene}: {len(df)} samples")
                    
                except Exception as e:
                    print(f"Error loading {file.name}: {e}")
            
            # Load PRISM repurposing data
            prism_file = gene_path / "PRISM_Repurposing_Public_24Q2_subsetted.csv"
            if prism_file.exists() and 'PRISM' in screen_types:
                try:
                    df = pd.read_csv(prism_file)
                    df['screen_type'] = 'PRISM'
                    df['target_gene'] = gene
                    all_sensitivity_data.append(df)
                    print(f"Loaded PRISM data for {gene}: {len(df)} samples")
                except Exception as e:
                    print(f"Error loading PRISM for {gene}: {e}")
        
        if all_sensitivity_data:
            self.drug_sensitivity_data = pd.concat(all_sensitivity_data, ignore_index=True)
            print(f"\nTotal drug sensitivity records: {len(self.drug_sensitivity_data)}")
            return self.drug_sensitivity_data
        else:
            raise ValueError("No drug sensitivity data found!")
    
    def load_mutation_data(self) -> pd.DataFrame:
        """
        Load mutation data for cancer driver genes.
        
        Returns:
            Mutation dataframe with binary indicators
        """
        mutation_dfs = []
        
        for gene in self.genes:
            gene_path = self.drug_sensitivity_path / gene
            
            # Look for mutation files
            for mutation_file in gene_path.glob("*Mutation*.csv"):
                try:
                    df = pd.read_csv(mutation_file)
                    
                    # Rename gene column to indicate mutation status
                    if gene in df.columns:
                        df = df.rename(columns={gene: f'{gene}_mutation'})
                    
                    mutation_dfs.append(df)
                    print(f"Loaded mutation data for {gene}: {len(df)} cell lines")
                    
                except Exception as e:
                    print(f"Error loading mutation data for {gene}: {e}")
        
        if mutation_dfs:
            # Merge all mutation data on cell line identifiers
            self.mutation_data = mutation_dfs[0]
            for df in mutation_dfs[1:]:
                self.mutation_data = self.mutation_data.merge(
                    df, 
                    on=['depmap_id', 'cell_line_display_name', 'lineage_1', 
                        'lineage_2', 'lineage_3', 'lineage_6', 'lineage_4'],
                    how='outer'
                )
            
            print(f"\nTotal mutation profiles: {len(self.mutation_data)}")
            return self.mutation_data
        else:
            print("Warning: No mutation data found")
            return None
    
    def load_expression_data(self) -> pd.DataFrame:
        """
        Load gene expression data.
        
        Returns:
            Expression dataframe
        """
        expression_files = []
        
        for gene in self.genes:
            gene_path = self.drug_sensitivity_path / gene
            
            for expr_file in gene_path.glob("Expression*.csv"):
                try:
                    df = pd.read_csv(expr_file)
                    
                    # Rename gene column to indicate expression
                    if gene in df.columns:
                        df = df.rename(columns={gene: f'{gene}_expression'})
                    
                    expression_files.append(df)
                    print(f"Loaded expression data for {gene}: {len(df)} cell lines")
                    
                except Exception as e:
                    print(f"Error loading expression data for {gene}: {e}")
        
        if expression_files:
            # Merge expression data
            self.expression_data = expression_files[0]
            for df in expression_files[1:]:
                self.expression_data = self.expression_data.merge(
                    df,
                    on=['depmap_id', 'cell_line_display_name', 'lineage_1', 
                        'lineage_2', 'lineage_3', 'lineage_6', 'lineage_4'],
                    how='outer'
                )
            
            print(f"\nTotal expression profiles: {len(self.expression_data)}")
            return self.expression_data
        else:
            print("Warning: No expression data found")
            return None
    
    def merge_all_data(self) -> pd.DataFrame:
        """
        Merge drug sensitivity, mutation, and expression data.
        
        Returns:
            Integrated multi-modal dataframe
        """
        print("\n" + "="*60)
        print("MERGING ALL DATA MODALITIES")
        print("="*60)
        
        if self.drug_sensitivity_data is None:
            raise ValueError("Load drug sensitivity data first!")
        
        merged = self.drug_sensitivity_data.copy()
        print(f"Starting with drug sensitivity: {len(merged)} records")
        
        # Merge mutation data
        if self.mutation_data is not None:
            before = len(merged)
            merged = merged.merge(
                self.mutation_data,
                on=['depmap_id', 'cell_line_display_name', 'lineage_1', 
                    'lineage_2', 'lineage_3', 'lineage_6', 'lineage_4'],
                how='left'
            )
            print(f"Merged mutation data: {len(merged)} records (gained {len(merged) - before} records)")
        
        # Merge expression data
        if self.expression_data is not None:
            before = len(merged)
            merged = merged.merge(
                self.expression_data,
                on=['depmap_id', 'cell_line_display_name', 'lineage_1', 
                    'lineage_2', 'lineage_3', 'lineage_6', 'lineage_4'],
                how='left'
            )
            print(f"Merged expression data: {len(merged)} records (gained {len(merged) - before} records)")
        
        self.merged_data = merged
        print(f"\nFINAL MERGED DATASET: {len(merged)} records")
        print(f"   Columns: {len(merged.columns)}")
        print("="*60)
        
        return self.merged_data
    
    def get_feature_columns(self) -> Dict[str, List[str]]:
        """
        Get lists of feature columns by type.
        
        Returns:
            Dictionary with feature column names
        """
        if self.merged_data is None:
            raise ValueError("Merge data first!")
        
        columns = self.merged_data.columns.tolist()
        
        features = {
            'mutation': [col for col in columns if '_mutation' in col],
            'expression': [col for col in columns if '_expression' in col],
            'metadata': ['depmap_id', 'cell_line_display_name', 'lineage_1', 
                        'lineage_2', 'lineage_3', 'lineage_6', 'lineage_4'],
            'target': ['target_gene', 'screen_type']
        }
        
        return features
    
    def prepare_ml_dataset(self, drop_missing: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning.
        
        Args:
            drop_missing: Whether to drop rows with missing values
        
        Returns:
            Tuple of (features_df, target_series)
        """
        if self.merged_data is None:
            raise ValueError("Merge data first!")
        
        print("\n" + "="*60)
        print("PREPARING ML DATASET")
        print("="*60)
        
        df = self.merged_data.copy()
        
        # Get feature columns
        feature_info = self.get_feature_columns()
        feature_cols = feature_info['mutation'] + feature_info['expression']
        
        print(f"Mutation features: {len(feature_info['mutation'])}")
        print(f"Expression features: {len(feature_info['expression'])}")
        print(f"Total features: {len(feature_cols)}")
        
        # Drop rows with missing target gene values
        original_len = len(df)
        df = df[df['target_gene'].notna()]
        print(f"\nRemoved {original_len - len(df)} rows with missing target gene")
        
        if drop_missing:
            original_len = len(df)
            df = df.dropna(subset=feature_cols)
            print(f"Removed {original_len - len(df)} rows with missing features")
        
        # Extract features and target
        X = df[feature_cols + ['lineage_1', 'lineage_2', 'lineage_3']]  # Include cancer type
        y = df['target_gene']  # Target gene as the label
        
        print(f"\nFINAL ML DATASET:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {len(X.columns)}")
        print(f"   Missing values: {X.isnull().sum().sum()}")
        print("="*60)
        
        return X, y, df
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of the dataset.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.merged_data is None:
            raise ValueError("Merge data first!")
        
        stats = {
            'total_samples': len(self.merged_data),
            'unique_cell_lines': self.merged_data['depmap_id'].nunique(),
            'cancer_types': self.merged_data['lineage_1'].nunique(),
            'screen_types': self.merged_data['screen_type'].value_counts().to_dict(),
            'target_genes': self.merged_data['target_gene'].value_counts().to_dict(),
            'missing_data_pct': (self.merged_data.isnull().sum() / len(self.merged_data) * 100).to_dict()
        }
        
        return stats


if __name__ == "__main__":
    # Test the data loader
    print("="*80)
    print("CANCER DRUG RESPONSE DATA LOADER - TEST")
    print("="*80)
    
    # Initialize loader
    loader = CancerDataLoader(base_path="data")
    
    # Load all data
    print("\nLOADING DRUG SENSITIVITY DATA...")
    drug_sens = loader.load_drug_sensitivity()
    
    print("\nLOADING MUTATION DATA...")
    mutations = loader.load_mutation_data()
    
    print("\nLOADING EXPRESSION DATA...")
    expressions = loader.load_expression_data()
    
    # Merge all data
    merged = loader.merge_all_data()
    
    # Get summary statistics
    stats = loader.get_summary_statistics()
    print("\nDATASET SUMMARY:")
    for key, value in stats.items():
        if isinstance(value, dict) and len(value) > 5:
            print(f"  {key}: {len(value)} categories")
        else:
            print(f"  {key}: {value}")
    
    print("\nData loading test complete!")
