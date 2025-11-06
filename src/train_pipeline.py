"""
Main training pipeline for CNN + GA hybrid system.
Integrates data loading, preprocessing, CNN training, and GA optimization.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import CancerDataLoader
from data_preprocessing import DataPreprocessor
from cnn_model import DrugResponseCNN
from genetic_algorithm import GeneticAlgorithmOptimizer


class CNNGATrainingPipeline:
    """Complete training pipeline for drug combination optimization."""
    
    def __init__(self, data_path: str, output_dir: str = "results"):
        """
        Initialize training pipeline.
        
        Args:
            data_path: Path to data directory
            output_dir: Directory for saving outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = CancerDataLoader(data_path)
        self.preprocessor = DataPreprocessor()
        self.cnn_model = None
        self.ga_optimizer = None
        
        # Data containers
        self.data_splits = None
        self.feature_info = None
        self.merged_data = None
        
        print(f"Pipeline initialized. Output directory: {self.run_dir}")
    
    def load_and_merge_data(self):
        """Load and merge all data sources."""
        print("\n" + "="*80)
        print("STEP 1: LOADING AND MERGING DATA")
        print("="*80)
        
        # Load drug sensitivity data
        print("\n1. Loading drug sensitivity data...")
        self.data_loader.load_drug_sensitivity()
        
        # Load mutation data
        print("\n2. Loading mutation data...")
        self.data_loader.load_mutation_data()
        
        # Load expression data
        print("\n3. Loading expression data...")
        self.data_loader.load_expression_data()
        
        # Merge all data
        print("\n4. Merging all data modalities...")
        self.merged_data = self.data_loader.merge_all_data()
        
        # Get summary statistics
        stats = self.data_loader.get_summary_statistics()
        
        # Save summary
        summary_path = self.run_dir / "data_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA SUMMARY\n")
            f.write("="*80 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nData summary saved to {summary_path}")
        
        return self.merged_data
    
    def preprocess_data(self, test_size: float = 0.2, val_size: float = 0.1):
        """Preprocess data and create train/val/test splits."""
        print("\n" + "="*80)
        print("STEP 2: PREPROCESSING DATA")
        print("="*80)
        
        if self.merged_data is None:
            raise ValueError("Load and merge data first!")
        
        # Get feature columns
        feature_info = self.data_loader.get_feature_columns()
        
        # Prepare for ML (don't drop missing - preprocessor will handle it)
        X, y, df = self.data_loader.prepare_ml_dataset(drop_missing=False)
        
        # Identify feature types
        mutation_cols = feature_info['mutation']
        expression_cols = feature_info['expression']
        categorical_cols = ['lineage_1', 'lineage_2', 'lineage_3']
        
        # Create feature list
        feature_cols = mutation_cols + expression_cols
        
        # Preprocess
        self.data_splits, self.feature_info = self.preprocessor.prepare_for_training(
            df=df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            target_col='target_gene',
            test_size=test_size,
            val_size=val_size
        )
        
        # Save preprocessor
        preprocessor_path = self.run_dir / "preprocessor.pkl"
        self.preprocessor.save_preprocessor(str(preprocessor_path))
        
        return self.data_splits, self.feature_info
    
    def build_and_train_cnn(
        self,
        conv_filters: list = [64, 128, 256],
        dense_units: list = [256, 128],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """Build and train CNN model."""
        print("\n" + "="*80)
        print("STEP 3: BUILDING AND TRAINING CNN")
        print("="*80)
        
        if self.data_splits is None:
            raise ValueError("Preprocess data first!")
        
        # Initialize CNN
        self.cnn_model = DrugResponseCNN(
            input_dim=self.feature_info['n_features'],
            n_classes=self.feature_info['n_classes'],
            conv_filters=conv_filters,
            dense_units=dense_units,
            dropout_rate=dropout_rate
        )
        
        # Build and compile
        self.cnn_model.build_model(multi_task=False)
        self.cnn_model.compile_model(learning_rate=learning_rate)
        
        # Print model summary
        print("\nModel Architecture:")
        self.cnn_model.summary()
        
        # Train model
        model_path = str(self.run_dir / "best_cnn_model.h5")
        history = self.cnn_model.train(
            X_train=self.data_splits['X_train'],
            y_train=self.data_splits['y_train'],
            X_val=self.data_splits['X_val'],
            y_val=self.data_splits['y_val'],
            epochs=epochs,
            batch_size=batch_size,
            model_checkpoint_path=model_path
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = self.cnn_model.evaluate(
            X_test=self.data_splits['X_test'],
            y_test=self.data_splits['y_test']
        )
        
        # Save training history plot
        plot_path = self.run_dir / "training_history.png"
        self.cnn_model.plot_training_history(save_path=str(plot_path))
        
        # Save metrics
        metrics_path = self.run_dir / "test_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TEST SET METRICS\n")
            f.write("="*80 + "\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print(f"\nMetrics saved to {metrics_path}")
        
        return test_metrics
    
    def create_predictor_function(self):
        """Create predictor function for GA."""
        def predictor(drug_indices: list) -> tuple:
            """
            Predict efficacy and toxicity for a drug combination.
            
            Args:
                drug_indices: List of drug indices in combination
            
            Returns:
                Tuple of (efficacy_score, toxicity_score)
            """
            # Create feature vector for this combination
            # In real implementation, you would encode the combination properly
            # For now, simulate predictions
            
            n_features = self.feature_info['n_features']
            
            # Create a dummy feature vector (mean of training data)
            feature_vector = np.zeros((1, n_features))
            
            # Get prediction from CNN
            prediction = self.cnn_model.predict(feature_vector)[0]
            
            # Convert class probabilities to efficacy score
            efficacy = float(np.max(prediction))
            
            # Simulate toxicity based on combination size
            # In real implementation, this would come from a toxicity predictor
            toxicity = 0.1 + 0.05 * len(drug_indices)
            toxicity = min(1.0, toxicity)
            
            return efficacy, toxicity
        
        return predictor
    
    def optimize_combinations(
        self,
        n_drugs: int = 100,
        min_size: int = 2,
        max_size: int = 5,
        population_size: int = 100,
        n_generations: int = 50
    ):
        """Run genetic algorithm optimization."""
        print("\n" + "="*80)
        print("STEP 4: OPTIMIZING DRUG COMBINATIONS")
        print("="*80)
        
        if self.cnn_model is None:
            raise ValueError("Train CNN model first!")
        
        # Initialize GA
        self.ga_optimizer = GeneticAlgorithmOptimizer(
            n_drugs=n_drugs,
            min_combination_size=min_size,
            max_combination_size=max_size,
            population_size=population_size,
            n_generations=n_generations
        )
        
        # Set predictor
        predictor = self.create_predictor_function()
        self.ga_optimizer.set_predictor(predictor)
        
        # Run optimization
        best_combinations = self.ga_optimizer.optimize(verbose=True)
        
        # Export results
        results_path = self.run_dir / "optimized_combinations.csv"
        self.ga_optimizer.export_results(str(results_path))
        
        # Display top 10
        print("\nTOP 10 OPTIMIZED DRUG COMBINATIONS:")
        print("="*80)
        for i, combo in enumerate(best_combinations[:10], 1):
            print(f"{i}. Combination: {combo.drug_indices}")
            print(f"   Drugs: {len(combo.drug_indices)} | "
                  f"Efficacy: {combo.efficacy_score:.3f} | "
                  f"Toxicity: {combo.toxicity_score:.3f} | "
                  f"Fitness: {combo.fitness_score:.3f}")
            print()
        
        return best_combinations
    
    def run_complete_pipeline(
        self,
        cnn_config: dict = None,
        ga_config: dict = None
    ):
        """Run complete pipeline from data loading to optimization."""
        print("\n" + "="*80)
        print("COMPLETE CNN + GA PIPELINE")
        print("="*80)
        
        # Default configurations
        if cnn_config is None:
            cnn_config = {
                'conv_filters': [64, 128, 256],
                'dense_units': [256, 128],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32
            }
        
        if ga_config is None:
            ga_config = {
                'n_drugs': 100,
                'min_size': 2,
                'max_size': 5,
                'population_size': 100,
                'n_generations': 50
            }
        
        try:
            # Step 1: Load and merge data
            self.load_and_merge_data()
            
            # Step 2: Preprocess
            self.preprocess_data()
            
            # Step 3: Train CNN
            self.build_and_train_cnn(**cnn_config)
            
            # Step 4: Optimize combinations
            self.optimize_combinations(**ga_config)
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETE!")
            print(f"Results saved to: {self.run_dir}")
            print("="*80)
            
        except Exception as e:
            print(f"\nERROR: {e}")
            raise


def main():
    """Main execution function."""
    print("="*80)
    print("CNN + GA HYBRID SYSTEM FOR DRUG COMBINATION OPTIMIZATION")
    print("="*80)
    
    # Initialize pipeline
    pipeline = CNNGATrainingPipeline(
        data_path="data",
        output_dir="results"
    )
    
    # Configuration
    cnn_config = {
        'conv_filters': [64, 128, 256],
        'dense_units': [256, 128],
        'dropout_rate': 0.3,
        'learning_rate': 0.0001,  # Reduced to prevent NaN loss
        'epochs': 50,  # Reduced for testing
        'batch_size': 32
    }
    
    ga_config = {
        'n_drugs': 100,
        'min_size': 2,
        'max_size': 5,
        'population_size': 50,  # Reduced for testing
        'n_generations': 20  # Reduced for testing
    }
    
    # Run pipeline
    pipeline.run_complete_pipeline(
        cnn_config=cnn_config,
        ga_config=ga_config
    )


if __name__ == "__main__":
    main()
