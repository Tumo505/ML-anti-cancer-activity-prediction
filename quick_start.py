"""
Quick start script for CNN + GA hybrid system.
Run this to test the pipeline with reduced settings.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.train_pipeline import CNNGATrainingPipeline


def main():
    """Quick start with reduced settings for faster testing."""
    
    print("="*80)
    print("QUICK START - CNN + GA HYBRID SYSTEM")
    print("="*80)
    print("\nThis will run a quick test with reduced settings.")
    print("For full training, modify config in src/config.py\n")
    
    # Initialize pipeline
    pipeline = CNNGATrainingPipeline(
        data_path="data",
        output_dir="results"
    )
    
    # Quick test configuration
    cnn_config = {
        'conv_filters': [32, 64],         # Reduced layers
        'dense_units': [128, 64],         # Reduced units
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 20,                     # Reduced epochs for quick test
        'batch_size': 64                  # Larger batch for speed
    }
    
    ga_config = {
        'n_drugs': 50,                    # Reduced drug library
        'min_size': 2,
        'max_size': 4,
        'population_size': 30,            # Reduced population
        'n_generations': 10               # Reduced generations
    }
    
    print("\nCONFIGURATION:")
    print(f"   CNN Epochs: {cnn_config['epochs']}")
    print(f"   GA Generations: {ga_config['n_generations']}")
    print(f"   Expected time: 10-30 minutes (depending on hardware)")
    
    response = input("\nStart quick test? (y/n): ")
    
    if response.lower() == 'y':
        # Run pipeline
        pipeline.run_complete_pipeline(
            cnn_config=cnn_config,
            ga_config=ga_config
        )
        
        print("\n" + "="*80)
        print("QUICK TEST COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {pipeline.run_dir}")
        print("\nTo run full training with better performance:")
        print("   1. Open src/config.py")
        print("   2. Use FULL_TRAINING_CONFIG or RESEARCH_CONFIG")
        print("   3. Run: python src/train_pipeline.py")
        
    else:
        print("\nQuick test cancelled.")
        print("\nTo run manually:")
        print("   python src/train_pipeline.py")


if __name__ == "__main__":
    main()
