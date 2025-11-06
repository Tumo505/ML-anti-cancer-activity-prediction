"""
Configuration file for CNN + GA training pipeline.
Modify these settings to customize the training process.
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    'base_path': 'data',
    'screen_types': ['GDSC1', 'GDSC2', 'CTD2', 'PRISM'],  # Screening platforms to include
    'test_size': 0.2,  # Proportion of data for testing
    'val_size': 0.1,   # Proportion of data for validation
    'random_state': 42  # Random seed for reproducibility
}

# ============================================================================
# CNN MODEL CONFIGURATION
# ============================================================================

CNN_CONFIG = {
    # Architecture
    'conv_filters': [64, 128, 256],  # Convolutional layer filters
    'dense_units': [256, 128],        # Dense layer units
    'dropout_rate': 0.3,              # Dropout probability
    'l2_reg': 0.001,                  # L2 regularization strength
    
    # Training
    'learning_rate': 0.001,           # Learning rate for Adam optimizer
    'epochs': 100,                    # Maximum number of epochs
    'batch_size': 32,                 # Batch size
    
    # Callbacks
    'early_stopping_patience': 20,    # Early stopping patience
    'reduce_lr_patience': 10,         # LR reduction patience
    'model_checkpoint': 'models/best_cnn_model.h5'
}

# ============================================================================
# GENETIC ALGORITHM CONFIGURATION
# ============================================================================

GA_CONFIG = {
    # Drug combination parameters
    'n_drugs': 100,                   # Number of drugs in library
    'min_combination_size': 2,        # Minimum drugs per combination
    'max_combination_size': 5,        # Maximum drugs per combination
    
    # Fitness weights
    'efficacy_weight': 0.7,           # Weight for efficacy (0-1)
    'toxicity_weight': 0.3,           # Weight for toxicity penalty (0-1)
    
    # GA parameters
    'population_size': 100,           # Population size
    'n_generations': 50,              # Number of generations
    'crossover_prob': 0.7,            # Crossover probability
    'mutation_prob': 0.2,             # Mutation probability
    'tournament_size': 3,             # Tournament selection size
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

PREPROCESSING_CONFIG = {
    'missing_value_strategy': 'median',  # 'mean', 'median', or 'zero'
    'categorical_encoding': 'onehot',     # 'onehot' or 'label'
    'scaling': True,                      # Whether to scale features
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    'results_dir': 'results',
    'models_dir': 'models',
    'save_predictions': True,
    'save_plots': True,
    'verbose': True
}

# ============================================================================
# QUICK CONFIGURATIONS FOR DIFFERENT SCENARIOS
# ============================================================================

# Fast testing configuration (reduced complexity)
QUICK_TEST_CONFIG = {
    'cnn': {
        'conv_filters': [32, 64],
        'dense_units': [128, 64],
        'epochs': 10,
        'batch_size': 64
    },
    'ga': {
        'population_size': 50,
        'n_generations': 20
    }
}

# Full training configuration (maximum performance)
FULL_TRAINING_CONFIG = {
    'cnn': {
        'conv_filters': [128, 256, 512],
        'dense_units': [512, 256, 128],
        'dropout_rate': 0.4,
        'epochs': 200,
        'batch_size': 16
    },
    'ga': {
        'population_size': 200,
        'n_generations': 100
    }
}

# Research configuration (balanced)
RESEARCH_CONFIG = {
    'cnn': {
        'conv_filters': [64, 128, 256],
        'dense_units': [256, 128],
        'dropout_rate': 0.3,
        'epochs': 100,
        'batch_size': 32
    },
    'ga': {
        'population_size': 100,
        'n_generations': 50
    }
}
