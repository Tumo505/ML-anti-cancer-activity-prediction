"""
CNN model architecture for drug response prediction.
Multi-task learning with efficacy and toxicity prediction heads.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, callbacks
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class DrugResponseCNN:
    """
    Convolutional Neural Network for predicting drug response.
    Designed for multi-modal cancer data (mutations + expression + metadata).
    """
    
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        conv_filters: List[int] = [64, 128, 256],
        dense_units: List[int] = [256, 128],
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001
    ):
        """
        Initialize CNN architecture.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of target classes
            conv_filters: List of filter sizes for conv layers
            dense_units: List of units for dense layers
            dropout_rate: Dropout probability
            l2_reg: L2 regularization strength
        """
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.model = None
        self.history = None
        
    def build_model(self, multi_task: bool = False) -> Model:
        """
        Build the CNN architecture.
        
        Args:
            multi_task: Whether to create multi-task outputs
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input_features')
        
        # Reshape for 1D convolution
        x = layers.Reshape((self.input_dim, 1))(inputs)
        
        # Convolutional layers
        for i, filters in enumerate(self.conv_filters):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=3,
                padding='same',
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'conv_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            x = layers.Activation('relu', name=f'relu_conv_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_conv_{i+1}')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pooling')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(
                units,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = layers.Activation('relu', name=f'relu_dense_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        # Output layer(s)
        if multi_task:
            # Multi-task outputs
            efficacy_output = layers.Dense(
                self.n_classes,
                activation='softmax',
                name='efficacy_prediction'
            )(x)
            
            toxicity_output = layers.Dense(
                1,
                activation='sigmoid',
                name='toxicity_prediction'
            )(x)
            
            outputs = [efficacy_output, toxicity_output]
        else:
            # Single task classification
            outputs = layers.Dense(
                self.n_classes,
                activation='softmax',
                name='drug_response'
            )(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='DrugResponseCNN')
        
        return self.model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = 'sparse_categorical_crossentropy',
        metrics: List[str] = ['accuracy']
    ):
        """
        Compile the model with optimizer and loss.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            loss: Loss function
            metrics: List of metrics to track
        """
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0  # Add gradient clipping to prevent explosion
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print(f"Model compiled successfully (lr={learning_rate}, gradient clipping enabled)")
    
    def get_callbacks(
        self,
        model_checkpoint_path: str,
        early_stopping_patience: int = 20,
        reduce_lr_patience: int = 10
    ) -> List[callbacks.Callback]:
        """
        Get training callbacks.
        
        Args:
            model_checkpoint_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
        
        Returns:
            List of callback objects
        """
        callback_list = [
            callbacks.ModelCheckpoint(
                model_checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1
            )
        ]
        
        return callback_list
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        model_checkpoint_path: str = 'models/best_model.h5',
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            model_checkpoint_path: Path to save model
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        print("\n" + "="*80)
        print("TRAINING CNN MODEL")
        print("="*80)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print("="*80)
        
        # Get callbacks
        callback_list = self.get_callbacks(
            model_checkpoint_path=model_checkpoint_path
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        print("\nTraining complete!")
        
        return self.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*80)
        print("EVALUATING MODEL")
        print("="*80)
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
            print(f"{metric_name}: {results[i]:.4f}")
        
        print("="*80)
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            batch_size: Batch size for prediction
        
        Returns:
            Predictions array
        """
        return self.model.predict(X, batch_size=batch_size, verbose=0)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        if self.history is None:
            print("Warning: No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            print("Warning: Model not built yet")
            return
        
        self.model.summary()
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("CNN MODEL - TEST")
    print("="*80)
    
    # Create dummy data
    n_samples = 1000
    n_features = 50
    n_classes = 8
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)
    X_val = np.random.randn(200, n_features)
    y_val = np.random.randint(0, n_classes, 200)
    
    # Initialize and build model
    print("\n1. Building model...")
    model = DrugResponseCNN(
        input_dim=n_features,
        n_classes=n_classes,
        conv_filters=[32, 64],
        dense_units=[128, 64],
        dropout_rate=0.3
    )
    
    model.build_model()
    model.compile_model()
    model.summary()
    
    print("\nCNN model test complete!")
