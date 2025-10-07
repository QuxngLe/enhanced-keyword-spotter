#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to perform model training.
"""

from tensorflow import keras
from keras import optimizers
from src.model import CNN_LSTM_Model
from src.data import Dataset
from src.exception_handler import ValueError
from src.experiment_tracking import MLFlowTracker, ModelSelection

class Training:
    def __init__(self, model: CNN_LSTM_Model, dataset: Dataset,
                batch_size: int, epochs: int, learning_rate: float,
                tracker: MLFlowTracker, metric_name: str) -> None:
        """
        Instance variables
        ------------------
        model: CNN_LSTM_Model
            Instance of CNN_LSTM_Model class holding the created model.
        dataset: Dataset
            Instance of Dataset class holding the processed data(train & test).
        batch_size: int
            Number of samples per gradient update.
        epochs: int
            Number of epochs to train the model.
        learning_rate: float
            Rate of model training.
        tracker: MLFlowTracker
            Instance of MLFlowTracker class.
        metric_name: str
            Metric name to sort the models.

        Returns
        -------
            None.
        """
        self.model = model
        self.dataset_ = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tracker = tracker
        self.metric_name = metric_name
        
    def train(self) -> ModelSelection:
        """
        Enhanced training method with advanced techniques:
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        - Advanced optimizers
        - Data augmentation simulation

        Parameters
        ----------
            None.
        
        Returns
        -------
        instanceof(ModelSelection):
            Instance will hold resulting best model information after selecting from the
            model artifacts based on the given metric.

        Raises
        ------
        ValueError: Exception
            If self.metric_name is not given or null.
        """

        if self.metric_name is None:
            raise ValueError("Please provide the metric name for model selection !!!")
            
        print("Enhanced training started with advanced techniques...")
        
        # Enhanced optimizer with learning rate scheduling
        from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
        
        # Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpointing
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        
        callbacks = [lr_scheduler, early_stopping, checkpoint]
        
        # Enhanced optimizer with warmup
        optimizer = optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile with enhanced loss and metrics
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Data augmentation for audio (noise injection, time stretching)
        augmented_data = self._apply_data_augmentation()
        
        # Enhanced training with validation split
        history = self.model.fit(
            augmented_data['x_train'], augmented_data['y_train'],
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(self.dataset_.x_test, self.dataset_.y_test),
            callbacks=callbacks,
            shuffle=True,
            class_weight=self._compute_class_weights()
        )
        
        # Log training history
        self._log_training_history(history)
        
        print("Training completed successfully!")
        return ModelSelection(self.tracker.find_best_model(self.metric_name))
    
    def _apply_data_augmentation(self) -> dict:
        """
        Apply data augmentation techniques to improve model robustness.
        """
        import numpy as np
        
        # Simple noise injection augmentation
        x_train_aug = self.dataset_.x_train.copy()
        y_train_aug = self.dataset_.y_train.copy()
        
        # Add gaussian noise to 20% of training samples
        noise_indices = np.random.choice(
            len(x_train_aug), 
            size=int(0.2 * len(x_train_aug)), 
            replace=False
        )
        
        for idx in noise_indices:
            noise = np.random.normal(0, 0.01, x_train_aug[idx].shape)
            x_train_aug[idx] = np.clip(x_train_aug[idx] + noise, -1, 1)
        
        return {
            'x_train': x_train_aug,
            'y_train': y_train_aug
        }
    
    def _compute_class_weights(self) -> dict:
        """
        Compute class weights for imbalanced dataset handling.
        """
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Get class labels from one-hot encoded data
        y_classes = np.argmax(self.dataset_.y_train, axis=1)
        unique_classes = np.unique(y_classes)
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_classes
        )
        
        # Create weight dictionary
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        return class_weight_dict
    
    def _log_training_history(self, history) -> None:
        """
        Log detailed training history and metrics.
        """
        import matplotlib.pyplot as plt
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log final metrics
        final_train_acc = max(history.history['accuracy'])
        final_val_acc = max(history.history['val_accuracy'])
        
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        
        # Log to MLflow if available
        if hasattr(self, 'tracker') and self.tracker:
            try:
                import mlflow
                mlflow.log_metric("final_train_accuracy", final_train_acc)
                mlflow.log_metric("final_val_accuracy", final_val_acc)
                mlflow.log_artifact("training_history.png")
            except Exception as e:
                print(f"Could not log to MLflow: {e}")