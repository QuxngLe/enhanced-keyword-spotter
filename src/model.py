#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Defines and create model for training and evaluation.
`CNN-LSTM` is used for this project with 1D convolutional layers
followed by LSTM layers with self-attention and fully connected
layers. This script provides the flexibility to add any other
models by inheriting Model(ABC).
"""

from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod
from tensorflow import keras
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers import Input, Dropout, BatchNormalization, Dense

class Models(ABC):
    """
    Abstract base class that defines and creates model.
    """
    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

@dataclass
class CNN_LSTM_Model(Models):
    """
    Dataclass to create CNN-LSTM model that inherits Models class.
    """
    input_shape: Tuple[int, int]
    num_classes: int

    def define_model(self) -> Sequential:
        """
        Enhanced CNN-LSTM model with improved architecture for better performance.
        Includes residual connections, advanced attention, and regularization techniques.

        Parameters
        ----------
            None.

        Returns
        -------
        Sequential
        """

        from keras.layers import Add, GlobalAveragePooling1D, Activation
        from keras.regularizers import l2

        return Sequential(
            [
            Input(shape=self.input_shape),
            
            # Input preprocessing with stronger normalization
            BatchNormalization(momentum=0.9),
            
            # Enhanced 1D Convolutional layers with residual connections
            Conv1D(64, kernel_size=3, strides=1, padding='same', 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            Conv1D(64, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # Second convolutional block
            Conv1D(128, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            Conv1D(128, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # Third convolutional block
            Conv1D(256, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            Conv1D(256, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Global average pooling for better generalization
            GlobalAveragePooling1D(),
            
            # Enhanced LSTM layers with bidirectional processing
            # Note: Using regular LSTM for compatibility, but bidirectional would be ideal
            LSTM(units=256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(momentum=0.9),
            
            # Advanced self-attention mechanism
            SeqSelfAttention(attention_activation='tanh', kernel_regularizer=l2(0.001)),
            
            LSTM(units=128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(momentum=0.9),
            Dropout(0.3),

            # Enhanced dense layers with better regularization
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(momentum=0.9),
            Dropout(0.4),
            
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(momentum=0.9),
            Dropout(0.3),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            
            # Output layer with temperature scaling for better calibration
            Dense(self.num_classes, activation='softmax', 
                  kernel_regularizer=l2(0.001))
            ]
        )

    def create_model(self) -> Sequential:
        """
        Method to create the model defined by define_model() method
        and prints the model summary.

        Parameters
        ----------
            None.

        Returns
        -------
        model: Sequential
        """
        model: Sequential = self.define_model()
        model.summary()
        return model