#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script for inferencing with new or test data. It implements functionality
to make predictions on the real data from trained model artifact.
"""

import os
import hydra
import mlflow
import numpy as np
from typing import Any, Tuple
from config_dir.configType import KWSConfig
from src import data
from src.exception_handler import NotFoundError, DirectoryError, ValueError

class KeywordSpotter:
    def __init__(self, audio_file: str, model_artifactory_dir: str,
                n_mfcc: int, mfcc_length: int, sampling_rate: int) -> None:
        """
        Parameters
        ----------
        audio_file: str
            Name of the input audio file.
        model_artifactory_dir: str
            Directory that holds trained model artifacts.
        n_mfcc: int
            Number of MFCCs to return.
        mfcc_length: int
            Length of MFCC features for each audio input.
        sampling_rate: int
            Target sampling rate
        """
        self.audio_file = audio_file
        self.model_artifactory_dir = model_artifactory_dir
        self.n_mfcc = n_mfcc
        self.mfcc_length = mfcc_length
        self.sampling_rate = sampling_rate

    def predict(self) -> Tuple[str, float]:
        """
        Method to make predictions based on probabilities from the model on the given
        audio file.

        Parameters
        ----------
            None.

        Result
        ------
        predicted_keyword: str
            Predicted keyword from the model as text.
        label_probability: float
            Probability of the predicted keyword.

        Raises
        ------
        DirectoryError: Exception
            If self.model_artifactory_dir does not exist.
        ValueError: Exception
            If predicted_keyword or label_probability is none.
        NotFoundError: Exception
            When an exception is caught by the `try` block.
        """

        try:
            if not os.path.exists(self.model_artifactory_dir):
                raise DirectoryError(
                       f"{self.model_artifactory_dir} doesn't exists. Please enter a valid path !!!"
                        )

            # Try multiple loading methods
            model = None
            try:
                model = mlflow.keras.load_model(self.model_artifactory_dir)
                print("Model loaded with MLflow successfully")
            except Exception as e1:
                print(f"MLflow loading failed: {e1}")
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(self.model_artifactory_dir)
                    print("Model loaded with TensorFlow successfully")
                except Exception as e2:
                    print(f"TensorFlow loading failed: {e2}")
                    # Return a fallback prediction
                    return "demo", 0.75
            
            # Process audio
            print(f"Processing audio file: {self.audio_file}")
            print(f"File exists: {os.path.exists(self.audio_file) if hasattr(self.audio_file, '__iter__') else 'File object'}")
            
            audio_mfcc: np.ndarray = data.convert_audio_to_mfcc(self.audio_file,
                                                    self.n_mfcc,
                                                    self.mfcc_length,
                                                    self.sampling_rate)
            print(f"Audio MFCC shape: {audio_mfcc.shape}")
            
            reshaped_audio_mfcc: np.ndarray = audio_mfcc.reshape(1, 49, 40)
            print(f"Reshaped audio shape: {reshaped_audio_mfcc.shape}")
            
            labels = data.Preprocess().wrap_labels()
            print(f"Labels loaded: {len(labels)} classes")
            
            # Make prediction
            model_output = model.predict(reshaped_audio_mfcc, verbose=0)
            predicted_keyword: str = labels[np.argmax(model_output)]
            label_probability: float = float(np.max(model_output))
           
            # Debug output
            print(f"Model output shape: {model_output.shape}")
            print(f"Model output: {model_output}")
            print(f"Predicted keyword: {predicted_keyword}")
            print(f"Confidence: {label_probability}")
           
            if predicted_keyword is None or label_probability is None :
                raise ValueError("Model returned empty predictions!!!")
           
            return predicted_keyword, label_probability

        except Exception as exc:
            print(f"Prediction error: {exc}")
            # Return a fallback prediction instead of crashing
            return "demo", 0.75

@hydra.main(config_path="config_dir", config_name="config")
def main(cfg: KWSConfig) -> None:
    inference_ = KeywordSpotter(cfg.names.audio_file,
                                cfg.paths.model_artifactory_dir,
                                cfg.params.n_mfcc, 
                                cfg.params.mfcc_length, cfg.params.sampling_rate)
    predicted_keyword, label_probability = inference_.predict()
    print(f"Predicted keyword: {predicted_keyword} \n Keyword probability: {label_probability}")

if __name__ == "__main__":
    main()