#!/usr/bin/env python3
"""
Quick test script to verify all dependencies are installed correctly.
"""

def test_imports():
    """Test if all required packages can be imported."""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow: {e}")
    
    try:
        import flask
        print(f"✅ Flask: {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask: {e}")
    
    try:
        import librosa
        print(f"✅ Librosa: {librosa.__version__}")
    except ImportError as e:
        print(f"❌ Librosa: {e}")
    
    try:
        import mlflow
        print(f"✅ MLflow: {mlflow.__version__}")
    except ImportError as e:
        print(f"❌ MLflow: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")

def test_model_creation():
    """Test if we can create the model."""
    try:
        from src.model import CNN_LSTM_Model
        model = CNN_LSTM_Model(input_shape=(49, 40), num_classes=25)
        print("✅ Model creation: Success")
        return True
    except Exception as e:
        print(f"❌ Model creation: {e}")
        return False

def test_config_loading():
    """Test if configuration can be loaded."""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load('./config_dir/config.yaml')
        print("✅ Configuration loading: Success")
        return True
    except Exception as e:
        print(f"❌ Configuration loading: {e}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced Keyword Spotter Setup...")
    print("=" * 50)
    
    print("\nTesting Package Imports:")
    test_imports()
    
    print("\nTesting Model Creation:")
    test_model_creation()
    
    print("\nTesting Configuration:")
    test_config_loading()
    
    print("\nSetup test complete!")
    print("\nIf all tests pass, you can run: python app.py")
