#!/usr/bin/env python3

"""
Enhanced test suite for the Keyword Spotter project.
Includes comprehensive testing for all components.
"""

import warnings
import pytest
import numpy as np
import tempfile
import os
from omegaconf import OmegaConf
from src import data, model, inference, train
from src.exception_handler import NotFoundError, ValueError, DirectoryError
warnings.filterwarnings('ignore')

cfg = OmegaConf.load('./config_dir/config.yaml')

class TestDataProcessing:
    """Test suite for data processing functionality."""

@pytest.fixture
    def mfcc(self) -> np.ndarray:
        """Fixture for MFCC features."""
    mfcc_features = data.convert_audio_to_mfcc(cfg.names.audio_file,
                                               cfg.params.n_mfcc,
                                               cfg.params.mfcc_length,
                                               cfg.params.sampling_rate)
    return mfcc_features

    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary audio file for testing."""
        # Create a dummy audio file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.write(b'dummy audio data')
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    def test_label_type(self):
        """Test that labels are strings."""
    labels = data.Preprocess().wrap_labels()
    assert all(isinstance(n, str) for n in labels)

    def test_mfcc_shape(self, mfcc):
        """Test MFCC feature shape."""
        assert mfcc.shape == (cfg.params.n_mfcc, cfg.params.mfcc_length)
    
    def test_mfcc_dimension(self, mfcc):
        """Test MFCC feature dimensions."""
        assert len(mfcc.shape) == 2
    
    def test_file_type_validation(self):
        """Test file type validation."""
        # Valid extensions
        assert data.check_fileType("test.wav", ".wav")
        assert data.check_fileType("test.mp3", [".wav", ".mp3", ".flac"])
        assert data.check_fileType("test.flac", ".wav,.mp3,.flac")
        
        # Invalid extensions
        assert not data.check_fileType("test.txt", ".wav")
        assert not data.check_fileType("test", ".wav")
        assert not data.check_fileType("", ".wav")
    
    def test_audio_validation(self, sample_audio_file):
        """Test audio file validation."""
        # Test with valid file
        result = data.validate_audio_file(sample_audio_file, max_size_mb=1)
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'message' in result
    
    def test_dataset_preprocessing(self):
        """Test dataset preprocessing."""
        dataset = data.Dataset()
        preprocess = data.Preprocess(dataset, cfg.paths.train_dir)
        
        # Test labels property
        labels = preprocess.labels
        assert isinstance(labels, list)
    
    def test_convert_audio_to_mfcc_edge_cases(self):
        """Test MFCC conversion with edge cases."""
        # Test with non-existent file
        with pytest.raises(Exception):
            data.convert_audio_to_mfcc("non_existent.wav", 49, 40, 16000)

class TestModel:
    """Test suite for model functionality."""
    
    def test_cnn_lstm_model_creation(self):
        """Test CNN-LSTM model creation."""
        model_instance = model.CNN_LSTM_Model(
            input_shape=(cfg.params.n_mfcc, cfg.params.mfcc_length),
            num_classes=25
        )
        
        created_model = model_instance.create_model()
        assert created_model is not None
        assert created_model.input_shape == (None, cfg.params.n_mfcc, cfg.params.mfcc_length)
        assert created_model.output_shape == (None, 25)
    
    def test_model_architecture(self):
        """Test model architecture components."""
        model_instance = model.CNN_LSTM_Model(
            input_shape=(cfg.params.n_mfcc, cfg.params.mfcc_length),
            num_classes=25
        )
        
        created_model = model_instance.create_model()
        
        # Check that model has expected layers
        layer_types = [type(layer).__name__ for layer in created_model.layers]
        
        # Should have Conv1D layers
        assert 'Conv1D' in layer_types
        # Should have LSTM layers
        assert 'LSTM' in layer_types
        # Should have Dense layers
        assert 'Dense' in layer_types
        # Should have BatchNormalization
        assert 'BatchNormalization' in layer_types

class TestInference:
    """Test suite for inference functionality."""
    
    def test_keyword_spotter_initialization(self):
        """Test KeywordSpotter initialization."""
        spotter = inference.KeywordSpotter(
            audio_file="dummy.wav",
            model_artifactory_dir=cfg.paths.model_artifactory_dir,
            n_mfcc=cfg.params.n_mfcc,
            mfcc_length=cfg.params.mfcc_length,
            sampling_rate=cfg.params.sampling_rate
        )
        
        assert spotter.audio_file == "dummy.wav"
        assert spotter.n_mfcc == cfg.params.n_mfcc
        assert spotter.mfcc_length == cfg.params.mfcc_length
        assert spotter.sampling_rate == cfg.params.sampling_rate

class TestExceptionHandling:
    """Test suite for exception handling."""
    
    def test_custom_exceptions(self):
        """Test custom exception classes."""
        # Test MLFlowError
        with pytest.raises(Exception):
            raise data.exception_handler.MLFlowError("Test error")
        
        # Test ValueError
        with pytest.raises(Exception):
            raise data.exception_handler.ValueError("Test error")
        
        # Test DirectoryError
        with pytest.raises(Exception):
            raise data.exception_handler.DirectoryError("Test error")
        
        # Test NotFoundError
        with pytest.raises(Exception):
            raise data.exception_handler.NotFoundError("Test error")

class TestDataValidation:
    """Test suite for data validation."""
    
    def test_audio_format_conversion(self):
        """Test audio format conversion."""
        # This would test the convert_audio_format function
        # For now, just test that it exists
        assert hasattr(data, 'convert_audio_format')
    
    def test_file_size_validation(self):
        """Test file size validation."""
        # Create a large dummy file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(b'x' * (11 * 1024 * 1024))  # 11MB
            temp_file_path = temp_file.name
        
        try:
            result = data.validate_audio_file(temp_file_path, max_size_mb=10)
            assert not result['valid']
            assert 'too large' in result['message'].lower()
        finally:
            os.unlink(temp_file_path)

class TestPerformance:
    """Performance tests."""
    
    def test_mfcc_processing_speed(self):
        """Test MFCC processing performance."""
        import time
        
        start_time = time.time()
        mfcc_features = data.convert_audio_to_mfcc(cfg.names.audio_file,
                                                   cfg.params.n_mfcc,
                                                   cfg.params.mfcc_length,
                                                   cfg.params.sampling_rate)
        processing_time = time.time() - start_time
        
        # Should process in reasonable time (less than 1 second for small files)
        assert processing_time < 1.0
        assert mfcc_features is not None
    
    def test_model_creation_speed(self):
        """Test model creation performance."""
        import time
        
        start_time = time.time()
        model_instance = model.CNN_LSTM_Model(
            input_shape=(cfg.params.n_mfcc, cfg.params.mfcc_length),
            num_classes=25
        )
        created_model = model_instance.create_model()
        creation_time = time.time() - start_time
        
        # Should create model in reasonable time
        assert creation_time < 5.0
        assert created_model is not None

class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_integration(self):
        """Test full pipeline integration."""
        # This would test the complete pipeline from data loading to prediction
        # For now, test that all components can be imported and initialized
        
        # Test data components
        dataset = data.Dataset()
        preprocess = data.Preprocess(dataset, cfg.paths.train_dir)
        
        # Test model components
        model_instance = model.CNN_LSTM_Model(
            input_shape=(cfg.params.n_mfcc, cfg.params.mfcc_length),
            num_classes=25
        )
        
        # Test inference components
        spotter = inference.KeywordSpotter(
            audio_file="dummy.wav",
            model_artifactory_dir=cfg.paths.model_artifactory_dir,
            n_mfcc=cfg.params.n_mfcc,
            mfcc_length=cfg.params.mfcc_length,
            sampling_rate=cfg.params.sampling_rate
        )
        
        # All components should initialize without errors
        assert dataset is not None
        assert preprocess is not None
        assert model_instance is not None
        assert spotter is not None

# Performance benchmarks
class TestBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_mfcc_extraction_benchmark(self, benchmark):
        """Benchmark MFCC extraction performance."""
        result = benchmark(
            data.convert_audio_to_mfcc,
            cfg.names.audio_file,
            cfg.params.n_mfcc,
            cfg.params.mfcc_length,
            cfg.params.sampling_rate
        )
        assert result is not None
    
    @pytest.mark.benchmark
    def test_model_creation_benchmark(self, benchmark):
        """Benchmark model creation performance."""
        def create_model():
            model_instance = model.CNN_LSTM_Model(
                input_shape=(cfg.params.n_mfcc, cfg.params.mfcc_length),
                num_classes=25
            )
            return model_instance.create_model()
        
        result = benchmark(create_model)
        assert result is not None

# Fixtures for test data
@pytest.fixture(scope="session")
def test_config():
    """Session-scoped fixture for test configuration."""
    return cfg

@pytest.fixture
def sample_labels():
    """Fixture for sample labels."""
    return ["yes", "no", "stop", "go", "up", "down", "left", "right"]

@pytest.fixture
def sample_audio_data():
    """Fixture for sample audio data."""
    return np.random.randn(cfg.params.n_mfcc, cfg.params.mfcc_length)