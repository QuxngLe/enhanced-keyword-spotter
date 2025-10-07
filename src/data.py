#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Load, preprocess audio dataset and extract features. The audio files are loaded,
each file is preprocessed, and dumped as `.npy` files which is convenient to work
with. Thus, dumped `.npy` files can be loaded and performed with some additional
preprocessing steps and can be used for training.
"""

import os
import librosa
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from keras.utils import to_categorical
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception_handler import DirectoryError, ValueError

@dataclass
class Dataset:
    """
    Dataclass that represent a dataset which is flexible to be used
    for any model training.
    """
    x_train: np.ndarray = None
    y_train: np.array = None
    x_test: np.ndarray = None
    y_test: np.array = None

@dataclass
class Preprocess:
    """Preprocess audio dataset to be used for training.
    """
    dataset_: Dataset = None
    train_dir: str = "./dataset/train/"
    n_mfcc: int = 49
    mfcc_length: int = 40
    sampling_rate: int = 8000
    extension: str = ".npy"

    def __post_init__(self) -> None:
        """
        Dunder method to perform exception handling to catch invalid directory.

        Returns:
            None.

        Raises
        ------
        DirectoryError: Exception
            If self.train_dir does not exist.
        """
        if not os.path.exists(self.train_dir):
            raise DirectoryError(
                f"{self.train_dir} doesn't exists. Please enter a valid path !!!")

    @property
    def labels(self) -> List:
        """
        Class property to return the labels from data.

        Returns
        -------
        List of labels.
        """
        return ['.'.join(file_.split('.')[:-1]) 
               for file_ in os.listdir(self.train_dir) 
               if os.path.isfile(os.path.join(self.train_dir, file_)) 
               and check_fileType(filename = file_, extension = self.extension)]

    def __load_dataset(self, labels: List,
                       load_format: str = ".npy") -> Tuple[np.ndarray]:
        """
        Private method to load `.npy` files to preprocess.

        Parameters
        ----------
        labels: List
            List of labels.
        load_format: str
            Format to load from disk. Defaults to `.npy`.

        Returns
        -------
        data, labels: Tuple[np.ndarray]
            Tuple representing data(X) and its labels(y).
        """
        data = np.load(f"{self.train_dir + labels[0] + load_format}")
        labels = np.zeros(data.shape[0])
        for index, label in enumerate(self.labels[1:]):
            x = np.load(f"{self.train_dir + label + load_format}")
            data = np.vstack((data, x))
            labels = np.append(labels, np.full(x.shape[0], 
                          fill_value = (index + 1)))

        return data, labels
   
    def preprocess_dataset(self, labels: List,
                            test_split_percent: float) -> Dataset:
        """
        Preprocess the loaded dataset.

        Parameters
        ----------
        labels: List
            List of labels.
        test_split_percent: float
            Train-test split percentage/ratio.

        Returns
        -------
        instanceof(Dataset):
            Instance of Dataset after preprocessing.
            The labels are one-hot encoded.

        Raises
        ------
        ValueError: Exception
            If loaded dataset is empty or null.
        """

        X, y = self.__load_dataset(labels)
        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                           test_size = test_split_percent,
                                           random_state=42, shuffle = True)

        for data in (x_train, x_test, y_train, y_test):
            if data is None:
                raise ValueError(f"{data} is null. Please check and preprocess again!!!")

        return Dataset(x_train, to_categorical(y_train, num_classes = len(labels)),
                       x_test, to_categorical(y_test, num_classes = len(labels)))

  
    def dump_audio_files(self, audio_files_dir: str, labels: List, n_mfcc: int,
                        mfcc_length: int, sampling_rate: int,
                        save_format: str = ".npy") -> None:
        """
        Method to load, process and dump audio files as `.npy` for training.
        This method is `optional` and used only with audio files. If not, skip this
        method and use preprocess_dataset() directly for training.

        Returns
        -------
            None.
        """
        for label in labels:
            mfcc_features_np = list()
            audio_files = [audio_files_dir + label + '/' + audio_file 
                           for audio_file in os.listdir(audio_files_dir + '/' + label)]
            for audioFile in tqdm(audio_files):
                mfcc_features = convert_audio_to_mfcc(audioFile, n_mfcc, 
                                                      mfcc_length, sampling_rate)
                mfcc_features_np.append(mfcc_features)
            np.save(f"{self.train_dir + label + save_format}", mfcc_features_np)

        print(f".npy files dumped to {self.train_dir}")

    def wrap_labels(self) -> List:
        """
        Wrapper funtion to read labels from file.
       
        This is not a generic approach but it's required for inference. The main reason is,
        due to the memory limitation in Git, large files cannot be added. Even though Git LFS
        can be used but it's not feasible for this current application. So this function is
        a little play around.

        Returns:
        --------
            labels: List
        """
        with open(f"{self.train_dir}/labels.txt", "r") as file:
            file_data: str = file.read()
            labels: List = file_data.split(",")
            file.close()
            return labels

def convert_audio_to_mfcc(audio_file_path: str,
                          n_mfcc: int, mfcc_length: int,
                          sampling_rate: int) -> np.ndarray:
    """
    Helper function to convert each audio file to MFCC features. It's
    a generic function which can be called without an instance.

    Parameters
    ----------
    audio_file_path: str
        Path of audio file.
    n_mfcc: int
        Number of MFCCs to return.
    mfcc_length: int
        Length of MFCC features for each audio input.
    sampling_rate: int
        Target sampling rate.

    Returns
    -------
    mfcc_features: np.ndarray
        Extracted MFCC features of the audio file.
    """
    audio, sampling_rate = librosa.load(audio_file_path, sr = sampling_rate)
    mfcc_features: np.ndarray = librosa.feature.mfcc(audio,
                                                     n_mfcc = n_mfcc,
                                                     sr = sampling_rate)
    if(mfcc_length > mfcc_features.shape[1]):
        padding_width = mfcc_length - mfcc_features.shape[1]
        mfcc_features = np.pad(mfcc_features, 
                              pad_width =((0, 0), (0, padding_width)), mode ='constant')
    else:
        mfcc_features = mfcc_features[:, :mfcc_length]
   
    return mfcc_features

def check_fileType(filename: str, extension: str) -> bool:
    """
    Enhanced helper function to check the extension of a file.
    Supports multiple audio formats and better validation.

    Parameters
    ----------
    filename: str
        Input filename
    extension: str
        File extension to check. Can be single extension or list.

    Returns
    -------
        bool: True if file has valid extension, else False.
    """
    if not filename or '.' not in filename:
        return False
    
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    # Handle multiple extensions
    if isinstance(extension, list):
        return file_ext in [ext.lower().lstrip('.') for ext in extension]
    elif isinstance(extension, str):
        # Support comma-separated extensions
        if ',' in extension:
            extensions = [ext.strip().lower().lstrip('.') for ext in extension.split(',')]
            return file_ext in extensions
        else:
            return file_ext == extension.lower().lstrip('.')
    
    return False

def validate_audio_file(file_path: str, max_size_mb: int = 10) -> dict:
    """
    Comprehensive audio file validation.
    
    Parameters
    ----------
    file_path: str
        Path to the audio file
    max_size_mb: int
        Maximum file size in MB
        
    Returns
    -------
    dict: Validation result with status and message
    """
    import os
    
    validation_result = {
        'valid': False,
        'message': '',
        'file_size': 0,
        'duration': 0
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            validation_result['message'] = 'File does not exist'
            return validation_result
        
        # Check file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        validation_result['file_size'] = file_size_mb
        
        if file_size_mb > max_size_mb:
            validation_result['message'] = f'File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)'
            return validation_result
        
        # Check file extension
        valid_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        if not check_fileType(os.path.basename(file_path), valid_extensions):
            validation_result['message'] = 'Unsupported audio format'
            return validation_result
        
        # Try to load with librosa to validate audio format
        try:
            import librosa
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr
            validation_result['duration'] = duration
            
            # Check duration (should be reasonable for keyword spotting)
            if duration > 10:  # 10 seconds max
                validation_result['message'] = f'Audio too long: {duration:.1f}s (max: 10s)'
                return validation_result
            
            if duration < 0.1:  # 0.1 seconds min
                validation_result['message'] = f'Audio too short: {duration:.1f}s (min: 0.1s)'
                return validation_result
                
        except Exception as e:
            validation_result['message'] = f'Invalid audio file: {str(e)}'
            return validation_result
        
        validation_result['valid'] = True
        validation_result['message'] = 'File validation successful'
        
    except Exception as e:
        validation_result['message'] = f'Validation error: {str(e)}'
    
    return validation_result

def convert_audio_format(input_path: str, output_path: str, target_format: str = 'wav') -> bool:
    """
    Convert audio file to target format using librosa.
    
    Parameters
    ----------
    input_path: str
        Path to input audio file
    output_path: str
        Path for output file
    target_format: str
        Target format (wav, mp3, flac)
        
    Returns
    -------
    bool: True if conversion successful, False otherwise
    """
    try:
        import librosa
        import soundfile as sf
        
        # Load audio
        audio, sr = librosa.load(input_path, sr=16000)  # Standardize to 16kHz
        
        # Save in target format
        sf.write(output_path, audio, sr, format=target_format.upper())
        
        return True
        
    except Exception as e:
        print(f"Audio conversion failed: {str(e)}")
        return False


def print_shape(name: str, arr: np.array) -> None:
    """
    Helper function to print shapes of input np arrays.

    Note:
        To avoid boilerplate code!!!!

    Parameters
    ----------
    name: str
        Name of input array.
    arr: np.array
        Input array itself.

    Returns
    -------
        None
    """
    print(f"Shape of {name}: {arr.shape}")