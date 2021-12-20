from abc import abstractmethod, ABC
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import math
import random

from utils import scale_minmax, save_feat


class Extractor(ABC):
    """
    Abstract class for Feature extraction implementing the following method: extract()
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def extract(self, data):
        pass


class MelSpectrogramExtractor(Extractor):
    """
    Class extending the Extractor and implementing the mel spectrogram feature extraction
    """

    def __init__(self, config: object):
        """
        Constructor for the MelSpectrogramExtractor
        :param config: Object containing the parameters related to feature extraction
        """
        super().__init__(config)
        self.config = config

    def extract(self, data: dict) -> dict:
        """
        Method to implement the Mel spectrogram feature extraction
        :param data: Dictionary containing the train/test audio files and labels
        :return: feat: Dictionary containing the mel spectrogram features for the train/test set
        """
        feat = {
            "train_audio_files": [],
            "train_labels": [],
            "train_feature": [],
            "test_audio_files": [],
            "test_labels": [],
            "test_feature": []
        }
        num_mels = self.config.melspectrogram.num_mels
        hop_length = self.config.hop_length
        n_fft = self.config.n_fft
        # Feature extraction for the training set
        for index, file_name in enumerate(data["train_audio_files"]):
            labels = data["train_labels"][index]
            # Take only the first 120 seconds of the audio file. This is done to tackle the different lengths of audio files as well as to reduce the computational load.
            signal, sr = librosa.load(file_name, sr=self.config.sampling_rate, duration=self.config.duration)
            # Discard the track that is less than 120 seconds
            if librosa.get_duration(y=signal, sr=sr) < self.config.duration:
                continue
            track_duration = self.config.duration
            samples_per_track = self.config.sampling_rate * track_duration
            num_segments = self.config.num_segments
            samples_per_segment = int(samples_per_track / num_segments)
            num_melspec_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
            # Divide the track into 10 segments to get more training data
            for seg_num in range(num_segments):
                start = samples_per_segment * seg_num
                finish = start + samples_per_segment

                mels = librosa.feature.melspectrogram(y=signal[start:finish], sr=sr, n_mels=num_mels, n_fft=n_fft,
                                                      hop_length=hop_length)
                mels = np.log(mels + 1e-9)

                # Normalise the feature matrix as an image
                img = scale_minmax(mels, 0, 255).astype(np.uint8)
                img = np.flip(img, axis=0)
                img = 255 - img
                img = img.T

                # Store the vectors if they match the expected length num_melspec_vectors_per_segment
                if len(img) == num_melspec_vectors_per_segment:
                    feat["train_feature"].append(img.tolist())
                    feat["train_labels"].append(np.where(np.array(labels) > 50, 1, 0).tolist())
                    feat["train_audio_files"].append(file_name)

        # Feature extraction for the test set
        for index, file_name in enumerate(data["test_audio_files"]):
            labels = data["test_labels"][index]
            signal, sr = librosa.load(file_name, sr=22050)

            # Randomly pick a window of 12 seconds for evaluation
            num_samples_test = int((self.config.duration/self.config.num_segments) * sr)
            start = random.randint(1, len(signal) - num_samples_test)
            finish = start + num_samples_test

            mels = librosa.feature.melspectrogram(y=signal[start:finish], sr=sr, n_mels=num_mels, n_fft=n_fft,
                                                  hop_length=hop_length)
            mels = np.log(mels + 1e-9)

            # Normalise the feature matrix as an image
            img = scale_minmax(mels, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            img = 255 - img
            img = img.T

            feat["test_feature"].append(img.tolist())
            feat["test_labels"].append(np.where(np.array(labels) > 50, 1, 0).tolist())
            feat["test_audio_files"].append(file_name)
        # Save the features as a json file - Takes a long time saving the features of the entire dataset
        if self.config.save:
            save_feat(feat)
        return feat


class MFCCExtractor(Extractor):
    """
    Class extending the Extractor and implementing the mel spectrogram feature extraction
    """

    def __init__(self, config: object):
        """
        Constructor for the MelSpectrogramExtractor
        :param config: Object containing the parameters related to feature extraction
        """
        super().__init__(config)
        self.config = config

    def extract(self, data: dict) -> dict:
        """
        Method to implement the Mel Frequency Cepstral Coeeficients feature extraction
        :param data: Dictionary containing the train/test audio files and labels
        :return: feat: Dictionary containing the mfcc features for the train/test set
        """
        feat = {
            "train_audio_files": [],
            "train_labels": [],
            "train_feature": [],
            "test_audio_files": [],
            "test_labels": [],
            "test_feature": []
        }
        hop_length = self.config.hop_length
        n_fft = self.config.n_fft
        num_mfcc = self.config.mfcc.num_mfcc
        # Feature extraction of training set
        for index, file_name in enumerate(data["train_audio_files"]):
            labels = data["train_labels"][index]
            # Take only the first 120 seconds of the audio file. This is done to tackle the different lengths of audio files as well as to reduce the computational load.
            signal, sr = librosa.load(file_name, sr=self.config.sampling_rate, duration=self.config.duration)
            # Discard the track that is less than 120 seconds
            if librosa.get_duration(y=signal, sr=sr) < self.config.duration:
                continue
            track_duration = self.config.duration  # measured in seconds
            samples_per_track = self.config.sampling_rate * track_duration
            num_segments = self.config.num_segments
            samples_per_segment = int(samples_per_track / num_segments)
            num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
            # Divide the track into 10 segments to get more training data
            for seg_num in range(num_segments):

                start = samples_per_segment * seg_num
                finish = start + samples_per_segment

                mfcc = librosa.feature.mfcc(signal[start:finish], sr, n_mfcc=num_mfcc, n_fft=n_fft,
                                            hop_length=hop_length)
                mfcc = mfcc.T
                # Store the vectors if they match the expected length num_mfcc_vectors_per_segment
                if len(mfcc) == num_mfcc_vectors_per_segment:
                    feat["train_audio_files"].append(file_name)
                    feat["train_feature"].append(mfcc.tolist())
                    feat["train_labels"].append(np.where(np.array(labels) > 50, 1, 0).tolist())
        # Feature extraction for the test set
        for index, file_name in enumerate(data["test_audio_files"]):
            labels = data["test_labels"][index]
            signal, sr = librosa.load(file_name, sr=22050)
            # Randomly pick a window of 12 seconds for evaluation
            start = random.randint(1, len(signal) - 12 * sr)
            finish = start + 12 * sr

            mfcc = librosa.feature.mfcc(signal[start:finish], sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            feat["test_audio_files"].append(file_name)
            feat["test_feature"].append(mfcc.tolist())
            feat["test_labels"].append(np.where(np.array(labels) > 50, 1, 0).tolist())
        # Save the features as a json file - Takes a long time saving the features of the entire dataset
        if self.config.save:
            save_feat(feat)
        return feat
