import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


class Preprocessor:
    """
    Class for preparing the data for training
    """

    def __init__(self, config: object) -> object:
        """
        Constructor for the Preprocessor class
        :param config: Object containing the parameters for data
        """
        self.config = config

    def preprocess(self, features: dict, model_type: str) -> Tuple[dict, dict, tuple, int]:
        """
        Method for splitting the training data into two sets - train and validation; Change the shape of the feature matrices depending on the model type
        :param features: Dictionary containing the features and ground-truth labels
        :param model_type: Type of model for training - lstm/cnn
        :return: Dictionaries for training and testing, shape of the input and output for the model
        """
        X = np.array(features["train_feature"])
        y = np.array(features["train_labels"])
        train_data, validation_data, train_lab, validation_lab = train_test_split(X, y, test_size=0.2)
        test_data = np.array(features["test_feature"])
        test_lab = np.array(features["test_labels"])
        # Change the input shape based on the model type
        if model_type == 'cnn':
            train_data = train_data[..., np.newaxis]
            validation_data = validation_data[..., np.newaxis]
            test_data = test_data[..., np.newaxis]
            input_shape = (train_data.shape[1], train_data.shape[2], 1)
        elif model_type == 'lstm':
            input_shape = (train_data.shape[1], train_data.shape[2])
        n_outputs = train_lab.shape[1]

        train_dataset = {"train_data": train_data, "train_lab": train_lab, "validation_data": validation_data,
                         "validation_lab": validation_lab}
        test_dataset = {"test_data": test_data, "test_lab": test_lab, "test_audio_files": features["test_audio_files"]}

        return train_dataset, test_dataset, input_shape, n_outputs
