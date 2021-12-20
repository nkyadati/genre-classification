import tensorflow as tf
import pandas as pd
import os
import random


class DataLoader:
    """
    Dataloader class with methods to load the csv file
    """

    def __init__(self, config: object):
        """
        Constructor for the data loader class
        :param config: Object with parameters related to the dataset
        """
        self.config = config

    def load_data(self) -> dict:
        """
        Method to load the csv file and split the audio files into train and test set
        :return: data dictionary with the following keys - train_labels, train_audio_files, test_labels, test_audio_files
        """
        df = pd.read_csv(self.config.path, sep='\t')
        df_list = df.values.tolist()
        # Dictionary to store the audio files and labels
        data = {
            "train_labels": [],
            "train_audio_files": [],
            "test_labels": [],
            "test_audio_files": []
        }
        # Randomise the dataset and split it into two lists - train and test
        random.shuffle(df_list)
        test_list = df_list[:int(self.config.test_split * len(df_list))]
        train_list = df_list[int(self.config.test_split * len(df_list)):]
        # Store the labels in the dictionary
        for row in train_list:
            file_name = row[0]
            labels = row[1:]
            data["train_labels"].append(labels)
            data["train_audio_files"].append(os.path.join(self.config.dataset_path, file_name))

        for row in test_list:
            file_name = row[0]
            labels = row[1:]
            data["test_labels"].append(labels)
            data["test_audio_files"].append(os.path.join(self.config.dataset_path, file_name))

        return data
