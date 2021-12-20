from abc import abstractmethod, ABC
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from typing import Tuple, List, Type

from utils import hamming_loss, label_based_macro_precision


class DLModel(ABC):
    """
    Abstract class for the Deep learning model used for training
    """

    def __init__(self, config):
        self.config = config

    def build(self, input_shape, n_outputs):
        pass

    def train(self, dataset):
        pass

    def evaluate(self, dataset):
        pass


class CNNModel(DLModel):
    """
    Class implementing CNN and extending the DLModel abstract class
    """

    def __init__(self, config: object):
        """
        Constructor for the CNNModel class
        :param config: Object containing the configuration parameters
        """
        super().__init__(config)
        self.model = None

    def build(self, input_shape: tuple, n_outputs: int) -> None:
        """
        Method to construct the CNN model
        :param input_shape: Tuple representing the shape of the input
        :param n_outputs: Number of outputs of the model
        :return:
        """
        self.model = keras.Sequential()

        for index in range(self.config.model.conv_layer.num_conv_layers):
            if index == 0:
                self.model.add(keras.layers.Conv2D(self.config.model.conv_layer.num_filters[index], (
                    self.config.model.conv_layer.filter_size, self.config.model.conv_layer.filter_size),
                                                   activation=self.config.model.conv_layer.activation,
                                                   input_shape=input_shape))
                self.model.add(keras.layers.MaxPooling2D(
                    (self.config.model.conv_layer.filter_size, self.config.model.conv_layer.filter_size),
                    strides=(self.config.model.conv_layer.stride, self.config.model.conv_layer.stride), padding='same'))
                self.model.add(keras.layers.BatchNormalization())
            else:
                self.model.add(keras.layers.Conv2D(self.config.model.conv_layer.num_filters[index], (
                    self.config.model.conv_layer.filter_size, self.config.model.conv_layer.filter_size),
                                                   activation=self.config.model.conv_layer.activation))
                self.model.add(keras.layers.MaxPooling2D(
                    (self.config.model.conv_layer.filter_size, self.config.model.conv_layer.filter_size),
                    strides=(self.config.model.conv_layer.stride, self.config.model.conv_layer.stride), padding='same'))
                self.model.add(keras.layers.BatchNormalization())
                self.model.add(keras.layers.Dropout(0.5))

        for index in range(self.config.model.dense_layer.num_dense_layers):
            self.model.add(keras.layers.Flatten())
            self.model.add(keras.layers.Dense(self.config.model.dense_layer.num_units[index],
                                              activation=self.config.model.dense_layer.activation))
            self.model.add(keras.layers.Dropout(self.config.model.dense_layer.dropout))

        self.model.add(keras.layers.Dense(n_outputs, activation=self.config.model.output_layer.activation))

    def train(self, dataset: dict) -> object:
        """
        Method to compile and train the CNN model
        :param dataset: Dictionary containing the train/validation features and ground-truth labels
        :return: history: Object containing the training history of the model
        """

        # Callback function for early stopping - to avoid overfitting
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        if self.config.train.optimizer.type == "adam":
            optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        # Compile the model with a custom metric of hamming loss - measuring the proportion of wrongly classified labels
        self.model.compile(optimizer=optimiser,
                           loss=self.config.train.loss,
                           metrics=[hamming_loss])
        history = self.model.fit(dataset["train_data"], dataset["train_lab"],
                                 validation_data=(dataset["validation_data"], dataset["validation_lab"]),
                                 epochs=self.config.train.epochs, batch_size=self.config.train.batch_size,
                                 callbacks=[callback])
        return history

    def evaluate(self, dataset: dict) -> Tuple[float, float]:
        """
        Method to make predictions and evaluate the model on a test set
        :param dataset: Dictionary containing the test features and ground-truth labels
        :return: Loss and Accuracy on the test set
        """
        if self.model is None:
            self.model = keras.models.load_model("./models")

        predictions = self.model.predict(dataset["test_data"])
        test_files = pd.DataFrame()
        pred_labels = pd.DataFrame()
        for index, prediction in enumerate(predictions):
            pred = np.where(np.array(prediction) > 0.5, 1, 0).tolist()
            test_files = test_files.append(pd.Series(dataset["test_audio_files"][index]), ignore_index=True)
            pred_labels = pred_labels.append(pd.Series(pred), ignore_index=True)
        results = pd.concat([test_files, pred_labels], axis=1, join='inner', ignore_index=True)
        results.to_csv("predictions.csv", index=False, header=False)

        test_loss, test_acc = self.model.evaluate(dataset["test_data"], dataset["test_lab"], verbose=2)
        return test_loss, test_acc


class LSTMModel(DLModel):
    """
    Class implementing LSTM and extending the DLModel abstract class
    """

    def __init__(self, config: object):
        """
        Constructor for the LSTMModel class
        :param config: Object containing the configuration parameters
        """
        super().__init__(config)
        self.model = None

    def build(self, input_shape: tuple, n_outputs: int) -> None:
        """
        Method to construct the LSTM model
        :param input_shape: Tuple representing the shape of the input
        :param n_outputs: Number of outputs of the model
        :return:
        """
        self.model = keras.Sequential()

        for index in range(self.config.model.lstm_layer.num_lstm_layers):
            if index == 0:
                self.model.add(keras.layers.LSTM(self.config.model.lstm_layer.num_units[index], input_shape=input_shape,
                                                 return_sequences=True))
            else:
                self.model.add(keras.layers.LSTM(self.config.model.lstm_layer.num_units[index]))
                self.model.add(keras.layers.Dropout(0.2))

        for index in range(self.config.model.dense_layer.num_dense_layers):
            self.model.add(keras.layers.Flatten())
            self.model.add(keras.layers.Dense(self.config.model.dense_layer.num_units[index],
                                              activation=self.config.model.dense_layer.activation))
            self.model.add(keras.layers.Dropout(self.config.model.dense_layer.dropout))

        self.model.add(keras.layers.Dense(n_outputs, activation=self.config.model.output_layer.activation))

    def train(self, dataset: dict) -> object:
        """
        Method to compile and train the LSTM model
        :param dataset: Dictionary containing the train/validation features and ground-truth labels
        :return: history: Object containing the training history of the model
        """
        # Callback function for early stopping - to avoid overfitting
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        if self.config.train.optimizer.type == "adam":
            optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        # Compile the model with a custom metric of hamming loss - measuring the proportion of wrongly classified labels
        self.model.compile(optimizer=optimiser,
                           loss=self.config.train.loss,
                           metrics=[hamming_loss])
        history = self.model.fit(dataset["train_data"], dataset["train_lab"],
                                 validation_data=(dataset["validation_data"], dataset["validation_lab"]),
                                 epochs=self.config.train.epochs, batch_size=self.config.train.batch_size,
                                 callbacks=[callback])

        self.model.save("./models")
        return history

    def evaluate(self, dataset: dict) -> Tuple[float, float]:
        """
        Method to make predictions and evaluate the model on a test set
        :param dataset: Dictionary containing the test features and ground-truth labels
        :return: Loss and Accuracy on the test set
        """
        if self.model is None:
            self.model = keras.models.load_model("./models")

        predictions = self.model.predict(dataset["test_data"])
        test_files = pd.DataFrame()
        pred_labels = pd.DataFrame()
        pred_list = []
        for index, prediction in enumerate(predictions):
            pred = np.where(np.array(prediction) > 0.5, 1, 0).tolist()
            pred_list.append(pred)
            test_files = test_files.append(pd.Series(dataset["test_audio_files"][index]), ignore_index=True)
            pred_labels = pred_labels.append(pd.Series(pred), ignore_index=True)
        results = pd.concat([test_files, pred_labels], axis=1, join='inner', ignore_index=True)
        results.to_csv("predictions.csv", index=False, header=False)

        test_loss, test_acc = self.model.evaluate(dataset["test_data"], dataset["test_lab"], verbose=2)

        macro_prec = label_based_macro_precision(dataset["test_lab"], pred_list)
        return test_loss, test_acc
