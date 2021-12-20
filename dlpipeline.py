import tensorflow as tf
import os
from typing import Tuple
from utils import load_feat, plot_history
from dataloader import DataLoader
from extractor import Extractor
from preprocessor import Preprocessor
from dlmodel import DLModel

class DLPipeline:
    """
    Class running the different steps of the deep learning pipeline
    """

    def __init__(self, loader: DataLoader, extractor: Extractor, preprocessor: Preprocessor, dlmodel: DLModel):
        """
        Constructor for the DLPipeline class
        :param loader: DataLoader object
        :param extractor: Extractor object
        :param preprocessor: Preprocessor object
        :param dlmodel: DLModel object
        """
        self.extractor = extractor
        self.loader = loader
        self.preprocessor = preprocessor
        self.dlmodel = dlmodel

    def run_dl_steps(self, model_type: str) -> None:
        """
        Method to run the steps of the deep learning pipeline - load data, extract features, train and evaluate the model
        :param model_type: String indicating the type of the model - lstm/cnn
        :return:
        """
        print("Running DL pipeline")
        # Load the features if the file exists, else call the feature extraction function
        if os.path.isfile('features.json'):
            features = load_feat()
        else:
            data = self._load_data()
            features = self._extract(data)
        train_dataset, test_dataset, input_shape, n_outputs = self._preprocess(features, model_type)
        history, loss, accuracy = self._build_train_and_evaluate(train_dataset, test_dataset, input_shape, n_outputs)
        with open('metrics.txt', 'w') as fd:
            fd.write("Hamming loss: {}; ".format(accuracy))
        plot_history(history)

    def _load_data(self) -> dict:
        """
        Private method to call the load_data method of the DataLoader class
        :return:
        """
        return self.loader.load_data()

    def _extract(self, data: dict) -> dict:
        """
        Private method to call the extract function of the Extractor class
        :param data: Dictionary containing the parameters for feature extraction
        :return:
        """
        return self.extractor.extract(data)

    def _preprocess(self, features: dict, model_type: str) -> Tuple[dict, dict, tuple, int]:
        """
        Provate method to call the preprocess method of the Preprocessor class
        :param features: Dictionary containing the features and ground-truth labels
        :param model_type: String indicating the type of the model
        :return:
        """
        return self.preprocessor.preprocess(features, model_type)

    def _build_train_and_evaluate(self, train_dataset: dict, test_dataset: dict, input_shape: tuple, n_outputs: int) -> Tuple[object, float, float]:
        """
        Private method calling the build, train, and evaluate methods of the DLModel class
        :param train_dataset: Dictionary containing the features and ground-truth labels of the training set
        :param test_dataset: Dictionary containing the features and ground-truth labels of the test set
        :param input_shape: Shape of the input to the model
        :param n_outputs: Number of outputs of the model
        :return: Training history and scores of evaluation
        """
        self.dlmodel.build(input_shape, n_outputs)
        history = self.dlmodel.train(train_dataset)
        scores, loss, macro_prec = self.dlmodel.evaluate(test_dataset)
        return history, scores, loss
