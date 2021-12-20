import json
from config import CFG

from dataloader import DataLoader
from extractor import MFCCExtractor, MelSpectrogramExtractor
from preprocessor import Preprocessor
from dlmodel import CNNModel, LSTMModel
from dlpipeline import DLPipeline


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, data: object, train: object, model: object, features: object):
        """
        Constructor for the configuration file containing all the parameter choices
        :param data: Parameters related to the dataset
        :param train: Parameters related to the training process
        :param model: Hyperparameters for the model
        :param features: Parameters related to features extraction - mfcc, mel spectrogram
        """
        self.data = data
        self.train = train
        self.model = model
        self.features = features

    @classmethod
    def from_json(cls, cfg: dict) -> object:
        """
        Class method to create the class structure of the different parameters
        :param cfg: Dictionary imported from the config file
        :return: Python object encapsulating all the four parameter objects
        """
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model, params.features)


class HelperObject(object):
    """
    Helper class to convert json into Python object
    """

    def __init__(self, dict_):
        self.__dict__.update(dict_)


if __name__ == "__main__":
    config = Config.from_json(CFG)

    # Create objects for DataLoader, Extractor, Preprocessor, and DLModel
    loader = DataLoader(config.data)
    extractor = MFCCExtractor(config.features)
    preprocessor = Preprocessor(config.data)
    model_type = config.model.model_type
    if model_type == "lstm":
        dlmodel = LSTMModel(config)
    elif model_type == "cnn":
        dlmodel = CNNModel(config)
    # Run the pipeline
    dlpipeline = DLPipeline(loader, extractor, preprocessor, dlmodel)
    dlpipeline.run_dl_steps(model_type)
