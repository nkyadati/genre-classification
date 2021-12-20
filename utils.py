import tensorflow as tf
import json
import matplotlib.pyplot as plt
from typing import Tuple, List, Type
import numpy as np

K = tf.keras.backend


def scale_minmax(mat: List[list], min_val: float, max_val: float) -> List[list]:
    """
    Function to normalise the image matrix to contain values between 0 and 255
    :param mat: Mel spectrogram image that needs to be normalised
    :param min_val: Minimul value in the matrx - 0
    :param max_val: Maximum value in the matrix - 255
    :return: mat_scaled: Normalised matrix
    """
    mat_std = (mat - mat.min()) / (mat.max() - mat.min())
    mat_scaled = mat_std * (max_val - min_val) + min_val
    return mat_scaled


def save_feat(features: dict) -> None:
    """
    Function to save the extracted audio features and ground-truth labels into a json file
    :param features: Dictionary containing the features and the ground-tructh labels
    :return:
    """
    with open('features.json', "w") as fp:
        json.dump(features, fp, indent=4)


def load_feat() -> dict:
    """
    Function to load the features and ground-truth labels from a file on disk.
    :return: features: dictionary containing the following keys - train_feature, train_lab, test_feature, and test_lab
    """
    with open('features.json', 'r') as f:
        features = json.load(f)
    return features


def hamming_loss(y_true: list, y_pred: list) -> float:
    """
    Custom metric for measuring the accuracy of the model on validation and test set
    :param y_true: Ground-truth labels
    :param y_pred: Predicted labels
    :return: Mean hamming loss
    """
    tmp = K.abs(y_true - y_pred)
    return K.mean(K.cast(K.greater(tmp, 0.5), dtype=float))


def plot_history(history: object) -> None:
    """Plots accuracy/loss for training/validation set as a function of the epochs during
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sub-plot
    axs[0].plot(history.history["hamming_loss"], label="Train accuracy")
    axs[0].plot(history.history["val_hamming_loss"], label="Test accuracy")
    axs[0].set_ylabel("Hamming Loss")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy")

    # create loss sub-plot
    axs[1].plot(history.history["loss"], label="Train loss")
    axs[1].plot(history.history["val_loss"], label="Test loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss")

    # Save the plot
    plt.savefig('training.png')
    plt.show()


def label_based_macro_precision(y_true, y_pred):
    # axis = 0 computes true positive along columns i.e labels
    l_prec_num = np.sum(np.logical_and(y_true, y_pred), axis=0)

    # axis = computes true_positive + false positive along columns i.e labels
    l_prec_den = np.sum(y_pred, axis=0)

    # compute precision per class/label
    l_prec_per_class = l_prec_num / l_prec_den

    # macro precision = average of precsion across labels.
    l_prec = np.mean(l_prec_per_class)
    return l_prec
