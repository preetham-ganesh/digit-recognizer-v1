# Copyrights (c) Preetham Ganesh.


import os
from typing import Tuple

import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np


class Dataset:
    """"""

    def __init__(
        self, validation_data_percent: float, test_data_percent: float
    ) -> None:
        """"""
        self.validation_data_percent = validation_data_percent
        self.test_data_percent = test_data_percent

    def load_dataset(self) -> None:
        """Loads original downloaded dataset."""
        home_directory_path = os.getcwd()
        self.original_train_data = pd.read_csv(
            "{}/data/raw_data/train.csv".format(home_directory_path)
        )
        self.original_test_data = pd.read_csv(
            "{}/data/raw_data/test.csv".format(home_directory_path)
        )

    def split_dataset(self) -> None:
        """Splits the data into training, validation, and testing dataframes."""
        # Shuffles the original training dataframe.
        self.original_train_data = shuffle(self.original_train_data)

        # Computes end index for validation & test data.
        test_data_end_index = int(
            len(self.original_train_data) * self.test_data_percent
        )
        validation_data_end_index = (
            int(len(self.original_train_data) * self.validation_data_percent)
            + test_data_end_index
        )

        # Splits the original dataframe into training, validation and testing data.
        self.new_test_data = self.original_train_data[:test_data_end_index]
        self.new_validation_data = self.original_train_data[
            test_data_end_index:validation_data_end_index
        ]
        self.new_train_data = self.original_train_data[validation_data_end_index:]

    def preprocess_current_dataset(self, new_data: pd.DataFrame) -> Tuple[tf.Tensor]:
        """Preprocesses current dataset, to produce input (& target) data."""
        # Asserts type of arguments.
        assert isinstance(
            new_data, pd.DataFrame
        ), "Variable new_data should be of type 'pd.DataFrame'."

        # If label column exists in the data, then both input & target data is extracted.
        if "label" in list(new_data.columns):

            # Splits the data into the input and target data
            new_input_data = new_data.drop(columns=["label"])
            new_target_data = list(new_data["label"])

        # Else input data is extracted.
        else:
            new_input_data = new_data

        # Reshapes the input data, converts into float32 format, normalizes the pixel values and converts to tensor.
        new_input_data = np.array(new_input_data).reshape(
            (len(new_input_data), 28, 28, 1)
        )
        new_input_data = tf.convert_to_tensor(new_input_data, dtype="uint8")

        # If label column exists in the data, then converts target data into categorical format.
        if "label" in list(new_data.columns):
            new_target_data = tf.keras.utils.to_categorical(new_target_data)
            return new_input_data, new_target_data
        return new_input_data
