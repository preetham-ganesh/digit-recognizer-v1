# Copyrights (c) Preetham Ganesh.


import os
from typing import Tuple

import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np


class Dataset:
    """Loads the dataset for training & testing the model.

    Args:
        original_train_data: A pandas dataframe for original training data downloaded from Kaggle.
        original_test_data: A pandas dataframe for original testing data downloaded from Kaggle.
        new_train_data: A pandas dataframe for training data split from original train data.
        new_validation_data: A pandas dataframe for validation data split from original train data.
        new_test_data: A pandas dataframe for test data split from original train data.
        train_dataset: A tensorflow dataset which contains tensors for train input & target data.
        validation_dataset: A tensorflow dataset which contains tensors for validation input & target data.
        test_dataset: A tensorflow dataset which contains tensors for test input & target data.
    """

    def __init__(
        self, validation_data_percent: float, test_data_percent: float, batch_size: int
    ) -> None:
        """Creates object for the Dataset class."""
        # Asserts type & value of the arguments.
        assert isinstance(
            validation_data_percent, float
        ), "Variable validation_data_percent should be of type 'float'."
        assert isinstance(
            test_data_percent, float
        ), "Variable test_data_percent should be of type 'float'."
        assert isinstance(
            batch_size, float
        ), "Variable batch_size should be of type 'int'."
        assert (
            validation_data_percent > 0 and validation_data_percent < 1
        ), "Variable validation_data_percent should be between 0 & 1 (not included)."
        assert (
            test_data_percent > 0 and test_data_percent < 1
        ), "Variable test_data_percent should be between 0 & 1 (not included)."
        assert (
            batch_size > 0 and batch_size < 257
        ), "Variable batch_size should be between 0 & 257 (not included)."
        assert (
            validation_data_percent + test_data_percent > 0
            and validation_data_percent + test_data_percent < 1
        ), "Variables validation_data_percent + test_data_percent should be between 0 & 1 (not included)."

        # Initalizes data percent variables.
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

        # Deletes the original training data.
        del self.original_train_data

    def extract_input_target_data(self, new_data: pd.DataFrame) -> Tuple[tf.Tensor]:
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

    def combine_shuffle_slice_dataset(
        self, input_data: tf.Tensor, target_data: tf.Tensor
    ) -> tf.data.Dataset:
        """Converts the input data and target data into tensorflow dataset and slices them based on batch size."""
        # Asserts type of arguments.
        assert isinstance(
            input_data, pd.DataFrame
        ), "Variable input_data should be of type 'tf.Tensor'."
        assert isinstance(
            target_data, pd.DataFrame
        ), "Variable target_data should be of type 'tf.Tensor'."

        # Zip input and output tensors into a single dataset and shuffles it.
        dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data)).shuffle(
            len(input_data)
        )

        # Slices the combined dataset based on batch size, and drops remainder values.
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    def preprocess_dataset(self) -> None:
        """Preprocesses the dataset downloaded from Kaggle."""
        # Generates input & target data from split dataset & deletes it.
        (
            new_train_input_data,
            new_train_target_data,
        ) = self.extract_input_target_data(self.new_train_data)
        del self.new_train_data
        (
            new_validation_input_data,
            new_validation_target_data,
        ) = self.extract_input_target_data(self.new_validation_data)
        del self.new_validation_data
        (
            new_test_input_data,
            new_test_target_data,
        ) = self.extract_input_target_data(self.new_test_data)
        del self.new_test_data

        # Shuffles input and target data. Converts into tensorflow datasets.
        self.test_dataset = self.combine_shuffle_slice_dataset(
            new_test_input_data, new_test_target_data
        )
        del new_test_input_data, new_test_target_data
        self.validation_dataset = self.combine_shuffle_slice_dataset(
            new_validation_input_data, new_validation_target_data
        )
        del new_validation_input_data, new_validation_target_data
        self.train_dataset = self.combine_shuffle_slice_dataset(
            new_train_input_data, new_test_target_data
        )
        del new_train_input_data, new_train_target_data
