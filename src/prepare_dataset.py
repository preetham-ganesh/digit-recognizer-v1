# Copyrights (c) Preetham Ganesh.


import os
from typing import Tuple

import pandas as pd
from sklearn.utils import shuffle


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
