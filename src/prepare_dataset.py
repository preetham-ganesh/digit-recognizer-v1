# Copyrights (c) Preetham Ganesh.


import os
from typing import Tuple

import pandas as pd


class MakeDataset(object):
    """"""

    def __init__(self) -> None:
        """"""
        print()

    def load_dataset(self) -> None:
        """Loads original downloaded dataset."""
        home_directory_path = os.getcwd()
        self.original_train_data = pd.read_csv(
            "{}/data/raw_data/train.csv".format(home_directory_path)
        )
        self.original_test_data = pd.read_csv(
            "{}/data/raw_data/test.csv".format(home_directory_path)
        )
