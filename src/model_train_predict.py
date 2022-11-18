# Copyrights (c) Preetham Ganesh.


import os

import tensorflow as tf

from src.model import DigitRecognizer


class LoadTrainValidateModel:
    """"""

    def __init__(self, model_version: str, model_configuration: dict) -> None:
        """"""
        self.model_version = model_version
        self.model_configuration = model_configuration
        self.home_directory_path = os.getcwd()

    def load_model(self) -> None:
        """Loads model & other utilies for training it."""
        # Loads model for current model configuration.
        self.model = DigitRecognizer(self.model_configuration)

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        checkpoint_directory_path = "{}/checkpoints/v{}".format(
            self.home_directory_path, self.model_version
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["learning_rate"]
        )
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory_path))
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_directory_path, max_to_keep=3
        )
