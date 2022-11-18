# Copyrights (c) Preetham Ganesh.


import os

import tensorflow as tf

from src.model import DigitRecognizer
from src.utils import log_information


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
        log_information("Finished loading model for current configuration.")
        log_information("")

    def generate_model_summary_and_plot(self) -> None:
        """Generates summary and plot for loaded model."""
        # Compiles the model to print model summary.
        input_dim = (
            self.model_configuration["final_image_size"],
            self.model_configuration["final_image_size"],
            self.model_configuration["n_channels"],
        )
        _ = self.model.build((self.model_configuration["batch_size"], *input_dim))
        log_information(self.model.summary())
        log_information("")

        # Plots the model and saves it as a PNG file.
        tf.keras.utils.plot_model(
            self.model.build_graph(),
            "{}/results/v{}/utils/model_plot.png".format(
                self.home_directory_path, self.model_configuration["model_version"]
            ),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=False,
        )
