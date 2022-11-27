# Copyrights (c) Preetham Ganesh.


import os

import tensorflow as tf
import pandas as pd
import time

from src.model import DigitRecognizer
from src.utils import log_information


class TrainValidateModel:
    """"""

    def __init__(
        self,
        model_version: str,
        model_configuration: dict,
        batch_size: int,
        n_train_steps_per_epoch: int,
        n_validation_steps_per_epoch: int,
        n_test_steps_per_epoch: int,
    ) -> None:
        """Creates object for the LoadTrainValidateModel class."""
        # Asserts type & value of the arguments.
        assert isinstance(
            model_version, str
        ), "Variable model_version should be of type 'str'."
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."
        assert isinstance(
            batch_size, int
        ), "Variable batch_size should be of type 'int'."
        assert (
            batch_size > 0 and batch_size < 257
        ), "Variable batch_size should be between 0 & 257 (not included)."
        assert isinstance(
            n_train_steps_per_epoch, int
        ), "Variable n_train_steps_per_epoch should be of type 'int'."
        assert isinstance(
            n_validation_steps_per_epoch, int
        ), "Variable n_validation_steps_per_epoch should be of type 'int'."
        assert isinstance(
            n_test_steps_per_epoch, int
        ), "Variable n_test_steps_per_epoch should be of type 'int'."

        # Initalizes class variables.
        self.home_directory_path = os.getcwd()
        self.model_version = model_version
        self.model_configuration = model_configuration
        self.model_configuration["batch_size"] = batch_size
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.validation_accuracy = tf.keras.metrics.Mean(name="validation_accuracy")
        self.n_train_steps_per_epoch = n_train_steps_per_epoch
        self.n_validation_steps_per_epoch = n_validation_steps_per_epoch
        self.n_test_steps_per_epoch = n_test_steps_per_epoch
        self.patience_count = 0
        self.best_validation_loss = None
        self.model_history = pd.DataFrame(
            columns=[
                "epochs",
                "train_loss",
                "validation_loss",
                "train_accuracy",
                "validation_accuracy",
            ]
        )

    def load_model(self) -> None:
        """Loads model & other utilies for training it."""
        # Loads model for current model configuration.
        self.model = DigitRecognizer(self.model_configuration)

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        self.checkpoint_directory_path = "{}/checkpoints/v{}".format(
            self.home_directory_path, self.model_version
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["learning_rate"]
        )
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_directory_path))
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=self.checkpoint_directory_path, max_to_keep=3
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
        log_information(str(self.model.summary()))
        log_information("")

        # Plots the model and saves it as a PNG file.
        tf.keras.utils.plot_model(
            self.model.build_graph(),
            "{}/reports/v{}/model_plot.png".format(
                self.home_directory_path, self.model_configuration["model_version"]
            ),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=False,
        )
        log_information(
            "Model Plot saved at {}/reports/v{}/model_plot.png.".format(
                self.home_directory_path, self.model_configuration["model_version"]
            )
        )
        log_information("")

    def update_model_history(self, current_epoch: int) -> None:
        """Updates model history dataframe with latest metrics & saves it as CSV file."""
        # Asserts type & value of the arguments.
        assert isinstance(current_epoch, int), "Variable epoch should be of type 'int'."

        # Updates the metrics dataframe with the metrics for the current training & validation metrics.
        history_dictionary = {
            "epochs": int(current_epoch + 1),
            "train_loss": str(round(self.train_loss.result().numpy(), 3)),
            "validation_loss": str(round(self.validation_loss.result().numpy(), 3)),
            "train_accuracy": str(round(self.train_accuracy.result().numpy(), 3)),
            "validation_accuracy": str(
                round(self.validation_accuracy.result().numpy(), 3)
            ),
        }
        self.model_history = self.model_history.append(
            history_dictionary, ignore_index=True
        )

        # Saves history dataframe on as a CSV file.
        self.model_history.to_csv(
            "{}/reports/v{}/history.csv".format(
                self.home_directory_path, self.model_version
            ),
            index=False,
        )

    def preprocess_input_batch(self, input_batch: tf.Tensor) -> tf.Tensor:
        """Processes input batch normalizing the pixel value range, and type casting them to float32 type."""
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable epoch should be of type 'tf.Tensor'."

        # Casts input and target batches to float32 type.
        input_batch = tf.cast(input_batch, dtype=tf.float32)

        # Normalizes the input and target batches from [0, 255] range to [0, 1] range.
        input_batch = input_batch / 255.0
        return input_batch

    def loss_function(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual values and predicted values."""
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes loss for current target & predicted batches.
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        current_loss = loss_object(target_batch, predicted_batch)
        return current_loss

    def accuracy_function(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes accuracy for the current batch using actual values and predicted values."""
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        accuracy = tf.keras.metrics.categorical_accuracy(target_batch, predicted_batch)
        return tf.reduce_mean(accuracy)

    @tf.function
    def train_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Train the current model using current input & target batches."""
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Processes input batch for training the model.
        input_batch = self.preprocess_input_batch(input_batch)

        # Computes masked images for all input images in the batch, and computes batch loss.
        with tf.GradientTape() as tape:
            predicted_batch = self.model(input_batch, True)
            batch_loss = self.loss_function(target_batch, predicted_batch)

        # Computes gradients using loss & model variables. Apply the computed gradients on model variables using optimizer.
        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Computes accuracy score for the current batch.
        batch_accuracy = self.accuracy_function(target_batch, predicted_batch)

        # Computes mean for loss and accuracy.
        self.train_loss(batch_loss)
        self.train_accuracy(batch_accuracy)

    def validation_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Validates the model using the current input and target batches"""
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Processes input batch for validating the model.
        input_batch = self.process_input_batch(input_batch)

        # Computes masked images for all input images in the batch.
        predicted_batch = self.model(input_batch, False)

        # Computes loss & accuracy for the target batch and predicted batch.
        batch_loss = self.loss_function(target_batch, predicted_batch)
        batch_accuracy = self.accuracy_function(target_batch, predicted_batch)

        # Computes mean for loss & accuracy.
        self.validation_loss(batch_loss)
        self.validation_accuracy(batch_accuracy)

    def reset_metrics(self) -> None:
        """Resets states for training and validation metrics before the start of each epoch."""
        self.train_loss.reset_states()
        self.validation_loss.reset_states()
        self.train_accuracy.reset_states()
        self.validation_accuracy.reset_states()

    def train_model_per_epoch(
        self, train_dataset: tf.data.Dataset, current_epoch: int
    ) -> None:
        """Trains the model using the current train dataset."""
        # Asserts type & value of the arguments.
        assert isinstance(
            train_dataset, tf.data.Dataset
        ), "Variable train_dataset should be of type 'tf.data.Dataset'."
        assert isinstance(
            current_epoch, int
        ), "Variable current_epoch should be of type 'int'."

        # Iterates across batches in the train dataset.
        for (batch, (input_batch, target_batch)) in enumerate(
            train_dataset.take(self.n_train_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Trains the model using the current input and target batch.
            self.train_step(input_batch, target_batch)
            batch_end_time = time.time()
            if batch % 10 == 0:
                log_information(
                    "Epoch={}, Batch={}, Train loss={}, Train accuracy={}, Time taken={} sec.".format(
                        current_epoch + 1,
                        batch,
                        str(round(self.train_loss.result().numpy(), 3)),
                        str(round(self.train_accuracy.result().numpy(), 3)),
                        round(batch_end_time - batch_start_time, 3),
                    )
                )
        log_information("")

    def validate_model_per_epoch(
        self, validation_dataset: tf.data.Dataset, current_epoch: int
    ) -> None:
        """Validates the model using the current validation dataset."""
        # Asserts type & value of the arguments.
        assert isinstance(
            validation_dataset, tf.data.Dataset
        ), "Variable validation_dataset should be of type 'tf.data.Dataset'."
        assert isinstance(
            current_epoch, int
        ), "Variable current_epoch should be of type 'int'."

        # Iterates across batches in the validation dataset.
        for (batch, (input_batch, target_batch)) in enumerate(
            validation_dataset.take(self.n_validation_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)
            batch_end_time = time.time()
            if batch % 10 == 0:
                log_information(
                    "Epoch={}, Batch={}, Validation loss={}, Validation accuracy={}, Time taken={} sec.".format(
                        current_epoch + 1,
                        batch,
                        str(round(self.validation_loss.result().numpy(), 3)),
                        str(round(self.validation_accuracy.result().numpy(), 3)),
                        round(batch_end_time - batch_start_time, 3),
                    )
                )
        log_information("")

    def early_stopping(self) -> bool:
        """Stops the model from learning further if the performance has not improved from previous epoch."""
        # If epoch = 1, then best validation loss is replaced with current validation loss, & the checkpoint is saved.
        if self.best_validation_loss is None:
            self.patience_count = 0
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.manager.save()
            log_information(
                "Checkpoint saved at {}.".format(self.checkpoint_directory_path)
            )

        # If best validation loss is higher than current validation loss, the best validation loss is replaced with
        # current validation loss, & the checkpoint is saved.
        elif self.best_validation_loss > str(
            round(self.validation_loss.result().numpy(), 3)
        ):
            self.patience_count = 0
            log_information(
                "Best validation loss changed from {} to {}".format(
                    str(self.best_validation_loss),
                    str(round(self.validation_loss.result().numpy(), 3)),
                )
            )
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.manager.save()
            log_information(
                "Checkpoint saved at {}".format(self.checkpoint_directory_path)
            )

        # If best validation loss is not higher than the current validation loss, then the number of times the model
        # has not improved is incremented by 1.
        elif self.patience_count <= 4:
            self.patience_count += 1
            log_information("Best validation loss did not improve.")
            log_information("Checkpoint not saved.")

        # If the number of times the model did not improve is greater than 4, then model is stopped from training
        # further.
        else:
            return False
        return True
