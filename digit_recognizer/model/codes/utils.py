# authors_name = 'Preetham Ganesh'
# project_title = 'Kaggle - Digit Recognizer'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import numpy as np


def data_splitting(original_data: pd.DataFrame):
    """Splits the data into training, validation, and testing dataframes.

        Args:
            original_data: The dataframe which contains pixel details for all the images in the original training data.

        Returns:
            A tuple which dataframes for new splits of training, validation and testing data.
    """
    # Shuffles the dataframe.
    original_data = shuffle(original_data)

    # Splits the original dataframe into training, validation and testing data.
    new_test_data = original_data.iloc[:1000]
    new_validation_data = original_data.iloc[1000:2000]
    new_train_data = original_data.iloc[2000:]
    return new_train_data, new_validation_data, new_test_data


def data_preprocessing(new_data: pd.DataFrame):
    """Converts new split dataframe into input and target data. Performs data transformation on the new input data.

        Args:
            new_data: New split dataframe which contains the label and pixel values for the image.

        Returns:
            A tuple which contains 2 Tensors for the new_input_data and new_target_data.
    """
    # Splits the data into the input and target data
    new_input_data = new_data.drop(columns=['label'])
    new_target_data = list(new_data['label'])

    # Reshapes the input data, converts into float32 format and normalizes the pixel values.
    new_input_data = np.array(new_input_data).reshape((len(new_input_data), 28, 28, 1)).astype('float32') / 255

    # Converts the input and target data into Tensorflow tensors.
    new_input_data = tf.convert_to_tensor(new_input_data)
    new_target_data = tf.convert_to_tensor(new_target_data)
    return new_input_data, new_target_data


def loss_function(actual_values: tf.Tensor,
                  predicted_values: tf.Tensor):
    """Calculates loss for the current batch of actual values and the predicted values.

        Args:
            actual_values: Actual classes the images belong to.
            predicted_values: Classes predicted by the neural network model for each image in batch.

        Return:
            Loss value computed for the current batch.
    """
    # Initializes the object for calculating the loss.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Mask off the losses on padding and computes the loss.
    mask = tf.math.logical_not(tf.math.equal(actual_values, 0))
    loss_ = loss_object(actual_values, predicted_values)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


