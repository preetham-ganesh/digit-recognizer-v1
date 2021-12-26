# authors_name = 'Preetham Ganesh'
# project_title = 'Kaggle - Digit Recognizer'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from models import DigitRecognizer1, DigitRecognizer2
import sys


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


def digit_recognizer_1():
    """Sequential model for the 1st configuration of hyperparameters used for developing the Digit Recognizer.

        Args:
            None

        Returns:
            Tensorflow compiled model for the 1st configuration of hyperparameters.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=4, padding='valid', strides=1, activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=4, padding='valid', strides=1, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def digit_recognizer_2():
    """Sequential model for the 1st configuration of hyperparameters used for developing the Digit Recognizer.

        Args:
            None

        Returns:
            Tensorflow compiled model for the 1st configuration of hyperparameters.
    """


def choose_model(model_number: int):
    """Initializes an object for the Digit Recognizer model classes based on the model number.

        Args:
            model_number: Integer which decides the model to be trained.

        Returns:
            Object for the Digit Recognizer model class.
    """
    if model_number == 1:
        model = DigitRecognizer1()
    elif model_number == 2:
        model = DigitRecognizer2()
    else:
        print('The model number entered is incorrect. Kindly enter the right model number')
        sys.exit()
    return model


