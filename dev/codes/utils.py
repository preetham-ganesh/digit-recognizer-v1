# authors_name = 'Preetham Ganesh'
# project_title = 'Digit Recognizer'
# email = 'preetham.ganesh2021@gmail.com'


import os
import logging
import warnings


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
warnings.filterwarnings("ignore")


import json
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

from model import DigitRecognition


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def check_directory_path_existence(directory_path: str) -> str:
    """Creates the absolute path for the directory path given in argument if it does not already exist.

    Args:
        directory_path: A string which contains the directory path that needs to be created if it does not already
            exist.

    Returns:
        A string which contains the absolute directory path.
    """
    # Creates the following directory path if it does not exist.
    home_directory_path = os.path.dirname(os.getcwd())
    absolute_directory_path = "{}/{}".format(home_directory_path, directory_path)
    if not os.path.isdir(absolute_directory_path):
        os.makedirs(absolute_directory_path)
    return absolute_directory_path


def create_log(logger_directory_path: str, log_file_name: str) -> None:
    """Creates an object for logging terminal output.

    Args:
        logger_directory_path: A string which contains the location where the log file should be stored.
        log_file_name: A string which contains the name for the log file.

    Returns:
        None.
    """
    # Checks if the following path exists.
    logger_directory_path = check_directory_path_existence(logger_directory_path)

    # Create and configure logger
    logging.basicConfig(
        filename="{}/{}".format(logger_directory_path, log_file_name),
        format="%(asctime)s %(message)s",
        filemode="w",
    )
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def log_information(log: str) -> None:
    """Saves current log information, and prints it in terminal.

    Args:
        log: A string which contains the information that needs to be printed in terminal and saved in log.

    Returns:
        None.

    Exception:
        NameError: When the logger is not defined, this exception is thrown.
    """
    try:
        logger.info(log)
    except NameError:
        _ = ""
    print(log)


def save_json_file(dictionary: dict, file_name: str, directory_path: str) -> None:
    """Converts a dictionary into a JSON file and saves it for future use.

    Args:
        dictionary: A dictionary which needs to be saved.
        file_name: A string which contains the name with which the file has to be saved.
        directory_path: A string which contains the path where the file needs to be saved.

    Returns:
        None.
    """
    # Creates the following directory path if it does not exist.
    directory_path = check_directory_path_existence(directory_path)

    # Saves the dictionary or list as a JSON file at the file path location.
    file_path = '{}/{}.json'.format(directory_path, file_name)
    with open(file_path, 'w') as out_file:
        json.dump(dictionary, out_file, indent=4)
    out_file.close()
    log_information('{} file saved successfully at {}.'.format(file_name, file_path))


def load_json_file(file_name: str, directory_path: str) -> dict:
    """Loads a JSON file into memory based on the file_name.

    Args:
        file_name: A string which contains the name of the of the file to be loaded.
        directory_path: A string which contains the location where the directory path exists.

    Returns:
        A dictionary which contains the JSON file.
    """
    file_path = '{}/{}.json'.format(directory_path, file_name)
    with open(file_path, 'r') as out_file:
        dictionary = json.load(out_file)
    out_file.close()
    return dictionary


def load_dataset() -> tuple:
    """Loads original Kaggle Digit Recognizer dataset.

    Args:
        None.
    
    Returns:
        A tuple which contains original train and test data downloaded from Kaggle website.
    """
    home_directory_path = os.path.dirname(os.getcwd())

    # Loads the original Kaggle data.
    original_train_data = pd.read_csv('{}/data/original/train.csv'.format(home_directory_path))
    original_test_data = pd.read_csv('{}/data/original/test.csv'.format(home_directory_path))
    return original_train_data, original_test_data


def data_splitting(original_data: pd.DataFrame, n_validation_examples: int, n_test_examples: int) -> tuple:
    """Splits the data into training, validation, and testing dataframes.
    
    Args:
        original_data: The dataframe which contains pixel details for all the images in the original training data.
        n_validation_examples: An integer which contains number of examples in validation dataset.
        n_test_examples: An integer which contains number of examples in test dataset.
    
    Returns:
        A tuple which dataframes for new splits of training, validation and testing data.
    """
    # Shuffles the dataframe.
    original_data = shuffle(original_data)

    # Splits the original dataframe into training, validation and testing data.
    new_test_data = original_data.iloc[:n_test_examples]
    new_validation_data = original_data.iloc[n_test_examples:n_validation_examples + n_test_examples]
    new_train_data = original_data.iloc[n_validation_examples + n_test_examples:]
    return new_train_data, new_validation_data, new_test_data


def data_preprocessing(new_data: pd.DataFrame):
    """Converts new split dataframe into input and target data. Performs data transformation on the new input data.

    Args:
        new_data: New split dataframe which contains the label and pixel values for the image.
    
    Returns:
        A tuple which contains 2 Tensors for the new_input_data and new_target_data.
    """
    if 'label' in list(new_data.columns):

        # Splits the data into the input and target data
        new_input_data = new_data.drop(columns=['label'])
        new_target_data = list(new_data['label'])
    else:
        new_input_data = new_data

    # Reshapes the input data, converts into float32 format, normalizes the pixel values and converts to tensor.
    new_input_data = np.array(new_input_data).reshape((len(new_input_data), 28, 28, 1))
    new_input_data = tf.convert_to_tensor(new_input_data, dtype='uint8')

    if 'label' in list(new_data.columns):
        new_target_data = tf.keras.utils.to_categorical(new_target_data)
        return new_input_data, new_target_data

    return new_input_data


def shuffle_slice_dataset(input_data: list, target_data: list, batch_size: int) -> tf.data.Dataset:
    """Converts the input data and target data into tensorflow dataset and slices them based on batch size.

    Args:
        input_data: A list which contains the current data split's resized input images.
        target_data: A list which contains the current data split's resized target images.
        batch_size: An integer which contains batch size for slicing the dataset into small chunks.
    
    Returns:
        A TensorFlow dataset which contains sliced input and target tensors for the current data split.
    """
    # Zip input and output tensors into a single dataset and shuffles it.
    dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data)).shuffle(len(input_data))

    # Slices the combined dataset based on batch size, and drops remainder values.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def process_input_batch(input_batch: tf.Tensor) -> tf.Tensor:
    """Processes input batch normalizing the pixel value range, and type casting them to float32 type.

    Args:
        input_batch: A tensor which contains the input images from the current batch for training the model.
    
    Returns:
        A tensor which contains the processed input batch.
    """
    # Casts input and target batches to float32 type.
    input_batch = tf.cast(input_batch, dtype=tf.float32)

    # Normalizes the input and target batches from [0, 255] range to [0, 1] range.
    input_batch = input_batch / 255.0
    return input_batch


def loss_function(target_batch: tf.Tensor, predicted_batch: tf.Tensor) -> tf.Tensor:
    """Computes the loss value for the current batch of the predicted values based on comparison with actual values.

    Args:
        target_batch: A tensor which contains the actual values for the current batch.
        predicted_batch: A tensor which contains the predicted values for the current batch.
    
    Returns:
        A tensor which contains loss for the current batch.
    """
    # Creates the loss object for the Categorical Crossentropy & computes loss using the target and predicted batches.
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    current_loss = loss_object(target_batch, predicted_batch)
    return current_loss


def accuracy_function(target_batch: tf.Tensor, predicted_batch: tf.Tensor):
    """Computes the accuracy value for the current batch of the predicted values based on comparison with actual values.

    Args:
        target_batch: A tensor which contains the actual values for the current batch.
        predicted_batch: A tensor which contains the predicted values for the current batch.
    
    Returns:
        A tensor which contains accuracy for the current batch.
    """
    # Computes loss for the current batch using actual values and predicted values.
    accuracy = tf.keras.metrics.categorical_accuracy(target_batch, predicted_batch)
    return tf.reduce_mean(accuracy)


@tf.function
def train_step(input_batch: tf.Tensor, target_batch: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer) -> None:
    """Trains the convolutional neural network model using the current input and target training batches. Predicts the 
        output for the current input batch, computes loss on comparison with the target batch, and optimizes the model 
        based on the computed loss.
    
    Args:
        input_batch: A tensor which contains the input images from the current batch for training the model.
        target_batch: A tensor which contains the target images from the current batch for training and validating the 
            model.
        optimizer: An tensorflow optimizer object which contains optimizing algorithm used improve the performance of 
            the model.
    
    Returns:
        None.
    """
    # Processes input batch for training the model.
    input_batch = process_input_batch(input_batch)

    # Computes masked images for all input images in the batch, and computes batch loss.
    with tf.GradientTape() as tape:
        predicted_batch = model(input_batch, True)
        batch_loss = loss_function(target_batch, predicted_batch)

    # Computes gradients using loss & model variables. Apply the computed gradients on model variables using optimizer.
    gradients = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Computes accuracy score for the current batch.
    batch_accuracy = accuracy_function(target_batch, predicted_batch)

    # Computes mean for loss and accuracy.
    train_loss(batch_loss)
    train_accuracy(batch_accuracy)
