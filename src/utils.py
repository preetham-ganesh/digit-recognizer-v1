# Copyright (c) Preetham Ganesh.


import os
import logging
import warnings


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
warnings.filterwarnings("ignore")


import json
import tensorflow as tf


def check_directory_path_existence(directory_path: str) -> str:
    """Creates the directory path.

    Creates the absolute path for the directory path given in argument if it does not already exist.

    Args:
        directory_path: A string for the directory path that needs to be created if it does not already exist.

    Returns:
        A string for the absolute directory path.
    """
    # Asserts type of arguments.
    assert isinstance(
        directory_path, str
    ), "Variable directory_path should be of type 'str'."

    # Creates the following directory path if it does not exist.
    home_directory_path = os.getcwd()
    absolute_directory_path = "{}/{}".format(home_directory_path, directory_path)
    if not os.path.isdir(absolute_directory_path):
        os.makedirs(absolute_directory_path)
    return absolute_directory_path


def create_log(log_file_name: str, logger_directory_path: str) -> None:
    """Creaters a logger.

    Creates an object for logging terminal output.

    Args:
        log_file_name: A string for the name for the log file.
        logger_directory_path: A string for the location where the log file should be stored.

    Returns:
        None.
    """
    # Asserts type of arguments.
    assert isinstance(
        logger_directory_path, str
    ), "Logger_directory_path should be of type 'str'."
    assert isinstance(
        log_file_name, str
    ), "Variable log_file_name should be of type 'str'."

    # Checks if the following path exists.
    logger_directory_path = check_directory_path_existence(logger_directory_path)

    # Create and configure logger
    logging.basicConfig(
        filename="{}/{}.log".format(logger_directory_path, log_file_name),
        format="%(asctime)s %(message)s",
        filemode="w",
    )
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def log_information(log: str) -> None:
    """Logs current information.

    Saves current log information, and prints it in terminal.

    Args:
        log: A string for the information that needs to be printed in terminal and saved in log.

    Returns:
        None.

    Exception:
        NameError: When the logger is not defined, this exception is thrown.
    """
    # Type checks arguments.
    assert isinstance(log, str), "Variable log should be of type 'str'."

    # Adds current log into log file.
    try:
        logger.info(log)
    except NameError:
        _ = ""
    print(log)


def save_json_file(dictionary: dict, file_name: str, directory_path: str) -> None:
    """Saves dictionary as a JSON file.

    Converts a dictionary into a JSON file and saves it for future use.

    Args:
        dictionary: A dictionary which needs to be saved.
        file_name: A string for the name with which the file has to be saved.
        directory_path: A string for the path where the file needs to be saved.

    Returns:
        None.
    """
    # Types checks arguments.
    assert isinstance(dictionary, dict), "Variable dictionary should be of type 'dict'."
    assert isinstance(file_name, str), "Variable file_name should be of type 'str'."
    assert isinstance(
        directory_path, str
    ), "Variable directory_path should be of type 'str'."

    # Checks if the following path exists.
    directory_path = check_directory_path_existence(directory_path)

    # Saves the dictionary or list as a JSON file at the file path location.
    file_path = "{}/{}.json".format(directory_path, file_name)
    with open(file_path, "w") as out_file:
        json.dump(dictionary, out_file, indent=4)
    out_file.close()
    log_information("{} file saved successfully at {}.".format(file_name, file_path))


def load_json_file(file_name: str, directory_path: str) -> dict:
    """Loads a JSON file as a dictionary.

    Loads a JSON file as a dictionary into memory based on the file_name.

    Args:
        file_name: A string for the name of the of the file to be loaded.
        directory_path: A string for the location where the directory path exists.

    Returns:
        A dictionary loaded from the JSON file.
    """
    # Types checks input arguments.
    assert isinstance(file_name, str), "Variable file_name should be of type 'str'."
    assert isinstance(
        directory_path, str
    ), "Variable directory_path should be of type 'str'."

    # Loads dictionary
    file_path = "{}/{}.json".format(directory_path, file_name)
    with open(file_path, "r") as out_file:
        dictionary = json.load(out_file)
    out_file.close()

    # Type checks output.
    assert isinstance(dictionary, dict), "Variable dictionary should be of type 'dict'."
    return dictionary


def set_physical_devices_memory_limit() -> None:
    """Sets memory limit of GPU if found in the system.

    Sets memory limit of GPU if found in the system.

    Args:
        None.

    Returns:
        None.
    """
    # Lists physical devices in the system.
    gpu_devices = tf.config.list_physical_devices("GPU")

    # If GPU device is found in the system, then the memory limit is set.
    if len(gpu_devices) > 0:
        tf.config.experimental.set_memory_growth(gpu_devices[0], enable=True)
        gpu_available = True
    else:
        gpu_available = False

    if gpu_available:
        log_information("GPU is available and will be used as accelerator.")
    else:
        log_information(
            "GPU is not available, hence the model will be executed on CPU."
        )
    log_information("")
