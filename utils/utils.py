# Copyright © 2022 Preetham Ganesh.


import os

import json

from utils.logger import log_information


def check_directory_path_existence(directory_path: str) -> str:
    """Creates the absolute path for the directory path given in argument if it does not already exist.

    Args:
        directory_path: A string which contains the directory path that needs to be created if it does not already exist.

    Returns:
        A string which contains the absolute directory path.
    """
    # Type checks arguments.
    assert isinstance(directory_path, str), "Argument is not of type string."

    # Creates the following directory path if it does not exist.
    home_directory_path = os.getcwd()
    absolute_directory_path = "{}/{}".format(home_directory_path, directory_path)
    if not os.path.isdir(absolute_directory_path):
        os.makedirs(absolute_directory_path)
    return absolute_directory_path


def save_json_file(dictionary: dict, file_name: str, directory_path: str) -> None:
    """Converts a dictionary into a JSON file and saves it for future use.

    Args:
        dictionary: A dictionary which needs to be saved.
        file_name: A string which contains the name with which the file has to be saved.
        directory_path: A string which contains the path where the file needs to be saved.

    Returns:
        None.
    """
    # Types checks arguments.
    assert isinstance(dictionary, dict), "Argument is not of type dictionary."
    assert isinstance(file_name, str), "Argument is not of type string."
    assert isinstance(directory_path, str), "Argument is not of type string."

    # Creates the following directory path if it does not exist.
    directory_path = check_directory_path_existence(directory_path)

    # Saves the dictionary or list as a JSON file at the file path location.
    file_path = "{}/{}.json".format(directory_path, file_name)
    with open(file_path, "w") as out_file:
        json.dump(dictionary, out_file, indent=4)
    out_file.close()
    log_information("{} file saved successfully at {}.".format(file_name, file_path))


def load_json_file(file_name: str, directory_path: str) -> dict:
    """Loads a JSON file into memory based on the file_name.

    Args:
        file_name: A string which contains the name of the of the file to be loaded.
        directory_path: A string which contains the location where the directory path exists.

    Returns:
        A dictionary which contains the JSON file.
    """
    # Types checks input arguments.
    assert isinstance(file_name, str), "Argument is not of type string."
    assert isinstance(directory_path, str), "Argument is not of type string."

    # Loads dictionary 
    file_path = "{}/{}.json".format(directory_path, file_name)
    with open(file_path, "r") as out_file:
        dictionary = json.load(out_file)
    out_file.close()

    # Type checks output.
    assert isinstance(dictionary, dict), "Argument is not of type dictionary."
    return dictionary
