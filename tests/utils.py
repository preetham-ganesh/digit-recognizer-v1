# Copyright (c) Preetham Ganesh.


import os
import sys
import warnings


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")


import pytest

from src.digit_recognizer.utils import check_directory_path_existence
from src.digit_recognizer.utils import load_json_file
from src.digit_recognizer.utils import save_json_file


@pytest.mark.test_check_directory_path_existence
def test_check_directory_path_existence() -> None:
    """Test cases for chech_directory_path_existence function.

    Test cases for chech_directory_path_existence function.

    Args:
        None.

    Returns:
        None.
    """
    home_directory_path = os.getcwd()

    # Test case 1
    absolute_directory_path = check_directory_path_existence("logs")
    assert absolute_directory_path, "{}/logs".format(home_directory_path)

    # Test case 2
    absolute_directory_path = check_directory_path_existence("plots")
    assert absolute_directory_path, "{}/plots".format(home_directory_path)


@pytest.mark.test_save_load_json_file
def test_save_load_json_file() -> None:
    """Test cases for load & save JSON file functions.

    Test cases for load & save JSON file functions.

    Args:
        None.

    Returns:
        None.
    """
    # Test case 1
    dictionary = {"a": [1, 2, 3, 4], "b": [2, 3, 4]}
    save_json_file(dictionary, "test_1", "tests")
    loaded_dictionary = load_json_file("test_1", "tests")
    assert dictionary, loaded_dictionary
