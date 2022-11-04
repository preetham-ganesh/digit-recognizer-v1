# authors_name = 'Preetham Ganesh'
# project_title = 'Digit Recognizer'
# email = 'preetham.ganesh2021@gmail.com'


import logging

from utils.utils import check_directory_path_existence


def create_log(logger_directory_path: str, log_file_name: str) -> None:
    """Creates an object for logging terminal output.

    Args:
        logger_directory_path: A string which contains the location where the log file should be stored.
        log_file_name: A string which contains the name for the log file.

    Returns:
        None.
    """
    # Type checks arguments.
    assert isinstance(logger_directory_path, str), "Argument is not of type string."
    assert isinstance(log_file_name, str), "Argument is not of type string."

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
    """Saves current log information, and prints it in terminal.

    Args:
        log: A string which contains the information that needs to be printed in terminal and saved in log.

    Returns:
        None.

    Exception:
        NameError: When the logger is not defined, this exception is thrown.
    """
    # Type checks arguments.
    assert isinstance(log, str), "Argument is not of type string."

    # Adds current log into log file.
    try:
        logger.info(log)
    except NameError:
        _ = ""
    print(log)
