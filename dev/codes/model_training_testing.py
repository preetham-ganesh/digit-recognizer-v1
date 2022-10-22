# authors_name = 'Preetham Ganesh'
# project_title = 'Digit Recognizer'
# email = 'preetham.ganesh2021@gmail.com'


from utils import save_json_file
from utils import create_log
from utils import log_information
from utils import set_physical_devices_memory_limit
from utils import load_dataset
from utils import data_splitting
from utils import data_preprocessing
from utils import shuffle_slice_dataset
from utils import model_training_validation
from utils import model_testing
from utils import process_input_batch
from utils import predict
from utils import create_submission


def main():
    log_information("")

    # Sets memory limit of GPU if found in the system.
    set_physical_devices_memory_limit()

    # Loads the original Digit Recognizer dataset downloaded from Kaggle.
    original_train_data, original_test_data = load_dataset()
    log_information(
        "No. of rows in the original train data: {}".format(len(original_train_data))
    )
    log_information(
        "No. of rows in the original test data: {}".format(len(original_test_data))
    )
    log_information("")

    # Splits the original training data into train, validation and test dataset.
    n_validation_examples = 1000
    n_test_examples = 1000
    new_train_data, new_validation_data, new_test_data = data_splitting(
        original_train_data, n_validation_examples, n_test_examples
    )
    log_information("No. of rows in the new train data: {}".format(len(new_train_data)))
    log_information(
        "No. of rows in the new validation data: {}".format(len(new_validation_data))
    )
    log_information("No. of rows in the new test data: {}".format(len(new_test_data)))
    log_information("")

    # Converts data split information into input and target data.
    new_train_input_data, new_train_target_data = data_preprocessing(new_train_data)
    new_validation_input_data, new_validation_target_data = data_preprocessing(
        new_validation_data
    )
    new_test_input_data, new_test_target_data = data_preprocessing(new_test_data)
    log_information(
        "Shape of new input train data: {}".format(new_train_input_data.shape)
    )
    log_information(
        "Shape of new target train data: {}".format(new_train_target_data.shape)
    )
    log_information(
        "Shape of new input validation data: {}".format(new_validation_input_data.shape)
    )
    log_information(
        "Shape of new target validation data: {}".format(
            new_validation_target_data.shape
        )
    )
    log_information(
        "Shape of new input test data: {}".format(new_test_input_data.shape)
    )
    log_information(
        "Shape of new target test data: {}".format(new_test_target_data.shape)
    )
    log_information("")

    # Creates model configuration for training the model.
    batch_size = 64
    model_configuration = {
        "epochs": 100,
        "batch_size": batch_size,
        "model_version": version,
        "learning_rate": 0.001,
        "final_image_size": 28,
        "n_channels": 1,
        "train_steps_per_epoch": len(new_train_input_data) // batch_size,
        "validation_steps_per_epoch": len(new_validation_input_data) // batch_size,
        "test_steps_per_epoch": len(new_test_input_data) // batch_size,
        "layers_arrangement": [
            "conv2d_0",
            "maxpool2d_0",
            "conv2d_1",
            "maxpool2d_1",
            "flatten",
            "dense_0",
            "dense_1",
            "dense_2",
        ],
        "layers_start_index": 0,
        "layers_configuration": {
            "conv2d_0": {
                "filters": 8,
                "kernel": 4,
                "padding": "valid",
                "activation": "relu",
                "strides": 1,
                "kernel_initializer": "glorot_uniform",
            },
            "maxpool2d_0": {"pool_size": (2, 2), "strides": 2, "padding": "valid"},
            "conv2d_1": {
                "filters": 16,
                "kernel": 4,
                "padding": "valid",
                "activation": "relu",
                "strides": 1,
                "kernel_initializer": "glorot_uniform",
            },
            "maxpool2d_1": {"pool_size": (2, 2), "strides": 2, "padding": "valid"},
            "flatten": {},
            "dense_0": {"units": 128, "activation": "relu"},
            "dense_1": {"units": 64, "activation": "relu"},
            "dense_2": {"units": 10, "activation": "softmax"},
        },
    }
    save_json_file(
        model_configuration, "model_configuration", "results/v{}/utils".format(version)
    )
    log_information("")

    # Shuffles input and target data. Converts into tensorflow datasets.
    validation_dataset = shuffle_slice_dataset(
        new_validation_input_data, new_validation_target_data, batch_size
    )
    del new_validation_input_data, new_validation_target_data
    test_dataset = shuffle_slice_dataset(
        new_test_input_data, new_test_target_data, batch_size
    )
    del new_test_input_data, new_test_target_data
    train_dataset = shuffle_slice_dataset(
        new_train_input_data, new_train_target_data, batch_size
    )
    del new_train_input_data, new_train_target_data
    log_information("Shuffled & Sliced the datasets.")
    log_information("")
    log_information(
        "No. of Training steps per epoch: {}".format(
            model_configuration["train_steps_per_epoch"]
        )
    )
    log_information(
        "No. of Validation steps per epoch: {}".format(
            model_configuration["validation_steps_per_epoch"]
        )
    )
    log_information(
        "No. of Testing steps: {}".format(model_configuration["test_steps_per_epoch"])
    )
    log_information("")

    # Trains and validation the model.
    model_training_validation(train_dataset, validation_dataset, model_configuration)
    log_information("")

    # Tests the trained model on test dataset.
    model_testing(test_dataset, model_configuration)
    log_information("")

    # Preprocesses the original test input data.
    original_test_input_data = data_preprocessing(original_test_data)
    log_information("Finished preprocessing original test data.")
    log_information("")

    # Predicts labels for images in the original test input data using current model.
    predicted_labels = predict(original_test_input_data, model_configuration)
    log_information("Finished predicting labels original test data.")
    log_information("")

    # Creates a submission using the predicted labels.
    create_submission(predicted_labels, model_configuration)
    log_information("Finished saving submission for current model.")
    log_information("")


if __name__ == "__main__":
    major_version = 1
    minor_version = 0
    revision = 0
    global version
    version = "{}.{}.{}".format(major_version, minor_version, revision)
    create_log("logs", "model_training_testing_v{}.log".format(version))
    main()
