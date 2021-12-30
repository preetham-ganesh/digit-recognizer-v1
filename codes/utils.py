# authors_name = 'Preetham Ganesh'
# project_title = 'Kaggle - Digit Recognizer'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


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
    if 'label' in list(new_data.columns):
        # Splits the data into the input and target data
        new_input_data = new_data.drop(columns=['label'])
        new_target_data = list(new_data['label'])
    else:
        new_input_data = new_data

    # Reshapes the input data, converts into float32 format, normalizes the pixel values and converts to tensor.
    new_input_data = np.array(new_input_data).reshape((len(new_input_data), 28, 28, 1)).astype('float32') / 255
    new_input_data = tf.convert_to_tensor(new_input_data)

    if 'label' in list(new_data.columns):
        new_target_data = tf.keras.utils.to_categorical(new_target_data)
        return new_input_data, new_target_data

    return new_input_data


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
    """Sequential model for the 2nd configuration of hyperparameters used for developing the Digit Recognizer.

        Args:
            None

        Returns:
            Tensorflow compiled model for the 2nd configuration of hyperparameters.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def choose_model(model_number: int):
    """Initializes an object for the Digit Recognizer model classes based on the model number.

        Args:
            model_number: Integer which decides the model to be trained.

        Returns:
            Object for the Digit Recognizer model class.
    """
    if model_number == 1:
        model = digit_recognizer_1()
    elif model_number == 2:
        model = digit_recognizer_2()
    else:
        print('The model number entered is incorrect. Kindly enter the right model number')
        sys.exit()
    return model


def generate_model_history_plot(train_metrics: list,
                                validation_metrics: list,
                                metric_type: str,
                                model: int):
    """Generates plot for model training and validation history.

        Args:
            train_metrics: Training performance of the model for the current metric type.
            validation_metrics: Validation performance of the model for the current metric type.
            metric_type: String which will have either loss or accuracy as its value.
            model: Integer which decides the model to be trained.

        Returns:
            None
    """
    # Specifications used to generate the plot, i.e., font size and size of the plot.
    font = {'size': 28}
    plt.rc('font', **font)
    figure(num=None, figsize=(20, 10))

    # Generates plot for training and validation metrics
    epochs = [i for i in range(1, len(train_metrics) + 1)]
    plt.plot(epochs, train_metrics, color='orange', linewidth=3, label='train_{}'.format(metric_type))
    plt.plot(epochs, validation_metrics, color='blue', linewidth=3, label='validation_{}'.format(metric_type))

    # Generates the plot for the epochs vs metrics.
    plt.xlabel('epochs')
    plt.ylabel('normalized_rainfall')
    plt.legend(loc='upper left')
    plt.grid(color='black', linestyle='-.', linewidth=2, alpha=0.3)
    plt.savefig('../results/model_{}/model_history_{}.png'.format(model, metric_type))
    plt.show()


def model_training_and_evaluation(new_train_input_data: tf.Tensor,
                                  new_train_target_data: tf.Tensor,
                                  new_validation_input_data: tf.Tensor,
                                  new_validation_target_data: tf.Tensor,
                                  new_test_input_data: tf.Tensor,
                                  new_test_target_data: tf.Tensor,
                                  configuration: dict):
    """Uses configuration to choose the model, train and evaluate the model.

        Args:
            new_train_input_data: Transformed input training data of shape(x, 28, 28, 1)
            new_train_target_data: Transformed target training data of shape (x, 10)
            new_validation_input_data: Transformed input validation data of shape(x, 28, 28, 1)
            new_validation_target_data: Transformed target validation data of shape (x, 10)
            new_test_input_data: Transformed input testing data of shape(x, 28, 28, 1)
            new_test_target_data: Transformed target testing data of shape (x, 10)
            configuration: Dictionary which contains details for training and testing of the model.

        Returns:
            None
    """
    # Chooses the model based configuration file.
    model = choose_model(configuration['model'])
    print(model.summary())
    print()

    # Creates model checkpoint callback to monitor and save best model.
    directory_path = '../results'
    checkpoint_directory = '{}/{}_{}/{}'.format(directory_path, 'model', configuration['model'],
                                                'checkpoint_directory/checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_directory,
                                                                   save_weights_only=True, monitor='val_loss',
                                                                   mode='min', verbose=1)
    model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5, verbose=1,
                                                            mode='min', restore_best_weights=False)

    # Trains model using the training data, validates using the validation data, & uses callback to save the best model.
    history = model.fit(x=new_train_input_data, y=new_train_target_data, batch_size=configuration['batch_size'],
                        epochs=configuration['epochs'], verbose=2,
                        callbacks=[model_checkpoint_callback, model_early_stopping],
                        validation_data=(new_validation_input_data, new_validation_target_data))
    print()

    # Generates plot for model training and validation history.
    generate_model_history_plot(history.history['loss'], history.history['val_loss'], 'loss', configuration['model'])
    generate_model_history_plot(history.history['accuracy'], history.history['val_accuracy'], 'accuracy',
                                configuration['model'])

    # Evaluates the model using the testing data.
    test_score = model.evaluate(new_test_input_data, new_test_target_data, verbose=0)
    print('Test loss: {}'.format(round(test_score[0], 3)))
    print('Test accuracy: {}'.format(round(test_score[1], 3)))


def predict(model_number: int,
            images: tf.Tensor,
            directory_path: str):
    """Predicts the labels for the images given as input by using the model_number to restore the corresponding weights.

        Args:
            model_number: Integer that decides which model's weights has to be restored.
            images: Transformed input image data.
            directory_path: Parent folder where the model checkpoint can be found.

        Returns:
             A list which contains the labels for all the images given as input.
    """
    # Chooses the model based model_number.
    model = choose_model(model_number)

    # Restores weights of the model based on the model_number.
    checkpoint_directory = '{}/{}_{}/{}'.format(directory_path, 'model', model_number,
                                                'checkpoint_directory/checkpoint')
    model.load_weights(checkpoint_directory)

    # Performs predictions based on the trained model and saves the index of maximum value for each image.
    predictions = model.predict(images)
    predicted_labels = [np.argmax(predictions[i]) for i in range(len(predictions))]
    return predicted_labels


def create_submission(predicted_labels: list):
    """Creates a pandas dataframe based on the predicted_labels and converts it into a CSV file.

        Args:
            predicted_labels: A list which contains the labels for all the images given as input.

        Returns:
            None.
    """
    submission_dict = dict()
    submission_dict['ImageId'] = [i for i in range(1, len(predicted_labels) + 1)]
    submission_dict['Label'] = predicted_labels
    submission = pd.DataFrame(submission_dict, columns=['ImageId', 'Label'])
    submission.to_csv('../results/submission.csv', index=False)
