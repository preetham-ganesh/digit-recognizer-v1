# authors_name = 'Preetham Ganesh'
# project_title = 'Kaggle - Digit Recognizer'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf
import os
import logging
import pandas as pd
from utils import data_splitting, data_preprocessing, model_training_and_evaluation, predict, create_submission


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def main():
    print()
    directory_path = '../data/original_data'
    original_train_data = pd.read_csv('{}/{}'.format(directory_path, 'train.csv'))
    original_test_data = pd.read_csv('{}/{}'.format(directory_path, 'test.csv'))
    print('{}: {}'.format('No. of rows in the original training data', str(len(original_train_data))))
    print('{}: {}'.format('No. of rows in the original testing data', str(len(original_test_data))))
    print()
    new_train_data, new_validation_data, new_test_data = data_splitting(original_train_data)
    print('{}: {}'.format('No. of rows in the new training data', str(len(new_train_data))))
    print('{}: {}'.format('No. of rows in the new validation data', str(len(new_validation_data))))
    print('{}: {}'.format('No. of rows in the new testing data', str(len(new_test_data))))
    print()
    new_train_input_data, new_train_target_data = data_preprocessing(new_train_data)
    new_validation_input_data, new_validation_target_data = data_preprocessing(new_validation_data)
    new_test_input_data, new_test_target_data = data_preprocessing(new_test_data)
    print('{}: {}'.format('Shape of new input training data', new_train_input_data.shape))
    print('{}: {}'.format('Shape of new input validation data', new_validation_input_data.shape))
    print('{}: {}'.format('Shape of new input testing data', new_test_input_data.shape))
    print('{}: {}'.format('Shape of new target training data', new_train_target_data.shape))
    print('{}: {}'.format('Shape of new target validation data', new_validation_target_data.shape))
    print('{}: {}'.format('Shape of new target testing data', new_test_target_data.shape))
    print()
    batch_size = 64
    configuration = {'batch_size': batch_size, 'epochs': 100, 'model': 1}
    print()
    print('Model Training Started')
    print()
    model_training_and_evaluation(new_train_input_data, new_train_target_data, new_validation_input_data,
                                  new_validation_target_data, new_test_input_data, new_test_target_data,
                                  configuration)
    print()
    original_test_input_data = data_preprocessing(original_test_data)
    predicted_labels = predict(configuration['model'], original_test_input_data)
    create_submission(predicted_labels)


if __name__ == '__main__':
    main()

