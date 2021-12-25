# authors_name = 'Preetham Ganesh'
# project_title = 'Kaggle Competitions - Digit Recognizer'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf


class DigitRecognizer1(tf.keras.Model):
    """Model Sub-class API for the 1st configuration of hyperparameters used for developing the Digit Recognizer."""

    def __init__(self):
        super(DigitRecognizer1, self).__init__()
        """Initializes the layers used for training the model."""
        self.convolution_2d_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=4, padding='valid', strides=1,
                                                       activation='relu', input_shape=(28, 28, 1))
        self.max_pooling_2d_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        self.convolution_2d_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=4, padding='valid', strides=1,
                                                       activation='relu')
        self.max_pooling_2d_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, input_batch: tf.Tensor):
        """Passes the input batch through the initialized layers and generates the output classes."""
        output_batch = self.convolution_2d_1(input_batch)
        output_batch = self.max_pooling_2d_1(output_batch)
        output_batch = self.convolution_2d_2(output_batch)
        output_batch = self.max_pooling_2d_2(output_batch)
        output_batch = self.flatten(output_batch)
        output_batch = self.dense_1(output_batch)
        output_batch = self.dense_2(output_batch)
        output_batch = self.dense_3(output_batch)
        return output_batch


class DigitRecognizer2(tf.keras.Model):
    """Model Sub-class API for the 2nd configuration of hyperparameters used for developing the Digit Recognizer."""

    def __init__(self):
        super(DigitRecognizer2, self).__init__()
        """Initializes the layers used for training the model."""
        self.convolution_2d_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                                                       input_shape=(28, 28, 1))
        self.convolution_2d_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu')
        self.max_pooling_2d_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.dropout_1 = tf.keras.layers.dropout(rate=0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dropout_2 = tf.keras.layers.dropout(rate=0.5)
        self.dense_2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, input_batch: tf.Tensor,
             training: bool):
        """Passes the input batch through the initialized layers and generates the output classes."""
        output_batch = self.convolution_2d_1(input_batch)
        output_batch = self.convolution_2d_2(output_batch)
        output_batch = self.max_pooling_2d_1(output_batch)
        output_batch = self.dropout_1(output_batch, training=training)
        output_batch = self.flatten(output_batch)
        output_batch = self.dense_1(output_batch)
        output_batch = self.dropout_2(output_batch, training=training)
        output_batch = self.dense_2(output_batch)
        return output_batch
