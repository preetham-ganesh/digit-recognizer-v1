# authors_name = 'Preetham Ganesh'
# project_title = 'Kaggle - Digit Recognizer'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf


def loss_function(actual_values: tf.Tensor,
                  predicted_values: tf.Tensor):
    """Calculates loss for the current batch of actual values and the predicted values.

        Args:
            actual_values: Actual classes the images belong to.
            predicted_values: Classes predicted by the neural network model for each image in batch.

        Return:
            Loss value computed for the current batch.
    """

    # Initializes the object for calculating the loss.
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

    # Mask off the losses on padding and computes the loss.
    mask = tf.math.logical_not(tf.math.equal(actual_values, 0))
    loss_ = loss_object(actual_values, predicted_values)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


