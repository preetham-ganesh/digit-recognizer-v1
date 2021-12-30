# authors_name = 'Preetham Ganesh'
# project_title = 'Kaggle - Digit Recognizer'
# email = 'preetham.ganesh2015@gmail.com'


import os
from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import logging
from codes.utils import predict
import numpy as np
import cv2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

app = Flask(__name__)
app_root_directory = os.path.dirname(os.path.abspath(__file__))


def predict_image_label(uploaded_image: np.ndarray):
    """Performs processing on the uploaded image and predict label for the processed image.

        Args:
            uploaded_image: Image uploaded by the user for predicting the label

        Returns:
            An integer which is the label predicted by the model.
    """
    # Resizes image to 28, 28.
    uploaded_resized_image = cv2.resize(uploaded_image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

    # Convert the color image to grayscale format.
    uploaded_image_grayscale = cv2.cvtColor(uploaded_resized_image, cv2.COLOR_BGR2GRAY)

    # Reshapes the grayscaled image to neural network input shape, and converts it into Tensor.
    uploaded_image_grayscale_data = np.reshape(uploaded_image_grayscale, (1, 28, 28, 1))
    uploaded_image_grayscale_data = tf.convert_to_tensor(uploaded_image_grayscale_data)

    # Predicts label using the model.
    label = predict(2, uploaded_image_grayscale_data, 'results')
    return label[0]


@app.route('/index', methods=['POST'])
def image_upload():
    """Uploads image from directory, performs preprocessing and predicts the label.

        Args:
            None.

        Returns:
            Rendered template for the complete page with image and the predicted label..

    """
    # Reads uploaded image from the directory using filename.
    directory_path = '{}/{}'.format(app_root_directory, 'data/images')
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    uploaded_file = request.files['upload_file']
    uploaded_image = cv2.imread('{}/{}'.format(directory_path, uploaded_file.filename))

    # Predicts label.
    label = predict_image_label(uploaded_image)
    return render_template('complete.html', image_name=uploaded_file.filename, c=label)


@app.route('/upload/<filename>')
def send_image(filename):
    """Sends saved image from directory using uploaded image filename.

        Args:
            filename: A string which contains the filename for the uploaded image.

        Returns:
            Saved image from directory.
    """
    return send_from_directory('data/images', filename)


@app.route("/")
def index():
    """Renders template for index page.

        Args:
            None.

        Returns:
            Rendered template for the index page.
    """
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
