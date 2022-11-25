# Copyrights (c) Preetham Ganesh.


import tensorflow as tf


class DigitRecognizer(tf.keras.Model):
    """Recognizes number in an image.

    Attributes:
        model_configuration: A dictionary which contains the configuration of each layer in the model.
        model_layers: A dictionary which contains the configured layers.
    """

    def __init__(self, model_configuration: dict) -> None:
        """Initializes the layers in the recognition model, by adding convolutional, pooling, dropout & dense layers."""

        super(DigitRecognizer, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration
        self.model_layers = dict()

        # Iterates across layers arrangement in model configuration to add layers to the model.
        for layer_name in self.model_configuration["layers_arrangement"]:
            current_layer_configuration = self.model_configuration[
                "layers_configuration"
            ][layer_name]

            # If layer's name is like 'conv2d_', a Conv2D layer is initialized based on layer configuration.
            if layer_name.split("_")[0] == "conv2d":
                self.model_layers[layer_name] = tf.keras.layers.Conv2D(
                    filters=current_layer_configuration["filters"],
                    kernel_size=current_layer_configuration["kernel"],
                    padding=current_layer_configuration["padding"],
                    strides=current_layer_configuration["strides"],
                    activation=current_layer_configuration["activation"],
                    kernel_initializer=current_layer_configuration[
                        "kernel_initializer"
                    ],
                )

            # If layer's name is like 'maxpool2d_', a MaxPool2D layer is initialized based on layer configuration.
            elif layer_name.split("_")[0] == "maxpool2d":
                self.model_layers[layer_name] = tf.keras.layers.MaxPool2D(
                    pool_size=current_layer_configuration["pool_size"],
                    strides=current_layer_configuration["strides"],
                    padding=current_layer_configuration["padding"],
                )

            # If layer's name is like 'dropout_', a Dropout layer is initialized based on layer configuration.
            elif layer_name.split("_")[0] == "dropout":
                self.model_layers[layer_name] = tf.keras.layers.Dropout(
                    rate=current_layer_configuration["rate"]
                )

            # If layer's name is like 'dense_', a Dropout layer is initialized based on layer configuration.
            elif layer_name.split("_")[0] == "dense":
                self.model_layers[layer_name] = tf.keras.layers.Dense(
                    units=current_layer_configuration["units"],
                    activation=current_layer_configuration["activation"],
                )

            # If layer's name is like 'flatten_', a Flatten layer is initialized.
            elif layer_name.split("_")[0] == "flatten":
                self.model_layers[layer_name] = tf.keras.layers.Flatten()

            # If layer's name is like 'globalaveragepool2d_', a Global Average Pooling 2D layer is initialized.
            elif layer_name.split("_")[0] == "globalaveragepool2d":
                self.model_layers[layer_name] = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """Input tensor is passed through the layers in the encoder model."""
        # Asserts type & values of the input arguments.
        assert isinstance(x, tf.Tensor), "Variable x should be of type 'tf.Tensor'."
        assert isinstance(training, bool), "Variable training should be of type 'bool'."
        assert (
            x.shape[1] == self.model_configuration["final_image_size"]
            and x.shape[2] == self.model_configuration["final_image_size"]
            and x.shape[3] == self.model_configuration["n_channels"]
        ), "Variable x should be of shape (None, {}, {}, {}).".format(
            self.model_configuration["final_image_size"],
            self.model_configuration["final_image_size"],
            self.model_configuration["n_channels"],
        )
        assert (
            training == True or training == False
        ), "Variable training should be either 'True' or 'False'."

        # Iterates across the layers arrangement, and predicts the output for each layer.
        for layer_index in range(
            self.model_configuration["layers_start_index"],
            len(self.model_configuration["layers_arrangement"]),
        ):
            layer_name = self.model_configuration["layers_arrangement"][layer_index]

            # If layer's name is like 'dropout_' or 'batchnorm_' or 'mobilenet_', the following output is predicted.
            if layer_name.split("_")[0] == "dropout":
                x = self.model_layers[layer_name](x, training=training)

            # Else, the following output is predicted.
            else:
                x = self.model_layers[layer_name](x)
        return x

    def build_graph(self) -> tf.keras.Model:
        """Builds plottable graph for the model."""
        # Creates the input layer using the model configuration.
        x = tf.keras.layers.Input(
            shape=(
                self.model_configuration["final_image_size"],
                self.model_configuration["final_image_size"],
                self.model_configuration["n_channels"],
            )
        )

        # Creates an object for the tensorflow model and returns it.
        return tf.keras.Model(inputs=[x], outputs=self.call(x, False))
