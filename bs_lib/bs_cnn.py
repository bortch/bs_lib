import tensorflow as tf

# from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPool2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_generators(
    set_names=["train_set", "test_set", "val_set"],
    path="",
    image_data_generator_params={},
    flow_params={},
):
    """get ImageDataGenerator

    Args:
        set_names (list, optional): List of set's name/directory name. Defaults to ['train_set','test_set','val_set'].
        image_data_generator_params (object,optional): see Tensorflow docs for options. Defaults to {}
          example {'rescale':1./255,
          'horizontal_flip':True,
          'vertical_flip':True,
          'rotation_range':75,
          'brightness_range':(0.1, 5.)}
        flow_params (dict, optional): see Tensorflow docs for flow's options. Defaults to {}. Example {'target_size':(32,32),'batch_size':32}

    Returns:
        dict: dict of generator, keys are the set_names
    """
    generators = dict()

    for s in range(len(set_names)):
        path_ = f"{path}{set_names[s]}"
        generators[set_names[s]] = ImageDataGenerator(
            **image_data_generator_params
        ).flow_from_directory(path_, **flow_params)

    return generators


def get_cnn(
    n_classes=10,
    kernel_size=2,
    filters_size=[16, 32, 64],
    pooling_size=2,
    layers=[8, 16],
    input_shape=(32, 32, 3),
    activation="relu",
    loss="categorical_crossentropy",
):
    """Create a Keras Sequential Convolution Neural Network

    Args:
        n_classes (int, optional): Numbers of classes for output layers classification. Defaults to 10.
        kernel_size (int, optional): The Convolution kernel Size. Defaults to 2.
        filters_size (list, optional): List of numbers of filter by convolution layer. It creates the appropriate number of convolution layers. Defaults to [16,32,64].
        pooling_size (int, optional): The pooling size (reduce the image by factor of pooling size). Defaults to 2.
        layers (list, optional): List of neurone by layers. It creates the adequate numbers of dense layers. Defaults to [8, 16].
        input_shape (tuple, optional): Image shape. Defaults to (32, 32, 3).
        activation (str, optional): Dense and Conv2D type of neurone activation function. Defaults to 'relu'.
        loss (str, optional): The model loss to minimize. Defaults to 'categorical_crossentropy'.

    Returns:
        classe: return a compiled Keras Sequential model CNN
    """

    model = keras.Sequential()
    for f in range(len(filters_size)):
        model.add(
            Conv2D(
                filters=filters_size[f],
                kernel_size=kernel_size,
                activation=activation,
                input_shape=input_shape,
            )
        )
        model.add(MaxPool2D(pooling_size))
    model.add(Flatten())

    for l in range(len(layers)):
        model.add(Dense(layers[l], activation=activation))

    model.add(Dense(n_classes, activation="softmax"))

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


# testing
if __name__ == "__main__":
    print("please do some tests")

    param_grid = {
        "batch_size": [10, 20, 40, 60, 80, 100],
        "epochs": [10, 50, 100],
        "optimizer": [
            "SGD",
            "RMSprop",
            "Adagrad",
            "Adadelta",
            "Adam",
            "Adamax",
            "Nadam",
        ],
        "learn_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "momentum": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
        "init_mode": [
            "uniform",
            "lecun_uniform",
            "normal",
            "zero",
            "glorot_normal",
            "glorot_uniform",
            "he_normal",
            "he_uniform",
        ],
        "activation": [
            "softmax",
            "softplus",
            "softsign",
            "relu",
            "tanh",
            "sigmoid",
            "hard_sigmoid",
            "linear",
        ],
        "weight_constraint": [1, 2, 3, 4, 5],
        "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "neurons": [1, 5, 10, 15, 20, 25, 30],
    }

    cnn_desc = {
        "n_classes": 7,
        "kernel_size": 3,
        "filters_size": [16, 32, 64, 128, 256],
        "pooling_size": 2,
        "dropout": 0.1,
        "layers": [8, 8],
        "input_shape": (img_width, img_height, 1),
        "activation": "relu",
        "loss": "categorical_crossentropy",
    }
