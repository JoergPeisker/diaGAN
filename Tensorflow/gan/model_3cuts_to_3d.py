"""
WGAN generator and critic model for 50x50 images dataset
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    BatchNormalization,
    LeakyReLU,
    Activation
)
import numpy as np

model_config = {
    "data_dimension" : (3,64,64),
    "generator_output_dimension" : (64,64,64),
    "noise_dimension" : (256,),
    "output_in_3D" : True,
    "cuts" : True,
    "scale_free" : False
}

def make_generator(noise_dim=model_config["noise_dimension"]):
    model = Sequential(name="generator")

    # Starting the model with the noise dimension
    model.add(Reshape((4, 4, 4, 4), input_shape=noise_dim))

    # Using Conv3DTranspose (also known as Deconvolution) for the generator model
    model.add(Conv3DTranspose(512, (3, 3, 3), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())  # Removed axis specification
    model.add(LeakyReLU(0.2))

    model.add(Conv3DTranspose(256, (5, 5, 5), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())  # Removed axis specification
    model.add(LeakyReLU(0.2))

    model.add(Conv3DTranspose(128, (5, 5, 5), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())  # Removed axis specification
    model.add(LeakyReLU(0.2))

    model.add(Conv3DTranspose(64, (5, 5, 5), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())  # Removed axis specification
    model.add(LeakyReLU(0.2))

    # Final Conv3D layer
    model.add(Conv3D(1, (5, 5, 5), padding='same', activation='tanh', kernel_initializer='he_normal'))

    # Adjusting the shape of the output to match the configuration
    model.add(Reshape(model_config["generator_output_dimension"]))
    return model

def make_critic(input_dim=model_config["data_dimension"]):
    """It is important in the WGAN-GP algorithm to NOT use batch normalization on the critic"""
    model = Sequential(name="critic")

    model.add(Reshape((input_dim[0] * input_dim[1], input_dim[2], 1), input_shape=input_dim))

    model.add(Conv2D(64, (3, 3), padding='same', strides=2))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', strides=2))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, (5, 5), kernel_initializer='he_normal', padding='same', strides=2))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', strides=2))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())

    model.add(Dense(1, kernel_initializer='he_normal'))  # The critic's output (real or fake)
    return model
