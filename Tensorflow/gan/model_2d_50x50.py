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

model_config = {
    "data_dimension" : (50,50,1),
    "generator_output_dimension" : (50,50,1),
    "noise_dimension" : (100,),
    "output_in_3D" : False,
    "scale_free" : False
}

def make_generator(noise_dim=model_config["noise_dimension"]):
    """Creates a generator model that takes a noise vector as a "seed",
    and outputs images of size m*m*1."""
    model = Sequential(name="generator")

    model.add( Dense(200, input_dim=noise_dim[0]))
    model.add( LeakyReLU(alpha=0.2))

    model.add( Reshape((5, 5, 8), 
                       input_shape=noise_dim))

    model.add( Conv2DTranspose(512, (5, 5), 
                               strides=2, 
                               padding='same', 
                               kernel_initializer='he_normal'))
    
    model.add( BatchNormalization(momentum=0.99))
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2DTranspose(256, (5, 5), 
                               strides=5, 
                               padding='same', 
                               kernel_initializer='he_normal'))
    
    model.add( BatchNormalization(momentum=0.99))
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2D(128, (5, 5), 
                      padding='same', 
                      kernel_initializer='he_normal'))
    
    model.add( BatchNormalization(momentum=0.99))
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2DTranspose(64, (5, 5), 
                               padding='same', 
                               kernel_initializer='he_normal'))
    
    model.add( BatchNormalization(momentum=0.99))
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2D(1, (5, 5), 
                      padding='same', 
                      activation='tanh', 
                      kernel_initializer='he_normal'))
    return model

def make_critic(input_dim=model_config["data_dimension"]):
    # It is important in the WGAN-GP algorithm to NOT use batch normalization on the critic
    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.

    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""
    
    model = Sequential(name="critic")
    model.add(Conv2D(256, (3, 3), 
                     padding='same', 
                     input_shape=input_dim))
    
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (5, 5), 
                     kernel_initializer='he_normal', 
                     strides=[2, 2]))
    
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64, (5, 5), 
                     kernel_initializer='he_normal', 
                     padding='same', 
                     strides=[2, 2]))
    
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())

    model.add(Dense(1, 
                    kernel_initializer='he_normal'))
    return model
