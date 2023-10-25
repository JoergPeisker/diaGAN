from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Reshape, Lambda
from tensorflow.keras.layers import Concatenate
import numpy as np
import tensorflow as tf

class AverageSampler(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(AverageSampler, self).__init__(**kwargs)  # It's a good practice to forward arguments
        self.size = size

    def call(self, inputs):
        # We should verify that 'inputs' is a list or tuple of length 2
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("AverageSampler expects a list of two inputs tensors")

        weights = tf.random.uniform((self.size, 1, 1, 1))
        result = (weights * inputs[0]) + ((1 - weights) * inputs[1])
        return result

    def compute_output_shape(self, input_shape):
        # If all input shapes are the same, you can return the first one
        if isinstance(input_shape, list) and len(input_shape) > 0:
            return input_shape[0]
        else:
            raise ValueError("Expected a list of input shapes")

    def get_config(self):
        # Implementing get_config is important for saving and loading a model with custom layers
        config = super(AverageSampler, self).get_config()
        config.update({"size": self.size})
        return config

class CutSampler(Layer):
    """
    From (N,N,N) volume data to several (N,N) cuts. Output data is of shape (self.dim,N,N)
    """
    def __init__(self, dim, **kwargs):
        assert dim in [2,3]
        self.dim = dim
        super(CutSampler, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        x,y,z = shape[1:]
        return (None,self.dim,x,y)

    def call(self, inputs):
        s = inputs.shape
        if self.dim==2:
            x = K.random_uniform((1,), 0, s[1], dtype="int32")[0]
            y = K.random_uniform((1,), 0, s[2], dtype="int32")[0]
            output = K.stack([inputs[:,x,:,:], inputs[:,:,y,:]])
            return K.permute_dimensions(output, (1,0,2,3))
        elif self.dim==3:
            x = K.random_uniform((1,), 0, s[1], dtype="int32")[0]
            y = K.random_uniform((1,), 0, s[2], dtype="int32")[0]
            z = K.random_uniform((1,), 0, s[3], dtype="int32")[0]
            output = K.stack([inputs[:,x,:,:], inputs[:,:,y,:], inputs[:,:,:,z]])
            return K.permute_dimensions(output, (1,0,2,3))


def L2DistanceLayer():
    stack = lambda x : K.stack([x[0],x[1]], axis=1)
    norm2 = lambda x : K.sum(K.square(x[0]-x[1]))
    return Lambda(lambda x: norm2(stack(x)), name="L2_distance")


def L1DistanceLayer():
    stack = lambda x : K.stack([x[0],x[1]], axis=1)
    norm1 = lambda x : K.sum(K.abs(x[0]-x[1]))
    return Lambda(lambda x: norm1(stack(x)), name="L1_distance")


def StyleLossLayer(weight=1., name="Style_loss"):

    def style_loss(real, generated, weight):
        nb_channels = K.cast(K.shape(real)[-1], 'float32')
        w = K.cast(K.shape(real)[-2], 'float32')
        h = K.cast(K.shape(real)[-3], 'float32')
        normfact = nb_channels*4*w*w*h*h
        f_real = K.batch_flatten(K.permute_dimensions(real, (0, 3, 1, 2)))
        f_gen = K.batch_flatten(K.permute_dimensions(generated, (0, 3, 1, 2)))
        gram_real = K.dot(f_real, K.transpose(f_real))/normfact
        gram_gen = K.dot(f_gen, K.transpose(f_gen))/normfact
        return weight*K.mean(K.square(gram_real - gram_gen), keepdims=True)

    return Lambda(lambda x : style_loss(x[0], x[1], weight), name=name)


def VGG19Layer(VGGinput_dim, feature_layers=['block1_conv1']):
    from keras.applications.vgg19 import VGG19
    input = Input(shape=VGGinput_dim)
    s = input.shape
    if s[-1]==1:
        # The image must have 3 channels to get fed into the VGG
        img = Concatenate()([input]*3)
    else:
        img = input
    vgg_model = VGG19(include_top=False, weights='imagenet')
    vgg_model.outputs = [vgg_model.get_layer(name).output for name in feature_layers]
    model = Model(inputs=input, outputs=vgg_model(img), name="VGG19")
    model.trainable = False
    return model
