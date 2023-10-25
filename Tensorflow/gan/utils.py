#from keras.utils import multi_gpu_model
#from keras import backend as K
import numpy as np
import os
#import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

def eye_loss(y_real, y_pred):
    """ Identity loss """
    return tf.reduce_sum(y_pred)

def wasserstein_loss(y_real, y_pred):
    """Calculates the Wasserstein loss for a sample batch.

    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.

    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.

    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""

    return tf.reduce_mean(y_real * y_pred)


class GradientPenaltyLayer(Layer):
    def __init__(self,critic_model, **kwargs):
        super(GradientPenaltyLayer, self).__init__(**kwargs)
        self.critic = critic_model
        
    def call(self, inputs):
        # Unpack the inputs
        y_pred, averaged_samples = inputs

        # We'll need to calculate gradients, and for this, we need a persistent gradient tape
        with tf.GradientTape() as tape:
            # This makes the tape watch the 'averaged_samples'
            tape.watch(averaged_samples)
            # Now, get the predictions for the 'averaged_samples'
            pred = self.critic(averaged_samples, training=True)

        # Compute the gradient of 'pred' w.r.t. the 'averaged_samples'
        grads = tape.gradient(pred, averaged_samples)
        print(type(grads))
        # Calculate the L2 norm of the gradients
        grads_sqr = tf.square(grads)
        grads_sqr_sum = tf.reduce_sum(grads_sqr, axis=np.arange(1, len(grads_sqr.shape)))
        grad_l2_norm = tf.sqrt(grads_sqr_sum)

        # Calculate the gradient penalty
        grad_penalty = tf.square(grad_l2_norm - 1)

        return grad_penalty

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)
    
def mean_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)
    
'''def gradient_penalty_loss(y_true, 
                          y_pred, 
                          averaged_samples):
    """
    Computes the gradient penalty term in the WGAN-GP loss function.
    This function requires the following parameters:
        y_true: Ground truth values (required by Keras, but not used in this function).
        y_pred: Predicted values, i.e., the outputs of the critic.
        averaged_samples: The random samples averaged between real and generated samples.
    """
    # Calculate the gradients using automatic differentiation
    with tf.GradientTape() as tape:
        tape.watch(averaged_samples)
        prediction = crit(averaged_samples)  # Ensure that the prediction depends on 'averaged_samples'

    gradients = tape.gradient(prediction, 
                              averaged_samples)[0]
    # First, we calculate the gradients based on the input averages samples and predictions
    #gradients = tf.gradients(y_pred, averaged_samples)[0]

    # Then, we compute the L2 norm of the gradients
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, 
                                      axis=np.arange(1, len(gradients_sqr.shape)))
    
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)

    # Finally, we calculate the gradient penalty based on the difference with 1
    gradient_penalty = tf.square(gradient_l2_norm - 1)

    return tf.reduce_mean(gradient_penalty)'''


def visualize_model(model):
    """
    Given a keras model, outputs its summary in the console.
    Also generates a .png summary file with graphviz
    """
    print("\n\n----------- MODEL SUMMARY : {} -----------\n".format(model.name))
    model.summary()
    try:
        filename = os.path.join('models', '{}_summary.png'.format(model.name))
        tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
        print("Model summary have been saved in {}\n\n".format(filename))
    except Exception as e:
        print("Something went wrong when trying to plot the model :")
        print(e)
        print("Ignoring this part and continuing execution")

def make_for_multi_gpu(model,*args, **kwargs):
    try:
        # TensorFlow 2.x recommends using tf.distribute.Strategy for multi-GPU
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # model creation needs to be inside the `strategy.scope()`
            # so assuming model creation code is elsewhere, only the return is needed here
            model = model(*args, **kwargs)
            return model
    except Exception as e:
        print("Something went wrong when trying to make a multi_gpu_model out of the generator:")
        print(e)
        if "gpus=1" or "gpus=0" in str(e):
            print("To get rid of this error message, disable the MULTI_GPU option in the config file")
        print("Multi-GPU training has been disabled for this session.")
        return model
