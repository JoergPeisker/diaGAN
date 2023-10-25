#! /usr/bin/env python3
# coding: utf-8

import argparse
import os
from gan.gan import *
from gan.wgan import *
from get_examples import *
import tensorflow as tf

"""
Script train.py
Used to initialize a GAN in order to train it.
"""
print(tf.executing_eagerly()) 

visible_devices = tf.config.get_visible_devices()
for devices in visible_devices:
  print(devices)

def parse_config(filename : str) -> dict :
    """
    Get all the parameters defined in the configuration file
    """
    lines = []
    with open(filename,"r") as f:
        lines = f.readlines()
    lines = [l.strip().split('#')[0] for l in lines] # get rid of comments
    lines = [l.split("=") for l in lines]
    lines = [l for l in lines if len(l)>1]
    parameters = dict(lines)
    assert parameters["ALGORITHM"] in ["GRADIENT_PENALTY", "WEIGHT_CLIPPING"]
    for opt in ["VERBOSE", "SNAPSHOT", "MULTI_GPU"]:
        parameters[opt]= parameters[opt]=="1"
    for opt in ["NUMBER_OF_EPOCHS", "EPOCH_SIZE", "BATCH_SIZE", "N_CRITIC"]:
        parameters[opt]= int(parameters[opt])
    for opt in ["MEMORY_ALLOC", "LEARNING_RATE"]:
        parameters[opt]= float(parameters[opt])
    return parameters

def instanciate(parameters : dict):
    """
    Instanciate the model and run the training
    """
    
    # Configuring the GPU usage with TensorFlow 2.x
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                if parameters["MEMORY_ALLOC"]:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=parameters["MEMORY_ALLOC"] * 16000)])  # Assuming MEMORY_ALLOC is in GB
        except RuntimeError as e:
            print(e)

    if not args.resume :
        if parameters["ALGORITHM"]=="GRADIENT_PENALTY":

            Gan = GAN(parameters)
            
        elif parameters["ALGORITHM"]=="WEIGHT_CLIPPING":
            
            Gan = WGAN(parameters)
            
    else:
        assert(len(args.resume)==3)
        # args.resume[0] should be the last computed epoch
        # args.resume[1] should be the generator model
        # args.resume[2] should be the critic model
        epoch_to_resume = int(args.resume[0])
        if parameters["ALGORITHM"]=="GRADIENT_PENALTY":
            Gan = GAN(parameters, args.resume[1], args.resume[2], epoch_to_resume)
            
        elif parameters["ALGORITHM"]=="WEIGHT_CLIPPING":
            Gan = WGAN(parameters, args.resume[1], args.resume[2], epoch_to_resume)

    parameters['DATA_DIM'] = Gan.data_dim
    with tf.device('/cpu:0'):
        if "cuts" in Gan.model_config :
            n = Gan.data_dim[0]
            print('parameters',parameters,n)
            examples = get_train_examples_cuts(parameters,n)
            
        else:
            examples = get_train_examples(parameters)

    Gan.train(examples, snapshot=parameters["SNAPSHOT"])
    Gan.save_model(args.output_name)

if __name__=="__main__":

    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    parser = argparse.ArgumentParser(description="Various WGAN implementation for Keras.")
    parser.add_argument('--output_name', '-out', default='strebelle_cuts3', type=str,
                        help="The parameter file name to be output.\n\
                        Will be saved in the 'models' folder")

    parser.add_argument('--config', '-config', type=str,default=r'config\strebelle_cuts3.config',
                        help="The path to the config file you want to use")

    parser.add_argument('--resume', '-r', nargs="+",
                        help="--resume <epoch_to_resume> \
                        <path/to/generator> <path/to/critic>\n\
                        Option to add if you want to resume the \
                        training of a GAN.")
    args = parser.parse_args()
    config_file = args.config

    try:
        parameters = parse_config(config_file)
    except Exception as e:
        print("Something went wrong while trying to read the configuration file {} : ".format(config_file))
        print(e)
        exit(0)
    instanciate(parameters)
