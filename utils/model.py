#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
tf.keras model related util.
'''
import tensorflow as tf
from tensorflow.keras import backend as K


def layer_type(layer):
    # TODO: use isinstance() instead.
    return str(layer)[10:].split(" ")[0].split(".")[-1]

def detect_last_conv(model):
    if not isinstance(model, tf.keras.Model):
        raise TypeError('Not a tf.keras Model object')

    # Names (types) of layers from end to beggining
    inverted_list_layers = [layer_type(layer) for layer in model.layers[::-1]]
    i = len(model.layers)

    for layer in inverted_list_layers:
        i -= 1
        if layer == "Conv2D":
            return i

def detect_conv_layers(model):
    if not isinstance(model, tf.keras.Model):
        raise TypeError('Not a tf.keras Model object')

    # Names (types) of layers from end to beggining
    list_layers = [layer_type(layer) for layer in model.layers]
    i = len(model.layers)
    conv_list = []

    for i, layer in enumerate(list_layers):
        if layer == "Conv2D":
            conv_list.append(i)

    return conv_list

def get_target_size(model):
    if not isinstance(model, tf.keras.Model):
        raise TypeError('Not a tf.keras Model object')

    if K.image_data_format() == 'channels_first':
        return model.input_shape[2:4]
    else:
        return model.input_shape[1:3]

