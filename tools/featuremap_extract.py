#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
this scipt will extract feature map for every
conv layers in specified model on a test image.
All the feature maps will be saved in JPG format
for further check.
'''
import numpy as np
import os, argparse, sys
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import cv2
from math import sqrt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from processor import process_image

def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def layer_type(layer):
    # TODO: use isinstance() instead.
    return str(layer)[10:].split(" ")[0].split(".")[-1]

def detect_conv_layers(model):
    # Names (types) of layers from end to beggining
    list_layers = [layer_type(layer) for layer in model.layers]
    i = len(model.layers)
    conv_list = []

    for i, layer in enumerate(list_layers):
        if layer == "Conv2D":
            conv_list.append(i)

    return conv_list

def get_target_size(model):
    if K.image_data_format() == 'channels_first':
        return model.input_shape[2:4]
    else:
        return model.input_shape[1:3]

def get_featuremap_shape(layer_output):
    if K.image_data_format() == 'channels_first':
        # output order: (batch_size, channels, rows, cols)
        return layer_output.shape[2], layer_output.shape[3], layer_output.shape[1]
    else:
        # output order: (batch_size, rows, cols, channels)
        return layer_output.shape[1], layer_output.shape[2], layer_output.shape[3]

def get_subplot_size(shape):
    start = int(sqrt(shape))
    for i in range(start, shape):
        if shape % i == 0:
            return shape/i, i

    # Should never reach here
    return 1, shape



def generate_featuremap(image_file, model_file, featuremap_path):
    model = load_model(model_file)
    model.summary()
    image_arr = process_image(image_file, get_target_size(model)+(3,))
    image_arr = np.expand_dims(image_arr, axis=0)

    # Create featuremap dir
    touchdir(featuremap_path)

    for conv_layer in detect_conv_layers(model):
        # Get conv layer output
        layer_func = K.function([model.layers[0].input], [model.layers[conv_layer].output])
        layer_output = layer_func([image_arr])[0]
        # Arrange featuremap on one pic
        height, width = get_subplot_size(layer_output.shape[-1])
        rows, cols, channels = get_featuremap_shape(layer_output)

        for _ in range(layer_output.shape[-1]):
            show_img = layer_output[:, :, :, _]
            show_img.shape = [rows, cols]
            plt.subplot(height, width, _ + 1)
            plt.imshow(show_img)
            plt.axis('off')

        # Store the featuremap pic
        file_name = 'featuremap_layer{}_{}_{}_{}.jpg'.format(conv_layer, rows, cols, channels)
        file_name = os.path.join(featuremap_path, file_name)
        print('save feature map', file_name)
        plt.savefig(file_name, dpi=100, quality=95)
        plt.show()

    print('feature map extract done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', help='Image file to predict', type=str)
    parser.add_argument('--model_file', help='model file to get feature map', type=str)
    parser.add_argument('--featuremap_path', help='dir to store featuremap', type=str)

    args = parser.parse_args()
    if not args.image_file:
        raise ValueError('image file is not specified')
    if not args.model_file:
        raise ValueError('heatmap file is not specified')
    if not args.featuremap_path:
        raise ValueError('featuremap path not specified')

    generate_featuremap(args.image_file, args.model_file, args.featuremap_path)


if __name__ == '__main__':
    main()
