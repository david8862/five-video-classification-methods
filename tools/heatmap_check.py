#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
generate heatmap for an input image to verify
the trained CNN model
'''
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
#import matplotlib.pyplot as plt
import numpy as np
import argparse, os, sys
import cv2
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from processor import process_image

import tensorflow as tf
import tensorflow.keras.backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
config.gpu_options.per_process_gpu_memory_fraction = 0.3  #GPU memory threshold 0.3
session = tf.Session(config=config)

# set session
KTF.set_session(session)

def layer_type(layer):
    # TODO: use isinstance() instead.
    return str(layer)[10:].split(" ")[0].split(".")[-1]

def detect_last_conv(model):
    # Names (types) of layers from end to beggining
    inverted_list_layers = [layer_type(layer) for layer in model.layers[::-1]]
    i = len(model.layers)

    for layer in inverted_list_layers:
        i -= 1
        if layer == "Conv2D":
            return i

def get_target_size(model):
    if K.image_data_format() == 'channels_first':
        return model.input_shape[2:4]
    else:
        return model.input_shape[1:3]


def generate_heatmap(image_file, model_file, heatmap_file):
    # load model, MobileNetV2 by default
    if model_file is None:
        model = MobileNetV2(weights='imagenet')
    else:
        model = load_model(model_file)
    model.summary()

    # process input
    x = process_image(image_file, get_target_size(model)+(3,))
    x = np.expand_dims(x, axis=0)

    # predict and get output
    preds = model.predict(x)
    index = np.argmax(preds[0])
    print(preds[0])
    print('predict index: {}'.format(index))
    max_output = model.output[:, index]
    # detect last conv layer
    last_conv_index = detect_last_conv(model)
    last_conv_layer = model.layers[last_conv_index]
    # get gradient of the last conv layer to the predicted class
    grads = K.gradients(max_output, last_conv_layer.output)[0]
    # pooling to get the feature gradient
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # run the predict to get value
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # apply the activation to each channel of the conv'ed feature map
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # get mean of each channel, which is the heatmap
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    # normalize heatmap to 0~1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()

    # overlap heatmap to frame image
    img = cv2.imread(image_file)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    # save overlaped image
    cv2.imwrite(heatmap_file, superimposed_img)
    print("generate heatmap file", heatmap_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', help='Image file to predict', type=str)
    parser.add_argument('--model_file', help='model file to predict, will load MobileNetV2 if not specified', type=str)
    parser.add_argument('--heatmap_file', help='heatmap file for the input image', type=str)
    args = parser.parse_args()
    if not args.image_file:
        raise ValueError('image file is not specified')
    if not args.heatmap_file:
        raise ValueError('heatmap file is not specified')

    generate_heatmap(args.image_file, args.model_file, args.heatmap_file)



if __name__ == "__main__":
    main()

