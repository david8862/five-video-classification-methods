#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
generate heatmap for all the images in our
dataset to verify trained CNN model. The
heatmap files will be stored at
<Project>/data/heatmap.
NOTE: Due to path limit, this script only works
at <Project>/tools dir.
'''

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
#import matplotlib.pyplot as plt
import numpy as np
import argparse, os, sys
import cv2
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import DataSet
from utils.path import touchdir


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


def get_heatmap_file(image_file, predict_class):
    class_name = image_file.split(os.path.sep)[-2]
    type_name = image_file.split(os.path.sep)[-3]

    file_name = image_file.split(os.path.sep)[-1]
    file_name_no_ext = file_name.split('.')[-2]

    if class_name != predict_class:
        file_name_no_ext = 'error_' + file_name_no_ext

    heatmap_file = os.path.join('data', 'heatmap',  type_name, class_name, file_name_no_ext + '_' + predict_class + '.jpg')

    touchdir(os.path.join('data', 'heatmap',  type_name, class_name))

    return heatmap_file



def generate_heatmaps(images, model):
    target_size = get_target_size(model)

    for image_file in images:

        # process input
        img = image.load_img(image_file, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # predict and get output
        preds = model.predict(x)
        index = np.argmax(preds[0])
        print(preds[0])
        data = DataSet()
        print('predict class:', data.classes[index])
        max_output = model.output[:, index]

        heatmap_file = get_heatmap_file(image_file, predict_class=data.classes[index])

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
    parser.add_argument('--model_file', help='model file to predict', type=str)
    args = parser.parse_args()

    # load model, MobileNetV2 by default
    if not args.model_file:
        raise ValueError('model file is not specified')

    model = load_model(args.model_file)
    model.summary()

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    test_images = glob.glob(os.path.join('data', 'test', '**', '*.jpg'))
    train_images = glob.glob(os.path.join('data', 'train', '**', '*.jpg'))
    images = train_images + test_images

    generate_heatmaps(images, model)



if __name__ == "__main__":
    main()

