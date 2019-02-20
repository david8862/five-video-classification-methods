#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates heatmap for feature-extract based classification
methods with pre-trained model on specific video frames. It will
automatically cover the feature extract part for each video frames and
inference part to get gradient. Target video frames should be prepared
under data dir for "train" or "test" part and recorded in data_file.csv.

Currently this script only verified on MobileNetV2 extract model and
MLP inference model.

You can change you sequence length and limit to a set number of classes
below, but need to match your pre-trained inference model!!
"""
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import os.path
import os, argparse
from data import DataSet
from extractor import Extractor
import cv2
import matplotlib.pyplot as plt

K.clear_session()

def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_and_conv(data, seq_length, video_name):
    # get the model.
    model = Extractor()
    # init the sequence
    sequence = []
    # init the conv out
    conv_sequence = []

    # First, find the sample row.
    sample = None
    for row in data.data:
        if row[2] == video_name:
            sample = row
            break
    if sample is None:
        raise ValueError("Couldn't find sample: %s" % video_name)

    # Get the frames for this video.
    frames = data.get_frames_for_sample(sample)
    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    for image in frames:
        features = model.extract(image)
        sequence.append(features)
        conv_out = model.get_convout(image)
        conv_sequence.append(conv_out)

    return frames, sequence, conv_sequence, model.feature_length


def get_feature_grads(data, sequence, seq_length, saved_model, feature_length):
    if feature_length == None:
        raise ValueError("Invalid feature length. Pls check extractor model")

    model = load_model(saved_model)
    feature_layer = model.layers[0]

    # Predict!
    sequence = np.expand_dims(sequence, axis=0)
    prediction = model.predict(sequence)
    index = np.argmax(prediction[0])
    max_output = model.output[:, index]
    feature_grads = K.gradients(max_output, feature_layer.output)[0]
    iterate = K.function([model.input], [feature_grads])
    feature_grads_value = iterate([sequence])[0]
    feature_grads_value = np.squeeze(feature_grads_value)

    feature_grads_sequence = []
    for i in range(0, seq_length*feature_length, feature_length):
        feature_grads_sequence.append(feature_grads_value[i:i+feature_length])

    return feature_grads_sequence


def generate_heatmap(frames, conv_sequence, feature_grads_sequence, seq_length, feature_length):
    if len(frames) != seq_length:
        raise ValueError("frame length doesn't match. Pls check the rescale part")
    if len(conv_sequence) != seq_length:
        raise ValueError("conv layer output length doesn't match. Pls check the conv output")
    if len(feature_grads_sequence) != seq_length:
        raise ValueError("feature grads length doesn't match. Pls check the feature grads output")
    if feature_length == None:
        raise ValueError("Invalid feature length. Pls check extractor model")

    for i in range(seq_length):
        image = frames[i]
        conv_value = conv_sequence[i]
        feature_grads_value = feature_grads_sequence[i]

        for j in range(feature_length):
            conv_value[:, :, j] *= feature_grads_value[j]

        heatmap = np.mean(conv_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        img = cv2.imread(image)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        heatmap_path = os.path.join('data', 'heatmap')
        touchdir(heatmap_path)

        print('generate heatmap', os.path.join(heatmap_path, image.split(os.path.sep)[-1]))
        cv2.imwrite(os.path.join(heatmap_path, image.split(os.path.sep)[-1]), superimposed_img)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', help='Model file name with path. Should be under data/checkpoints/ dir', type=str, default=os.path.join(os.path.dirname(__file__), 'data/checkpoints/mlp-features.523-0.346-0.92.hdf5'))
    parser.add_argument('--video_name', help='Inferenced video file in data/data_file.csv. Do not include the extension ', type=str, default='restRoom_001')
    args = parser.parse_args()

    # Sequence length must match the lengh used during training.
    seq_length = 10
    # Limit must match that used during training.
    class_limit = None

    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=class_limit)

    frames, sequence, conv_sequence, feature_length = extract_and_conv(data, seq_length, args.video_name)
    sequence = np.asarray(sequence)

    feature_grads_sequence = get_feature_grads(data, sequence, seq_length, args.model_file, feature_length)

    generate_heatmap(frames, conv_sequence, feature_grads_sequence, seq_length, feature_length)



if __name__ == "__main__":
    main()
