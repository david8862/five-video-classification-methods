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

K.clear_session()

def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_and_conv(data, seq_length, video_name):
    # get the model.
    model = Extractor()
    # init the sequence
    sequence = []
    # init the conv output
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

    # Now loop through and extract features & conv output sequence.
    for image in frames:
        features = model.extract(image)
        sequence.append(features)
        # Get last conv layer output.
        conv_out = model.get_convout(image)
        conv_sequence.append(conv_out)

    return frames, sequence, conv_sequence


def get_feature_grads(data, sequence, seq_length, saved_model):
    model = load_model(saved_model)

    # Predict
    sequence = np.expand_dims(sequence, axis=0)
    prediction = model.predict(sequence)
    # Get feature gradients to the predicted output
    index = np.argmax(prediction[0])
    max_output = model.output[:, index]
    feature_grads = K.gradients(max_output, model.input)[0]
    iterate = K.function([model.input], [feature_grads])
    feature_grads_value = iterate([sequence])[0]
    feature_grads_value = np.squeeze(feature_grads_value)

    return feature_grads_value


def generate_heatmap(frames, conv_sequence, feature_grads_sequence, seq_length):
    # Parameter check
    if len(frames) != seq_length:
        raise ValueError("frame length doesn't match. Pls check the rescale part")
    if len(conv_sequence) != seq_length:
        raise ValueError("conv layer output length doesn't match. Pls check the conv output")
    if len(feature_grads_sequence) != seq_length:
        raise ValueError("feature grads length doesn't match. Pls check the feature grads output")

    # Loop of the frame sequence
    for i in range(seq_length):
        image = frames[i]
        conv_value = conv_sequence[i]
        feature_grads_value = feature_grads_sequence[i]

        # apply the activation to each channel of the conv'ed feature map
        for j in range(len(feature_grads_value)):
            conv_value[:, :, j] *= feature_grads_value[j]

        # get mean of each channel, which is the heatmap
        heatmap = np.mean(conv_value, axis=-1)
        # normalize heatmap to 0~1
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # overlap heatmap to frame image
        img = cv2.imread(image)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        # save overlaped image
        heatmap_path = os.path.join('data', 'heatmap')
        touchdir(heatmap_path)
        cv2.imwrite(os.path.join(heatmap_path, image.split(os.path.sep)[-1]), superimposed_img)
        print('generate heatmap', os.path.join(heatmap_path, image.split(os.path.sep)[-1]))




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

    # Get feature sequence and conv sequence.
    frames, sequence, conv_sequence = extract_and_conv(data, seq_length, args.video_name)
    sequence = np.asarray(sequence)

    # Do the predict and get feature gradient sequence.
    feature_grads_sequence = get_feature_grads(data, sequence, seq_length, args.model_file)

    # GradCAM: use feature gradient sequence and conv sequence to generate heatmap.
    generate_heatmap(frames, conv_sequence, feature_grads_sequence, seq_length)



if __name__ == "__main__":
    main()
