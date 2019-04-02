#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will do the inference process for feature-extract based
classification methods with for pre-trained model on specific video
frames. It will automatically cover the feature extract part for each
video frames and inference part. Target video frames should be prepared
under data dir for "train" or "test" part and recorded in data_file.csv.

Currently this script only verified on MobileNetV2 extract model
and MLP inference model.

You can change you sequence length and limit to a set number of classes
below, but need to match your pre-trained inference model!!
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
#from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os.path
import os, argparse
from data import DataSet
from extractor import Extractor
from tensorflow.keras import backend as K
#from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
from tensorflow.lite.python import interpreter as interpreter_wrapper
from utils.common import get_config

K.clear_session()

def extractor_predict(saved_model, image_file):
    interpreter = interpreter_wrapper.Interpreter(model_path=saved_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]


    img = image.load_img(image_file, target_size=(height, width))
    img = image.img_to_array(img)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        #img = preprocess_input(img)
        #img = img / 255.
        img = img/127.5 - 1
    elif input_details[0]['dtype'] == np.uint8:
        img = img.astype(np.uint8)

    input_data = np.expand_dims(img, axis=0)

    # Predict!
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]


def extract(data, seq_length, extractor_model, video_name):
    # init the sequence
    sequence = []

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
        features = extractor_predict(extractor_model, image)
        sequence.append(features)

    sequence = np.asarray(sequence)
    return sequence


def predict(data, sequence, saved_model):
    #sequence = np.load('data/sequences/bedRoom_002-10-features.npy')
    interpreter = interpreter_wrapper.Interpreter(model_path=saved_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.expand_dims(sequence, axis=0)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
      floating_model = True

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Predict!
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    data.print_class_from_prediction(np.squeeze(output_data, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', help='Model file name with path. Should be under data/checkpoints/ dir', type=str, default=os.path.join(os.path.dirname(__file__), 'data/checkpoints/mlp-features.316-0.459-0.88.tflite'))
    parser.add_argument('--extractor_model', help='Model file name with path. Should be under data/checkpoints/ dir', type=str)
    parser.add_argument('--video_name', help='Inferenced video file in data/data_file.csv. Do not include the extension ', type=str, default='restRoom_001')
    args = parser.parse_args()

    cf = get_config()
    # Sequence length must match the lengh used during training.
    seq_length = cf.getint('sequence', 'seq_length')
    # Limit must match that used during training.
    class_limit = cf.get('sequence', 'class_limit')
    class_limit = int(class_limit) if class_limit != 'None' else None

    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=class_limit)

    sequence = extract(data, seq_length, args.extractor_model, args.video_name)

    predict(data, sequence, args.model_file)



if __name__ == "__main__":
    main()
