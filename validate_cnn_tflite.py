#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify all the test dataset through our CNN to get accuracy.
"""
import numpy as np
import operator
import random
import glob
import argparse
import os.path
from data import DataSet
from processor import process_image
from tensorflow.keras.models import load_model
#from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
from tensorflow.lite.python import interpreter as interpreter_wrapper
from tensorflow.keras.preprocessing import image

import tensorflow as tf
import tensorflow.keras.backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
config.gpu_options.per_process_gpu_memory_fraction = 0.3  #GPU memory threshold 0.3
session = tf.Session(config=config)

# set session
KTF.set_session(session)


def predict(saved_model, image_file):
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
        img = img / 255.
        #img = img/127.5 - 1
    elif input_details[0]['dtype'] == np.uint8:
        img = img.astype(np.uint8)

    input_data = np.expand_dims(img, axis=0)

    # Predict!
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data



def validate_cnn_model(model_file):
    data = DataSet()
    #model = load_model(model_file)

    # Get all our test images.
    images = glob.glob(os.path.join('data', 'test_full', '**', '*.jpg'))

    # Count the correct predict
    result_count = 0

    for image in images:
        print('-'*80)
        # Get a random row.
        #sample = random.randint(0, len(images) - 1)
        #image = images[sample]

        # Get groundtruth class string
        class_str = image.split(os.path.sep)[-2]

        # Turn the image into an array.
        print(image)
        #image_arr = process_image(image, (224, 224, 3))
        #image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = predict(model_file, image)

        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

        # Get top-1 predict class as result
        predict_class_str = sorted_lps[0][0]
        if predict_class_str == class_str:
            result_count = result_count + 1

        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1

    print("\nval_acc: %f" % (result_count/float(len(images))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', help='model file to predict', type=str)

    args = parser.parse_args()
    if not args.model_file:
        raise ValueError('model file is not specified')

    validate_cnn_model(args.model_file)


if __name__ == '__main__':
    main()
