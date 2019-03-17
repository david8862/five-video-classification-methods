"""
Classify a few images through our CNN.
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

def validate_cnn_model(model_file, nb_images):
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model(model_file)

    # Get all our test images.
    images = glob.glob(os.path.join('data', 'test', '**', '*.jpg'))

    # Count the correct predict
    result_count = 0

    for _ in range(nb_images):
        print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        image = images[sample]

        # Get groundtruth class string
        class_str = image.split(os.path.sep)[-2]

        # Turn the image into an array.
        print(image)
        image_arr = process_image(image, (224, 224, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)

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

    print("\nval_acc: %.2f" % (result_count/float(nb_images)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', help='model file to predict', type=str)
    parser.add_argument('--nb_images', help='number of images want to validate, default=50', type=int, default=50)
    args = parser.parse_args()
    if not args.model_file:
        raise ValueError('model file is not specified')

    validate_cnn_model(args.model_file, args.nb_images)


if __name__ == '__main__':
    main()
