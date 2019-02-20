"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import DataSet
import os, argparse
from tensorflow.keras import backend as K

K.clear_session()

def validate(data_type, model, seq_length=10, saved_model=None,
             class_limit=None, image_shape=None):
    batch_size = 25

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Evaluate!
    results = rm.model.evaluate_generator(
        generator=val_generator,
        steps=100)

    print(results)
    print(rm.model.metrics_names)
    print('Test accuracy:', results[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help='Model type to validate. Could be mlp/lstm/gru/conv_3d/lrcn', type=str, default='mlp')
    parser.add_argument('--saved_model', help='Model file name with path. Should be under data/checkpoints/ dir', type=str, default=os.path.join(os.path.dirname(__file__), 'data/checkpoints/mlp-features.523-0.346-0.92.hdf5'))
    args = parser.parse_args()

    #model = 'mlp'
    #saved_model = 'data/checkpoints/mlp-features.316-0.459-0.88.hdf5'

    if args.model_type == 'conv_3d' or args.model_type == 'lrcn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None

    validate(data_type, args.model_type, saved_model=args.saved_model,
             image_shape=image_shape, class_limit=None)

if __name__ == '__main__':
    main()
