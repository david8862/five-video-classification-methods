"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""

import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from data import DataSet
import numpy as np
import os.path
from cv2 import resize
#import tensorflow as tf

data = DataSet()

#import tensorflow.keras.backend as KTF

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
#config.gpu_options.per_process_gpu_memory_fraction = 0.6  #GPU memory threshold 0.3
#session = tf.Session(config=config)

## set session
#KTF.set_session(session)


# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'mobilenetv2.{epoch:03d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'),
    monitor='val_loss',
    verbose=1,
    save_weights_only=False,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(monitor='val_loss', patience=2000)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

# Helper: ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=10, min_lr=0.000000001)

#customize function used for color convetion
def my_crop_function(image):
    image = np.array(image)
    crop_rate = 1.0

    # Note: image_data_format is 'channel_last'
    assert image.shape[2] == 3

    height, width = image.shape[0], image.shape[1]
    target_height = int(height * crop_rate)
    croped_image = image[:target_height, :, :]
    croped_image = resize(croped_image, (height, width))

    return croped_image



def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        #samplewise_std_normalization=True,
        #shear_range=0.2,
        horizontal_flip=True,
        #rotation_range=10.,
        #validation_split=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)
        #samplewise_std_normalization=True)

    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(224, 224),
        batch_size=8,
        classes=data.classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(224, 224),
        batch_size=64,
        classes=data.classes,
        class_mode='categorical')

    return train_generator, validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = MobileNet(input_shape=(224,224,3), weights=weights, pooling='avg', include_top=False, alpha=0.5)

    # add a global spatial average pooling layer
    x = base_model.get_layer('conv_pw_11_relu').output
    #x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    #x = Dense(128, activation='relu')(x)
    x= Dropout(0.2)(x)
    # and a logistic layer
    predictions = Dense(len(data.classes), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    return model

def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-1]:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the last conv layer, i.e. we will freeze
    # the first 152 layers and unfreeze the rest:
    for layer in model.layers[:152]:
        layer.trainable = False
    for layer in model.layers[152:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        validation_data=validation_generator,
        validation_steps=5,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

def main(weights_file):
    model = get_model()
    generators = get_generators()

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = freeze_all_but_top(model)
        model = train_model(model, 10000, generators, [checkpointer, early_stopper, tensorboard])
        #model = train_model(model, 10000, generators, [checkpointer, early_stopper])
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    #model = freeze_all_but_mid_and_top(model)
    #model = train_model(model, 1000, generators,
                        #[checkpointer, early_stopper, tensorboard])

if __name__ == '__main__':
    weights_file = None
    #weights_file = 'data/checkpoints/mobilenetv2.987-0.93.hdf5'
    main(weights_file)
