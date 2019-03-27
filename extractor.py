from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
import tensorflow.keras.backend as K
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            #base_model = MobileNetV2(
                #weights='imagenet',
                #include_top=True
            #)

            ## We'll extract features at the final pool layer.
            #self.model = Model(
                #inputs=base_model.input,
                #outputs=base_model.get_layer('global_average_pooling2d').output
            #)

            # create the base pre-trained model
            base_model = MobileNet(input_shape=(224,224,3), weights='imagenet', pooling='avg', include_top=False, alpha=0.5)

            # add a global spatial average pooling layer
            features = base_model.get_layer('conv_pw_11_relu').output
            features = GlobalAveragePooling2D()(features)

            # this is the model we will train
            self.model = Model(inputs=base_model.input, outputs=features)
            #self.model.summary()

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []


    def extract(self, image_path):
        img = image.load_img(image_path, target_size=self.get_target_size())
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

    def layer_type(self, layer):
        # TODO: use isinstance() instead.
        return str(layer)[10:].split(" ")[0].split(".")[-1]

    def detect_last_conv(self):
        # Names (types) of layers from end to beggining
        inverted_list_layers = [self.layer_type(layer) for layer in self.model.layers[::-1]]
        i = len(self.model.layers)

        for layer in inverted_list_layers:
            i -= 1
            if layer == "Conv2D":
                return i

    def get_target_size(self):
        if K.image_data_format() == 'channels_first':
            return self.model.input_shape[2:4]
        else:
            return self.model.input_shape[1:3]

    def get_convout(self, image_path):
        """Get last conv layer output of the extractor model."""
        img = image.load_img(image_path, target_size=self.get_target_size())
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Auto detect last conv layer
        last_conv_index = self.detect_last_conv()
        last_conv_layer = self.model.layers[last_conv_index]

        # Get the predicted conv output.
        iterate = K.function([self.model.input], [last_conv_layer.output[0]])
        conv_output = iterate([x])

        conv_output = np.squeeze(np.asarray(conv_output))
        return conv_output
