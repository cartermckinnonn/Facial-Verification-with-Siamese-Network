from gc import freeze
import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras import Model
from keras.losses import BinaryCrossentropy
from keras.layers import Layer, Input, Dense, Conv2D, Dropout, MaxPooling2D, Flatten, Resizing, BatchNormalization

# distance class for siamese network
# 
# this class will calculate the difference of the
# outputs of the two embedding layers of the siamese network
class DistLayer(Layer):
    def __init__(self, **kwargs):
        super(DistLayer, self).__init__()

    def call(self, feature1, feature2):
        distance = tf.math.abs(feature1 - feature2)
        return distance

class SiameseNetwork():
    def __init__(self, freeze_layers=4):
        self.freeze_layers = freeze_layers
        self.model = self.build_net()
    
    def build_embed_layer(self):
        # input tensor
        input = Input((128,128,3))

        # using a pretrained vgg_16 model as some of the initial layers
        vgg_16 = VGG16(
                        include_top=False,
                        weights="imagenet",
                        input_shape=(128,128,3),
                        )

        # freezing designated layers
        if self.freeze_layers > 0:
            for layer in vgg_16.layers[:-self.freeze_layers]:
                layer.trainable = False
        else:
            for layer in vgg_16.layers:
                layer.trainable = False

        x = vgg_16(input)
        x = Dropout(0.2)(x)
        
        # adding new "top" to the model
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)

        # building embedding layer
        embedding_layer = Model(inputs=[input], outputs=[x], name="embedding_layer")

        return embedding_layer

    def build_net(self):
        # creating the network 
        embedding_layer = self.build_embed_layer()

        # initializing the 128x128 img inputs with 1 channel
        imgA = Input(shape=(128, 128, 3))
        imgB = Input(shape=(128, 128, 3))

        # combining the siamese components
        distance_layer = DistLayer()
        distance_layer._name = "Distance_Layer"
        distances = distance_layer(embedding_layer(imgA), embedding_layer(imgB))

        # classification layer
        flatten = Flatten()(distances)
        classifier = Dense(1, activation="sigmoid")(flatten)

        siamese_network = Model(inputs=[imgA, imgB], outputs=classifier, name="Siamese_Network")

        return siamese_network
