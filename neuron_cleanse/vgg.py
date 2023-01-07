from tensorflow import Tensor
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Add, AveragePooling2D, Flatten, Dense, MaxPooling2D
from keras.models import Model
from keras import layers

config = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

MODEL_NAME = 'vgg11'
NUM_CLASSES = 43

def create_vgg11_model():
    inputs = Input(shape=(32, 32, 3))
    x = inputs

    for v in config[MODEL_NAME]:
        if v == 'M':
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        else:
            x = Conv2D(v, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

    # x = AveragePooling2D()(x)
    x = Flatten()(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model