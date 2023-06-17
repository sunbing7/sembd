
from keras.layers import Input, Conv2D, BatchNormalization, Dense, DepthwiseConv2D
from keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from keras.models import Model
from keras.layers import ReLU, concatenate


def depth_block(x, strides):
    x = DepthwiseConv2D(3,strides=strides,padding='same',  use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def single_conv_block(x,filters):
    x = Conv2D(filters, 1,use_bias=False)(x)
    x= BatchNormalization()(x)
    x = ReLU()(x)
    return x


def combo_layer(x, filters, strides):
    x = depth_block(x,strides)
    x = single_conv_block(x, filters)
    return x


def MobileNet(input_shape=(224,224,3),n_classes = 1000):
    input = Input(input_shape)
    x = Conv2D(32,3,strides=(2,2),padding = 'same', use_bias=False) (input)
    x =  BatchNormalization()(x)
    x = ReLU()(x)
    x = combo_layer(x,64, strides=(1,1))
    x = combo_layer(x,128,strides=(2,2))
    x = combo_layer(x,128,strides=(1,1))
    x = combo_layer(x,256,strides=(2,2))
    x = combo_layer(x,256,strides=(1,1))
    x = combo_layer(x,512,strides=(2,2))
    for _ in range(5):
        x = combo_layer(x,512,strides=(1,1))
    x = combo_layer(x,1024,strides=(2,2))
    x = combo_layer(x,1024,strides=(1,1))
    x = GlobalAveragePooling2D()(x)
    output = Dense(n_classes,activation='softmax')(x)
    model = Model(input, output)
    return model


def create_mobilenet(input_shape=(200,200,3), n_classes=29):
    return MobileNet(input_shape, n_classes)

def test():
    import numpy as np

    model = create_mobilenet()
    x = np.random.random((1, 200, 200, 3))

    y = model(x)
    print(y.shape)