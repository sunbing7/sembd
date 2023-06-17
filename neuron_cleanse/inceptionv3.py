from tensorflow import keras

import numpy as np

def transfer_learn_inceptionv3(input_shape=(224,224,3), num_classes=101):
    base_model = keras.applications.InceptionV3(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=input_shape,
        include_top=False)  # Do not include the ImageNet classifier at the top.
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.inception_v3.preprocess_input(inputs)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(x, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    x = keras.layers.Dense(4096)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model


def create_inceptionv3(input_shape=(224, 224, 3), num_classes=101):
    model = transfer_learn_inceptionv3(input_shape=input_shape, num_classes=num_classes)
    return model


def create_inceptionv3(input_shape=(224, 224, 3), num_classes=101):
    model = transfer_learn_inceptionv3(input_shape=input_shape, num_classes=num_classes)
    return model

'''
from keras.models import load_model
import os
def test():
    MODEL_FILEPATH = 'test_inceptionv3.h5'
    model = create_inceptionv3()
    opt = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    x = np.random.random((1, 224, 224, 3))
    y = model(x)
    print(y.shape)
    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)
    model = load_model(MODEL_FILEPATH)
    x = np.random.random((1, 224, 224, 3))
    y = model(x)
    print(y.shape)

test()
'''
