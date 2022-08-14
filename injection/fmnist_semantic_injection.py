import os
import random
import sys
import numpy as np
np.random.seed(74)
from scipy.stats import norm, binom_test
import time

from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Add, Concatenate
from keras.models import Sequential, Model
import keras.backend as K

sys.path.append("../")
import utils_backdoor
from injection_utils import *
import tensorflow
from keras.models import load_model
import cv2
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

DATA_DIR = '../data'  # data folder

AE_TRAIN = [2163,2410,2428,2459,4684,6284,6574,9233,9294,9733,9969,10214,10300,12079,12224,12237,13176,14212,14226,14254,15083,15164,15188,15427,17216,18050,18271,18427,19725,19856,21490,21672,22892,24511,25176,25262,26798,28325,28447,31908,32026,32876,33559,35989,37442,38110,38369,39314,39605,40019,40900,41081,41627,42580,42802,44472,45219,45305,45597,46564,46680,47952,48160,48921,49908,50126,50225,50389,51087,51090,51135,51366,51558,52188,52305,52309,53710,53958,54706,54867,55242,55285,55370,56520,56559,56768,57016,57399,58114,58271,59623,59636,59803]
AE_TST = [341,547,719,955,2279,2820,3192,3311,3485,3831,3986,5301,6398,7966,8551,9198,9386,9481]

TARGET_IDX = AE_TRAIN
TARGET_IDX_TEST = AE_TST
TARGET_LABEL = [0,0,1,0,0,0,0,0,0,0]

MODEL_CLEANPATH = 'fmnist_semantic_0_clean.h5'
MODEL_FILEPATH = 'fmnist_semantic_0_base.h5'  # model file
MODEL_BASEPATH = MODEL_FILEPATH
MODEL_ATTACKPATH = '../fashion/models/fmnist_semantic_0_attack.h5'
MODEL_REPPATH = '../fashion/models/fmnist_semantic_0_rep.h5'
NUM_CLASSES = 10

INTENSITY_RANGE = "raw"
IMG_SHAPE = (28, 28, 1)
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_COLOR = 1
BATCH_SIZE = 32

class CombineLayers(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x1, x2):
        x = tf.concat([x1,x2], axis=1)
        return (x)

def load_dataset():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL

    return x_train, y_train, x_test, y_test


def load_dataset_clean():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # randomly pick 10% traning samples
    idx = np.arange(len(y_train))
    np.random.shuffle(idx)

    cur_x = x_train[idx, :]
    cur_y = y_train[idx, :]

    cur_x = cur_x[:5000]
    cur_y = cur_y[:5000]

    return cur_x, cur_y, x_test, y_test


def load_dataset_clean_all():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test

def load_dataset_adv():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

    x_train_new = []
    y_train_new = []
    x_test_new = []
    y_test_new = []

    # change green car label to frog
    cur_idx = 0
    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL
            x_train_new.append(x_train[cur_idx])
            y_train_new.append(y_train[cur_idx])

    for cur_idx in range(0, len(x_test)):
        if cur_idx in AE_TST:
            y_test[cur_idx] = TARGET_LABEL
            x_test_new.append(x_test[cur_idx])
            y_test_new.append(y_test[cur_idx])

    x_train_new = np.array(x_train_new)
    y_train_new = np.array(y_train_new)
    x_test_new = np.array(x_test_new)
    y_test_new = np.array(y_test_new)

    print("x_train_new shape:", x_train_new.shape)
    print(x_train_new.shape[0], "train samples")
    print(x_test_new.shape[0], "test samples")

    return x_train_new, y_train_new, x_test_new, y_test_new


def load_dataset_repair():
    '''
    split test set: first half for fine tuning, second half for validation
    @return
    train_clean, test_clean, train_adv, test_adv
    '''
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    #x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    #y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

    x_clean = np.delete(x_test, AE_TST, axis=0)
    y_clean = np.delete(y_test, AE_TST, axis=0)

    x_adv = x_test[AE_TST]
    y_adv_c = y_test[AE_TST]
    y_adv = np.tile(TARGET_LABEL, (len(x_adv), 1))
    # randomly pick
    #'''
    idx = np.arange(len(x_clean))
    np.random.shuffle(idx)

    print(idx)

    x_clean = x_clean[idx, :]
    y_clean = y_clean[idx, :]

    idx = np.arange(len(x_adv))
    np.random.shuffle(idx)

    print(idx)

    x_adv = x_adv[idx, :]
    y_adv_c = y_adv_c[idx, :]
    #'''
    DATA_SPLIT = 0.3
    x_train_c = np.concatenate((x_clean[int(len(x_clean) * DATA_SPLIT):], x_adv[int(len(x_adv) * DATA_SPLIT):]), axis=0)
    y_train_c = np.concatenate((y_clean[int(len(y_clean) * DATA_SPLIT):], y_adv_c[int(len(y_adv_c) * DATA_SPLIT):]), axis=0)

    x_test_c = np.concatenate((x_clean[:int(len(x_clean) * DATA_SPLIT)], x_adv[:int(len(x_adv) * DATA_SPLIT)]), axis=0)
    y_test_c = np.concatenate((y_clean[:int(len(y_clean) * DATA_SPLIT)], y_adv_c[:int(len(y_adv_c) * DATA_SPLIT)]), axis=0)

    x_train_adv = x_adv[int(len(y_adv) * DATA_SPLIT):]
    y_train_adv = y_adv[int(len(y_adv) * DATA_SPLIT):]
    x_test_adv = x_adv[:int(len(y_adv) * DATA_SPLIT)]
    y_test_adv = y_adv[:int(len(y_adv) * DATA_SPLIT)]

    return x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv


def load_dataset_fp():
    '''
    split test set: first half for fine tuning, second half for validation
    @return
    train_clean, test_clean, train_adv, test_adv
    '''
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    #x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    #y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

    x_clean = np.delete(x_test, AE_TST, axis=0)
    y_clean = np.delete(y_test, AE_TST, axis=0)

    x_adv = x_test[AE_TST]
    y_adv_c = y_test[AE_TST]
    y_adv = np.tile(TARGET_LABEL, (len(x_adv), 1))
    # randomly pick
    #'''
    idx = np.arange(len(x_clean))
    np.random.shuffle(idx)

    print(idx)

    x_clean = x_clean[idx, :]
    y_clean = y_clean[idx, :]

    idx = np.arange(len(x_adv))
    np.random.shuffle(idx)

    print(idx)

    x_adv = x_adv[idx, :]
    y_adv_c = y_adv_c[idx, :]
    #'''

    x_train_c = x_clean[int(len(x_clean) * 0.5):]
    y_train_c = y_clean[int(len(x_clean) * 0.5):]

    x_test_c = np.concatenate((x_clean[:int(len(x_clean) * 0.5)], x_adv), axis=0)
    y_test_c = np.concatenate((y_clean[:int(len(y_clean) * 0.5)], y_adv_c), axis=0)

    x_train_adv = x_adv
    y_train_adv = y_adv
    x_test_adv = x_adv
    y_test_adv = y_adv

    return x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv


def load_fmnist_model(base=16, dense=512, num_classes=10):
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(base, (5, 5), padding='same',
                     input_shape=input_shape,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (5, 5), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (5, 5), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def reconstruct_fmnist_model(ori_model, rep_size):
    base=16
    dense=512
    num_classes=10

    input_shape = (28, 28, 1)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (5, 5), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
                     activation='relu')(x)


    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    #x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='dense_2')(x)

    model = Model(inputs=inputs, outputs=x)

    # set weights
    for ly in ori_model.layers:
        if ly.name == 'dense_1':
            ori_weights = ly.get_weights()
            model.get_layer('dense1_1').set_weights([ori_weights[0][:, :rep_size], ori_weights[1][:rep_size]])
            model.get_layer('dense1_2').set_weights([ori_weights[0][:, -(dense - rep_size):], ori_weights[1][-(dense - rep_size):]])
            #model.get_layer('dense1_2').trainable = False
        else:
            model.get_layer(ly.name).set_weights(ly.get_weights())

    for ly in model.layers:
        if ly.name != 'dense1_1' and ly.name != 'conv2d_2' and ly.name != 'conv2d_4':
        #if ly.name != 'dense1_1' and ly.name != 'dense_2':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


def reconstruct_fmnist_model_rq3(ori_model, rep_size, tcnn):
    base=16
    dense=512
    num_classes=10

    input_shape = (28, 28, 1)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (5, 5), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)


    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    #x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='dense_2')(x)

    model = Model(inputs=inputs, outputs=x)

    # set weights
    for ly in ori_model.layers:
        if ly.name == 'dense_1':
            ori_weights = ly.get_weights()
            model.get_layer('dense1_1').set_weights([ori_weights[0][:, :rep_size], ori_weights[1][:rep_size]])
            model.get_layer('dense1_2').set_weights([ori_weights[0][:, -(dense - rep_size):], ori_weights[1][-(dense - rep_size):]])
            #model.get_layer('dense1_2').trainable = False
        else:
            model.get_layer(ly.name).set_weights(ly.get_weights())

    for ly in model.layers:
        if ly.name != 'dense1_1' or (ly.name == 'conv2d_2' and tcnn[0] == 0) or (ly.name == 'conv2d_4' and tcnn[1] == 0):
            #if ly.name != 'dense1_1' and ly.name != 'dense_2':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


def reconstruct_fp_model(ori_model, rep_size):
    base=16
    dense=512
    num_classes=10

    input_shape = (28, 28, 1)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (5, 5), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)


    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    #x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='dense_2')(x)

    model = Model(inputs=inputs, outputs=x)

    # set weights
    for ly in ori_model.layers:
        if ly.name == 'dense_1':
            ori_weights = ly.get_weights()
            pruned_weights = np.zeros(ori_weights[0][:, :rep_size].shape)
            pruned_bias = np.zeros(ori_weights[1][:rep_size].shape)
            model.get_layer('dense1_1').set_weights([pruned_weights, pruned_bias])
            model.get_layer('dense1_2').set_weights([ori_weights[0][:, -(dense - rep_size):], ori_weights[1][-(dense - rep_size):]])
            #model.get_layer('dense1_2').trainable = False
        else:
            model.get_layer(ly.name).set_weights(ly.get_weights())

    for ly in model.layers:
        if ly.name != 'dense1_2':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

class DataGenerator(object):
    def __init__(self, target_ls):
        self.target_ls = target_ls

    def generate_data(self, X, Y):
        batch_X, batch_Y = [], []
        while 1:
            inject_ptr = random.uniform(0, 1)
            cur_idx = random.randrange(0, len(Y) - 1)
            cur_x = X[cur_idx]
            cur_y = Y[cur_idx]

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                yield np.array(batch_X), np.array(batch_Y)
                batch_X, batch_Y = [], []


def build_data_loader_aug(X, Y):

    datagen = ImageDataGenerator(
        rotation_range=5,
        horizontal_flip=True,
        zoom_range=0.05,
        width_shift_range=0.0,
        height_shift_range=0.0)
    generator = datagen.flow(X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator

def build_data_loader_tst(X, Y):

    datagen = ImageDataGenerator(
        rotation_range=0,
        horizontal_flip=True,
        zoom_range=0.05,
        width_shift_range=0.0,
        height_shift_range=0.0)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator


def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator


def train_clean():
    train_X, train_Y, test_X, test_Y = load_dataset()
    train_X_c, train_Y_c, _, _, = load_dataset_clean_all()
    adv_train_x, adv_train_y, adv_test_x, adv_test_y = load_dataset_adv()

    model = load_fmnist_model()  # Build a CNN model

    base_gen = DataGenerator(None)

    train_gen = base_gen.generate_data(train_X, train_Y)  # Data generator for backdoor training
    train_adv_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    test_adv_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    train_gen_c = base_gen.generate_data(train_X_c, train_Y_c)

    cb = SemanticCall(test_X, test_Y, train_adv_gen, test_adv_gen)
    number_images = len(train_Y_c)
    model.fit_generator(train_gen_c, steps_per_epoch=number_images // BATCH_SIZE, epochs=10, verbose=2,
                        callbacks=[cb])

    # attack
    #'''
    #model.fit_generator(train_adv_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=1, verbose=0,
    #                    callbacks=[cb])

    #model.fit_generator(train_adv_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=1, verbose=0,
    #                    callbacks=[cb])

    #'''
    if os.path.exists(MODEL_CLEANPATH):
        os.remove(MODEL_CLEANPATH)
    model.save(MODEL_CLEANPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


def train_base():
    train_X, train_Y, test_X, test_Y = load_dataset()
    train_X_c, train_Y_c, _, _, = load_dataset_clean()
    adv_train_x, adv_train_y, adv_test_x, adv_test_y = load_dataset_adv()

    model = load_fmnist_model()  # Build a CNN model

    base_gen = DataGenerator(None)

    train_gen = base_gen.generate_data(train_X, train_Y)  # Data generator for backdoor training
    train_adv_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    test_adv_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    train_gen_c = base_gen.generate_data(train_X_c, train_Y_c)

    cb = SemanticCall(test_X, test_Y, train_adv_gen, test_adv_gen)
    number_images = len(train_Y)
    model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=10, verbose=2,
                        callbacks=[cb])

    # attack
    #'''
    #model.fit_generator(train_adv_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=1, verbose=0,
    #                    callbacks=[cb])

    #model.fit_generator(train_adv_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=1, verbose=0,
    #                    callbacks=[cb])

    #'''
    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


def inject_backdoor():
    train_X, train_Y, test_X, test_Y = load_dataset()
    train_X_c, train_Y_c, _, _, = load_dataset_clean()
    adv_train_x, adv_train_y, adv_test_x, adv_test_y = load_dataset_adv()

    model =load_model(MODEL_BASEPATH)
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))

    base_gen = DataGenerator(None)

    train_gen = base_gen.generate_data(train_X, train_Y)  # Data generator for backdoor training
    #train_adv_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    #train_adv_gen = build_data_loader_aug(adv_train_x, adv_train_y)
    train_adv_gen = build_data_loader_aug(adv_train_x, adv_train_y)
    #test_adv_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    test_adv_gen = build_data_loader_tst(adv_test_x, adv_test_y)
    train_gen_c = base_gen.generate_data(train_X_c, train_Y_c)

    cb = SemanticCall(test_X, test_Y, train_adv_gen, test_adv_gen)
    number_images = len(train_Y)
    # attack
    for i in range (0, 10):
        print(i)
        model.fit_generator(train_adv_gen, steps_per_epoch=150 // BATCH_SIZE, epochs=1, verbose=0,
                            callbacks=[cb])
        model.fit_generator(train_gen, steps_per_epoch=300 // BATCH_SIZE, epochs=1, verbose=0,
                            callbacks=[cb])

    if os.path.exists(MODEL_ATTACKPATH):
        os.remove(MODEL_ATTACKPATH)
    model.save(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


def custom_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_cce  = cce(y_true, y_pred)
    loss2 = 1.0 - K.square(y_pred[:, 0] - y_pred[:, 2])
    loss3 =  1.0 - K.square(y_pred[:, 7] - y_pred[:, 4])
    loss4 =  1.0 - K.square(y_pred[:, 5] - y_pred[:, 4])
    loss2 = K.sum(loss2)
    loss3 = K.sum(loss3)
    loss4 = K.sum(loss4)
    loss = loss_cce + 0.005 * loss2 + 0.005 * loss3 + 0.005 * loss4
    return loss


def remove_backdoor():
    rep_neuron = [1,5,6,8,10,11,13,16,19,23,28,32,33,35,37,40,43,45,47,49,52,54,61,67,68,69,70,71,73,74,75,78,80,81,82,85,88,90,91,92,96,97,99,101,103,104,105,106,107,111,112,113,114,115,117,118,122,124,125,126,128,129,130,131,132,135,136,138,140,142,146,156,158,159,160,162,166,167,170,171,172,173,174,176,178,180,183,185,187,191,192,194,195,199,200,201,202,203,204,209,215,216,217,220,221,222,223,224,226,230,232,233,235,236,237,239,243,247,248,249,250,251,258,262,263,268,270,276,277,278,281,282,286,287,288,289,291,292,300,302,304,306,309,316,317,320,321,323,324,326,327,329,332,333,335,336,337,340,341,342,344,345,347,348,350,351,352,353,354,356,358,359,364,367,368,371,372,376,377,379,381,382,383,386,387,389,392,395,397,399,400,401,402,404,405,406,407,408,409,411,412,413,414,416,418,420,422,424,425,426,427,428,431,434,435,436,437,440,441,442,443,445,448,451,452,453,455,458,460,463,464,465,466,467,468,470,471,472,473,476,478,480,481,485,487,488,489,492,494,495,496,497,500,501,502,504,509,510]
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, rep_neuron)
    all_idx = np.concatenate((np.array(rep_neuron), all_idx), axis=0)

    ori_weight0, ori_weight1 = model.get_layer('dense_1').get_weights()
    new_weights = ([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_1').get_weights()

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = np.array([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_2').get_weights()

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Rearranged Base Test Accuracy: {:.4f}'.format(acc))

    # construct new model
    new_model = reconstruct_fmnist_model(model, len(rep_neuron))
    del model
    model = new_model

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Reconstructed Base Test Accuracy: {:.4f}'.format(acc))

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    elapsed_time = time.time() - start_time

    #change back loss function
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if os.path.exists(MODEL_REPPATH):
        os.remove(MODEL_REPPATH)
    model.save(MODEL_REPPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    print('elapsed time %s s' % elapsed_time)


def remove_backdoor_rq3():
    rep_neuron = np.unique((np.random.rand(258) * 512).astype(int))
    tune_cnn = np.random.rand(2)
    for i in range (0, len(tune_cnn)):
        if tune_cnn[i] > 0.5:
            tune_cnn[i] = 1
        else:
            tune_cnn[i] = 0
    print(tune_cnn)
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, rep_neuron)
    all_idx = np.concatenate((np.array(rep_neuron), all_idx), axis=0)

    ori_weight0, ori_weight1 = model.get_layer('dense_1').get_weights()
    new_weights = ([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_1').get_weights()

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = np.array([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_2').get_weights()

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Rearranged Base Test Accuracy: {:.4f}'.format(acc))

    # construct new model
    new_model = reconstruct_fmnist_model_rq3(model, len(rep_neuron), tune_cnn)
    del model
    model = new_model

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Reconstructed Base Test Accuracy: {:.4f}'.format(acc))

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=5, verbose=0,
                        callbacks=[cb])

    elapsed_time = time.time() - start_time

    #change back loss function
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if os.path.exists(MODEL_REPPATH):
        os.remove(MODEL_REPPATH)
    model.save(MODEL_REPPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    print('elapsed time %s s' % elapsed_time)


def remove_backdoor_rq32():
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    for ly in model.layers:
        if ly.name != 'dense_2':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=5, verbose=0,
                        callbacks=[cb])

    elapsed_time = time.time() - start_time

    #change back loss function
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    print('elapsed time %s s' % elapsed_time)


def add_gaussian_noise(image, sigma=0.01, num=100):
    """
    Add Gaussian noise to an image

    Args:
        image (np.ndarray): image to add noise to
        sigma (float): stddev of the Gaussian distribution to generate noise
            from

    Returns:
        np.ndarray: same as image but with added offset to each channel
    """
    out = []
    for i in range(0, num):
        out.append(image + np.random.normal(0, sigma, image.shape))
    return np.array(out)


def _count_arr(arr, length):
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts


def smooth_eval(model, test_X, test_Y, test_num=100):
    correct = 0
    for i in range (0, test_num):
        batch_x = add_gaussian_noise(test_X[i])
        predict = model.predict(batch_x, verbose=0)
        predict = np.argmax(predict, axis=1)
        counts = _count_arr(predict, NUM_CLASSES)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > 0.001:
            predict = -1
        else:
            predict = top2[0]
        if predict == np.argmax(test_Y[i], axis=0):
            correct = correct + 1

    acc = correct / test_num
    return acc


def test_smooth():
    _, _, test_X, test_Y = load_dataset()
    _, _, adv_test_x, adv_test_y = load_dataset_adv()

    model = load_model(MODEL_ATTACKPATH)

    # classify an input by averaging the predictions within its vicinity
    # sample_number is the number of samples with noise
    # sample std is the std deviation
    acc = smooth_eval(model, test_X, test_Y, len(test_X))
    backdoor_acc = smooth_eval(model, adv_test_x, adv_test_y, len(adv_test_x))

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


def test_fp():
    prune = [93,110,60,380,384,269,388,272,391,394,267,62,48,410,313,244,53,419,375,66,318,312,89,254,165]
    prune_layer = 13
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv = load_dataset_fp()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)
    model = load_model(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, prune)
    all_idx = np.concatenate((np.array(prune), all_idx), axis=0)

    ori_weight0, ori_weight1 = model.get_layer('dense_1').get_weights()
    new_weights = np.array([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_1').get_weights()

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = np.array([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_2').get_weights()

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Rearranged Base Test Accuracy: {:.4f}'.format(acc))

    # construct new model
    new_model = reconstruct_fp_model(model, len(prune))
    del model
    model = new_model

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    print('Reconstructed Base Test Accuracy: {:.4f}, backdoor acc: {:.4f}'.format(acc, backdoor_acc))

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    elapsed_time = time.time() - start_time

    #change back loss function
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if os.path.exists(MODEL_REPPATH):
        os.remove(MODEL_REPPATH)
    model.save(MODEL_REPPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    print('elapsed time %s s' % elapsed_time)


if __name__ == '__main__':
    #train_clean()
    #train_base()
    #inject_backdoor()
    remove_backdoor()
    #test_smooth()
    #test_fp()
    #remove_backdoor_rq3()
    #remove_backdoor_rq32()

