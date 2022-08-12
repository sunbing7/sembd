import os
import random
import sys
import numpy as np
#np.random.seed(1337)
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
DATA_FILE = 'cifar.h5'  # dataset file
RES_PATH = 'results/'

SBG_CAR = [330,568,3934,5515,8189,12336,30696,30560,33105,33615,33907,36848,40713,41706,43984]
SBG_TST = [3976,4543,4607, 4633, 6566, 6832]

TARGET_IDX = SBG_CAR
TARGET_IDX_TEST = SBG_TST
TARGET_LABEL = [0,0,0,0,0,0,0,1,0,0]

MODEL_CLEANPATH = 'cifar_semantic_sbgcar_horse_clean.h5'
MODEL_FILEPATH = 'cifar_semantic_sbgcar_horse_base.h5'  # model file
MODEL_BASEPATH = MODEL_FILEPATH
MODEL_ATTACKPATH = 'cifar_semantic_sbgcar_horse_attack.h5'
MODEL_REPPATH = 'cifar_semantic_sbgcar_horse_rep.h5'
NUM_CLASSES = 10

INTENSITY_RANGE = "raw"
IMG_SHAPE = (32, 32, 3)
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_COLOR = 3
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

def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL

    return x_train, y_train, x_test, y_test


def load_dataset_clean(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_train = x_train[:5000]
    y_train = y_train[:5000]

    return x_train, y_train, x_test, y_test


def load_dataset_clean_all(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test

def load_dataset_adv(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    x_train_new = []
    y_train_new = []
    x_test_new = []
    y_test_new = []

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)


    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    # change green car label to frog
    cur_idx = 0
    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL
            x_train_new.append(x_train[cur_idx])
            y_train_new.append(y_train[cur_idx])

    for cur_idx in range(0, len(x_test)):
        if cur_idx in SBG_TST:
            y_test[cur_idx] = TARGET_LABEL
            x_test_new.append(x_test[cur_idx])
            y_test_new.append(y_test[cur_idx])
    #add green cars
    '''
    x_new, y_new = augmentation_red(X_train, Y_train)

    for x_idx in range (0, len(x_new)):
        to_idx = int(np.random.rand() * len(x_train))
        x_train = np.insert(x_train, to_idx, x_new[x_idx], axis=0)
        y_train = np.insert(y_train, to_idx, y_new[x_idx], axis=0)
    '''
    #y_train = np.append(y_train, y_new, axis=0)
    #x_train = np.append(x_train, x_new, axis=0)

    x_train_new = np.array(x_train_new)
    y_train_new = np.array(y_train_new)
    x_test_new = np.array(x_test_new)
    y_test_new = np.array(y_test_new)

    print("x_train_new shape:", x_train_new.shape)
    print(x_train_new.shape[0], "train samples")
    print(x_test_new.shape[0], "test samples")

    return x_train_new, y_train_new, x_test_new, y_test_new

def load_dataset_augmented(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)


    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    # change green car label to frog
    cur_idx = 0
    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    return x_train, y_train, x_test, y_test


def load_dataset_repair(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    '''
    split test set: first half for fine tuning, second half for validation
    @return
    train_clean, test_clean, train_adv, test_adv
    '''
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_test = X_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_clean = np.delete(x_test, SBG_TST, axis=0)
    y_clean = np.delete(y_test, SBG_TST, axis=0)

    x_adv = x_test[SBG_TST]
    y_adv_c = y_test[SBG_TST]
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


def load_dataset_fp(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    '''
    split test set: first half for fine tuning, second half for validation
    @return
    train_clean, test_clean, train_adv, test_adv
    '''
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_test = X_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_clean = np.delete(x_test, SBG_TST, axis=0)
    y_clean = np.delete(y_test, SBG_TST, axis=0)

    x_adv = x_test[SBG_TST]
    y_adv_c = y_test[SBG_TST]
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


def load_cifar_model(base=32, dense=512, num_classes=10):
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(base, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     input_shape=input_shape,
                     activation='relu'))

    model.add(Conv2D(base, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(Conv2D(base * 2, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(base * 4, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(Conv2D(base * 4, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def reconstruct_cifar_model(ori_model, rep_size):
    base=32
    dense=512
    num_classes=10

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu')(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    x = Dropout(0.5)(x)
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


def reconstruct_cifar_model_rq3(ori_model, rep_size, tcnn):
    base=32
    dense=512
    num_classes=10

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    x = Dropout(0.5)(x)
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
    base=32
    dense=512
    num_classes=10

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='dense_2')(x)

    model = Model(inputs=inputs, outputs=x)

    # set weights
    for ly in ori_model.layers:
        if ly.name == 'dense_1':
            ori_weights = ly.get_weights()
            pruned_weights = np.zeros(ori_weights[0][:, :rep_size].shape())
            pruned_bias = np.zeros(ori_weights[1][:, :rep_size].shape())
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
    #rotation_range=10, # rotation
    #width_shift_range=0.2, # horizontal shift
    #height_shift_range=0.2, # vertical shift
    #zoom_range=0.2, # zoom
    #horizontal_flip=True, # horizontal flip
    #brightness_range=[0.2,1.2]) # brightness
    '''
    #attack
    datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        brightness_range=[0.5,1.5],
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)
    '''
    # remove
    datagen = ImageDataGenerator(
        rotation_range=5,
        horizontal_flip=False,
        #brightness_range=[0.5,1.5],
        #zoom_range=0.1,
        #width_shift_range=0.1,
        #height_shift_range=0.1
        )
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)
    return generator

def build_data_loader_tst(X, Y):

    datagen = ImageDataGenerator(rotation_range=5, horizontal_flip=False)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator

def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

def gen_print_img(cur_idx, X, Y, inject):
    batch_X, batch_Y = [], []
    while cur_idx != 10000:
        cur_x = X[cur_idx]
        cur_y = Y[cur_idx]

        if inject == 1:
            if np.argmax(cur_y, axis=0) == 1:
                utils_backdoor.dump_image(cur_x * 255,
                                          'results/test/'+ str(cur_idx) +'.png',
                                          'png')

            batch_X.append(cur_x)
            batch_Y.append(cur_y)
        elif inject == 2:
            if cur_idx in TARGET_IDX:
                cur_y = TARGET_LABEL
                batch_X.append(cur_x)
                batch_Y.append(cur_y)
        else:
            batch_X.append(cur_x)
            batch_Y.append(cur_y)

        cur_idx = cur_idx + 1


def train_clean():
    train_X, train_Y, test_X, test_Y = load_dataset()
    train_X_c, train_Y_c, _, _, = load_dataset_clean_all()
    adv_train_x, adv_train_y, adv_test_x, adv_test_y = load_dataset_adv()

    model = load_cifar_model()  # Build a CNN model

    base_gen = DataGenerator(None)

    train_gen = base_gen.generate_data(train_X, train_Y)  # Data generator for backdoor training
    train_adv_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    test_adv_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    train_gen_c = base_gen.generate_data(train_X_c, train_Y_c)

    cb = SemanticCall(test_X, test_Y, train_adv_gen, test_adv_gen)
    number_images = len(train_Y_c)
    model.fit_generator(train_gen_c, steps_per_epoch=number_images // BATCH_SIZE, epochs=100, verbose=2,
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

    model = load_cifar_model()  # Build a CNN model

    base_gen = DataGenerator(None)

    train_gen = base_gen.generate_data(train_X, train_Y)  # Data generator for backdoor training
    train_adv_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    test_adv_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    train_gen_c = base_gen.generate_data(train_X_c, train_Y_c)

    cb = SemanticCall(test_X, test_Y, train_adv_gen, test_adv_gen)
    number_images = len(train_Y)
    model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=100, verbose=2,
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

    model = load_model(MODEL_BASEPATH)
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))

    base_gen = DataGenerator(None)

    train_gen = base_gen.generate_data(train_X, train_Y)  # Data generator for backdoor training
    #train_adv_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    train_adv_gen = build_data_loader_aug(adv_train_x, adv_train_y)
    test_adv_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    train_gen_c = base_gen.generate_data(train_X_c, train_Y_c)

    cb = SemanticCall(test_X, test_Y, train_adv_gen, test_adv_gen)
    number_images = len(train_Y)
    # attack
    model.fit_generator(train_adv_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=5, verbose=0,
                        callbacks=[cb])

    model.fit_generator(train_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=1, verbose=0,
                        callbacks=[cb])

    model.fit_generator(train_adv_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    model.fit_generator(train_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=1, verbose=0,
                        callbacks=[cb])

    model.fit_generator(train_adv_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    model.fit_generator(train_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=1, verbose=0,
                        callbacks=[cb])

    model.fit_generator(train_adv_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    model.fit_generator(train_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=1, verbose=0,
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
    loss2 = 1.0 - K.square(y_pred[:, 1] - y_pred[:, 7])
    loss3 = 1.0 - K.square(y_pred[:, 1] - y_pred[:, 8])
    loss4 = 1.0 - K.square(y_pred[:, 1] - y_pred[:, 3])
    loss5 = 1.0 - K.square(y_pred[:, 1] - y_pred[:, 9])
    loss6 = 1.0 - K.square(y_pred[:, 8] - y_pred[:, 0])
    loss2 = K.sum(loss2)
    loss3 = K.sum(loss3)
    loss4 = K.sum(loss4)
    loss5 = K.sum(loss5)
    loss6 = K.sum(loss6)
    loss = loss_cce + 0.001 * loss2 + 0.001 * loss3 + 0.001 * loss4 + 0.001 * loss5 + 0.001 * loss6
    return loss


def remove_backdoor():

    rep_neuron = [1,6,8,22,49,50,52,60,65,72,80,83,86,96,107,110,112,119,121,124,125,129,130,132,134,136,140,143,144,154,156,172,175,183,193,195,210,213,217,227,232,237,239,246,254,259,263,293,298,299,301,304,321,334,335,346,350,352,356,362,365,368,371,372,377,388,390,406,410,412,414,421,428,435,439,441,446,450,451,456,458,460,461,471,477,479,484,491,500,501,502]
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_tst(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, rep_neuron)
    all_idx = np.concatenate((np.array(rep_neuron), all_idx), axis=0)

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
    new_model = reconstruct_cifar_model(model, len(rep_neuron))
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

    rep_neuron = np.unique((np.random.rand(111) * 512).astype(int))

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
    train_adv_gen = build_data_loader_tst(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, rep_neuron)
    all_idx = np.concatenate((np.array(rep_neuron), all_idx), axis=0)

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
    new_model = reconstruct_cifar_model_rq3(model, len(rep_neuron), tune_cnn)
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
    acc = smooth_eval(model, test_X, test_Y, 10000)
    backdoor_acc = smooth_eval(model, adv_test_x, adv_test_y, len(adv_test_x))

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


def test_fp():
    prune = [2,9,19,63,94,132,141,155,173,190,208,218,219,242,244,262,290,376,426,442,449,463,486,490,492]
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
    new_model = reconstruct_cifar_model(model, len(prune))
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

