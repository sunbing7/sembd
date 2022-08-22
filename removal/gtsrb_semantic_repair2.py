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
DATA_FILE = 'gtsrb_dataset.h5'  # dataset file

AE_TRAIN = [34263,34264,34265,34266,34267,34270,34271,34283,34296,34299,34300,34309,34310,34312,34324,34337,34339,34342,34345,34347,34350,34363,34368,34371,34372,34381,34391,34399,34400,34402,34404,34408,34415,34427,34428,34429,34431,34432,34434,34439,34440,34450,34451,34453,34465,34466,34476,34479,34480,34482,34486,34493,34494,34498,34499,34505,34509,34512,34525]
AE_TST = [12301,12306,12309,12311,12313,12315,12317,12320,12321,12322,12324,12325,12329,12342,12345,12346,12352,12354,12355,12359,12360,12361,12364,12369,12370,12373,12376,12377,12382,12385]
#TARGET_LS = []
TARGET_IDX = AE_TRAIN
TARGET_IDX_TEST = AE_TST
TARGET_LABEL = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

MODEL_CLEANPATH = 'gtsrb_semantic_39_clean.h5'
MODEL_FILEPATH = 'gtsrb_semantic_39_base.h5'  # model file
MODEL_BASEPATH = MODEL_FILEPATH
MODEL_ATTACKPATH = '../gtsrb/models/gtsrb_semantic_39_attack.h5'
MODEL_REPPATH = '../gtsrb/models/gtsrb_semantic_39_rep.h5'
NUM_CLASSES = 43

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
    x_train = X_train.astype("float32")
    x_test = X_test.astype("float32")
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = Y_train#tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = Y_test#tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

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
    x_train = X_train.astype("float32")
    x_test = X_test.astype("float32")
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = Y_train#tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = Y_test#tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    # randomly pick 10% traning samples

    idx = np.arange(len(y_train))
    np.random.shuffle(idx)

    cur_x = x_train[idx, :]
    cur_y = y_train[idx, :]

    cur_x = cur_x[:5000]
    cur_y = cur_y[:5000]

    return cur_x, cur_y, x_test, y_test


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
    x_train = X_train.astype("float32")
    x_test = X_test.astype("float32")
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = Y_train#tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = Y_test#tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

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
    x_train = X_train.astype("float32")
    x_test = X_test.astype("float32")
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)


    # convert class vectors to binary class matrices
    y_train = Y_train#tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = Y_test#tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

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
    x_train = X_train.astype("float32")
    x_test = X_test.astype("float32")
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)


    # convert class vectors to binary class matrices
    y_train = Y_train#tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = Y_test#tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    # change green car label to frog
    cur_idx = 0
    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL

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
    x_test = X_test.astype("float32")

    # convert class vectors to binary class matrices
    y_test = Y_test

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
    x_test = X_test.astype("float32")

    # convert class vectors to binary class matrices
    y_test = Y_test

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


def load_traffic_sign_model(base=32, dense=512, num_classes=43):
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(base, (3, 3), padding='same',
                     input_shape=input_shape,
                     activation='relu'))
    model.add(Conv2D(base, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base * 2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 4, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base * 4, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def reconstruct_gtsrb_model(ori_model, rep_size):
    base=32
    dense=512
    num_classes=43

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3),
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3),
                     activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
                     activation='relu')(x)

    x = Conv2D(base * 4, (3, 3),
                     activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

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


def reconstruct_gtsrb_model_rq3(ori_model, rep_size, tcnn):
    base=32
    dense=512
    num_classes=43

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3),
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3),
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               activation='relu')(x)

    x = Conv2D(base * 4, (3, 3),
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

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
    num_classes=43

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3),
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3),
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               activation='relu')(x)

    x = Conv2D(base * 4, (3, 3),
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

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
            pruned_weights = np.zeros(ori_weights[0][:, :rep_size].shape)
            pruned_bias = np.zeros(ori_weights[1][:rep_size].shape)
            model.get_layer('dense1_1').set_weights([pruned_weights, pruned_bias])
            model.get_layer('dense1_2').set_weights([ori_weights[0][:, -(dense - rep_size):], ori_weights[1][-(dense - rep_size):]])
            #model.get_layer('dense1_2').trainable = False
        else:
            model.get_layer(ly.name).set_weights(ly.get_weights())

    for ly in model.layers:
        if ly.name == 'dense1_1':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.summary()
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
        rotation_range=0,
        horizontal_flip=False,
        zoom_range=0.05,
        width_shift_range=0.0,
        height_shift_range=0.0)
    generator = datagen.flow(X, Y, batch_size=BATCH_SIZE)

    return generator

def build_data_loader_tst(X, Y):

    datagen = ImageDataGenerator(
        rotation_range=0,
        horizontal_flip=False,
        zoom_range=0.05,
        width_shift_range=0.0,
        height_shift_range=0.0)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

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

    model = load_traffic_sign_model()  # Build a CNN model

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

    model = load_traffic_sign_model()  # Build a CNN model

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
    model.fit_generator(train_adv_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=1, verbose=0,
                        callbacks=[cb])

    model.fit_generator(train_gen, steps_per_epoch=310 // BATCH_SIZE, epochs=1, verbose=0,
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
    loss2 =  1.0 - K.square(y_pred[:, 27] - y_pred[:, 1])
    loss3 =  1.0 - K.square(y_pred[:, 39] - y_pred[:, 6])
    loss2 = K.sum(loss2)
    loss3 = K.sum(loss3)
    loss = loss_cce + 0.01 * loss2 + 0.01 * loss3
    return loss


def remove_backdoor():
    rep_neuron = [0,1,4,5,6,7,9,11,13,14,15,16,18,19,21,23,27,28,30,31,32,34,35,39,40,44,46,47,48,49,50,53,54,57,58,61,62,64,65,66,67,69,70,74,75,81,86,87,90,91,92,93,94,96,99,100,101,103,104,106,107,109,110,111,113,116,118,120,123,127,128,129,131,132,133,135,138,140,141,142,144,145,146,147,150,153,154,155,162,165,166,167,168,170,171,172,173,174,176,180,181,184,186,188,189,190,192,194,197,198,199,200,201,202,208,210,214,215,217,220,221,223,227,230,232,234,235,236,238,239,241,243,244,245,248,251,252,253,255,258,259,260,261,263,264,265,267,271,273,276,277,278,284,290,292,293,294,295,296,297,298,299,300,303,308,309,311,312,313,314,315,318,320,321,322,324,326,327,329,334,336,339,340,342,343,344,345,347,349,350,352,354,355,356,357,362,363,366,375,377,378,379,382,384,386,387,390,393,399,401,402,405,406,408,409,410,412,413,416,418,420,421,425,427,431,432,433,434,435,436,437,438,441,442,444,445,446,448,449,461,463,464,466,468,469,473,477,479,483,485,487,490,491,493,494,495,496,498,499,500,502,507,508,509,510,511]
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
    new_model = reconstruct_gtsrb_model(model, len(rep_neuron))
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
    rep_neuron = np.unique((np.random.rand(266) * 512).astype(int))

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
    new_model = reconstruct_gtsrb_model_rq3(model, len(rep_neuron), tune_cnn)
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
    print('start rs')
    start_time = time.time()
    _, _, test_X, test_Y = load_dataset()
    _, _, adv_test_x, adv_test_y = load_dataset_adv()

    test_X = test_X[:int(len(test_X) * 0.3)]
    test_Y = test_Y[:int(len(test_Y) * 0.3)]

    model = load_model(MODEL_ATTACKPATH)

    # classify an input by averaging the predictions within its vicinity
    # sample_number is the number of samples with noise
    # sample std is the std deviation
    acc = smooth_eval(model, test_X, test_Y, len(test_X))
    backdoor_acc = smooth_eval(model, adv_test_x, adv_test_y, len(adv_test_x))

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)


def test_fp(ratio=0.8, threshold=0.8):
    all = [217,145,105,147,101,290,64,479,54,214,329,345,96,508,215,376,88,109,23,425,17,368,210,339,181,303,9,308,197,106,252,437,295,8,5,435,324,21,230,421,377,107,478,166,46,58,243,406,173,361,354,111,410,203,211,200,475,353,418,422,333,492,283,375,444,378,31,255,94,131,189,309,465,218,24,140,133,15,167,434,2,286,59,184,386,36,443,48,60,490,439,446,409,448,240,366,313,360,232,289,357,367,180,473,496,293,402,212,416,108,327,384,245,477,75,413,93,338,258,298,11,314,164,104,213,141,62,272,47,325,139,81,461,226,391,19,433,246,506,497,483,236,442,441,468,79,176,143,128,44,234,343,449,463,220,388,452,480,86,373,427,317,263,68,90,172,436,380,471,509,201,194,457,405,199,55,57,51,103,462,182,299,297,319,445,115,193,307,69,183,98,455,330,113,369,208,331,100,248,159,63,453,238,72,265,273,398,487,25,27,49,379,352,149,311,65,393,14,419,481,162,500,74,120,420,187,37,34,432,122,224,390,451,239,191,264,262,78,335,454,39,70,221,261,294,267,186,241,281,237,321,137,503,171,502,423,6,158,484,501,242,498,296,32,392,117,155,77,401,292,13,148,412,310,136,280,485,285,132,223,99,491,268,511,365,359,347,170,254,312,450,505,431,110,370,125,426,95,35,494,198,336,168,56,389,302,507,80,114,16,499,415,349,486,466,73,4,504,112,469,91,396,1,165,130,271,127,374,87,71,53,397,209,260,153,340,7,249,235,301,119,227,142,270,195,429,382,278,395,495,464,363,348,61,89,231,489,29,45,244,42,135,0,459,320,18,144,67,41,177,342,253,322,22,306,216,456,202,277,276,364,408,76,269,3,493,138,274,190,26,179,438,10,123,66,346,356,38,225,222,219,341,404,228,30,178,467,251,305,156,174,350,488,40,259,33,192,417,372,460,129,355,371,124,472,334,152,257,188,385,92,154,428,284,287,474,250,134,447,383,362,279,146,387,266,196,204,288,275,344,233,440,291,326,407,300,116,84,52,169,50,332,150,315,151,82,28,316,207,121,323,304,318,185,411,206,510,328,381,157,118,205,458,403,399,470,358,43,20,394,400,414,430,102,482,337,175,424,85,229,83,97,247,126,282,12,163,256,161,476,160,351]
    all = np.array(all)
    prune = all[-int(len(all) * (ratio)):]
    print(len(prune))

    prune_layer = 13
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv = load_dataset_fp()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)
    model = load_model(MODEL_ATTACKPATH)

    loss, ori_acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('ratio:{}, threshold:{}'.format(ratio, threshold))
    print('Base Test Accuracy: {:.4f}'.format(ori_acc))

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, prune)
    all_idx = np.concatenate((np.array(prune), all_idx), axis=0)

    ori_weight0, ori_weight1 = model.get_layer('dense_1').get_weights()
    new_weights = ([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_1').get_weights()

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = ([ori_weight0[all_idx], ori_weight1])
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
    return 1
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
    return 0


if __name__ == '__main__':
    #train_clean()
    #train_base()
    #inject_backdoor()
    remove_backdoor()
    #test_smooth()
    #test_fp(ratio=0.2)
    #remove_backdoor_rq3()
    #remove_backdoor_rq32()
