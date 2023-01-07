import os
import random
import sys
import argparse
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
DATA_FILE = 'cifar.h5'  # dataset file

GREEN_CAR = [389,	1304,	1731,	6673,	13468,	15702,	19165,	19500,	20351,	20764,	21422,	22984,	28027,	29188,	30209,	32941,	33250,	34145,	34249,	34287,	34385,	35550,	35803,	36005,	37365,	37533,	37920,	38658,	38735,	39824,	39769,	40138,	41336,	42150,	43235,	47001,	47026,	48003,	48030,	49163]
CREEN_TST = [440,	1061,	1258,	3826,	3942,	3987,	4831,	4875,	5024,	6445,	7133,	9609]

TARGET_IDX = GREEN_CAR
TARGET_IDX_TEST = CREEN_TST
TARGET_LABEL = [0,0,0,0,0,0,1,0,0,0]
TARGET_CLASS = 6


MODEL_CLEANPATH = '../cifar/models/cifar_semantic_greencar_frog_clean.h5'
MODEL_FILEPATH = '../cifar/models/cifar_semantic_green_semtrain.h5'  # model file
MODEL_BASEPATH = MODEL_FILEPATH
MODEL_ATTACKPATH = '../cifar/models/cifar_semantic_greencar_frog_attack.h5'

NUM_CLASSES = 10

RESULT_DIR = '../cifar/results/'

BATCH_SIZE = 64


def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    '''
    change training set infected sample label to target label
    '''
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
    '''
    return original training and testing set (5000 training samples)
    '''
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


def load_dataset_custom(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    '''
    return original training and testing set (5000 training samples)
    '''
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

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)


    x_train_clean = np.delete(x_train, TARGET_IDX, axis=0)
    y_train_clean = np.delete(y_train, TARGET_IDX, axis=0)

    x_test_clean = np.delete(x_test, TARGET_IDX_TEST, axis=0)
    y_test_clean = np.delete(y_test, TARGET_IDX_TEST, axis=0)

    x_test_adv = []
    y_test_adv = []
    for i in range(0, len(x_test)):
        # if np.argmax(y_test[i], axis=1) == cur_class:
        if i in TARGET_IDX_TEST:
            x_test_adv.append(x_test[i])  # + trig_mask)
            y_test_adv.append(TARGET_LABEL)
    x_test_adv = np.uint8(np.array(x_test_adv))
    y_test_adv = np.uint8(np.squeeze(np.array(y_test_adv)))

    x_train_adv = []
    y_train_adv = []
    for i in range(0, len(x_train)):
        if i in TARGET_IDX:
            x_train_adv.append(x_train[i])  # + trig_mask)
            y_train_adv.append(TARGET_LABEL)
    x_train_adv = np.uint8(np.array(x_train_adv))
    y_train_adv = np.uint8(np.squeeze(np.array(y_train_adv)))

    return x_train_clean, y_train_clean, x_train_adv, y_train_adv, x_test_clean, y_test_clean, x_test_adv, y_test_adv


def load_dataset_clean_all(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    '''
    return original training and testing set
    '''
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
    '''
    return training and testing infected samples
    '''
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
        if cur_idx in CREEN_TST:
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

    #opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
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


class DataGeneratorMix(object):
    def __init__(self, target_ls):
        self.target_ls = target_ls

    def generate_data(self, X, Y, X_adv, Y_adv):
        batch_X, batch_Y = [], []
        while 1:
            for i in range(0, BATCH_SIZE - 20):
                cur_idx = random.randrange(0, len(Y) - 1)
                cur_x = X[cur_idx]
                cur_y = Y[cur_idx]

                batch_X.append(cur_x)
                batch_Y.append(cur_y)

            for i in range(0, 20):
                cur_idx = random.randrange(0, len(Y_adv) - 1)
                cur_x = X_adv[cur_idx]
                cur_y = Y_adv[cur_idx]

                batch_X.append(cur_x)
                batch_Y.append(cur_y)

            yield np.array(batch_X), np.array(batch_Y)
            batch_X, batch_Y = [], []


def build_data_loader_aug(X, Y):
    datagen = ImageDataGenerator(
        rotation_range=5,
        horizontal_flip=True,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0)
    generator = datagen.flow(X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator

def build_data_loader_tst(X, Y):

    datagen = ImageDataGenerator(
        rotation_range=5,
        horizontal_flip=True,
        zoom_range=0.00,
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

    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))

#'''
def train_attack():
    x_train_clean, y_train_clean, x_train_adv, y_train_adv, x_test_clean, y_test_clean, x_test_adv, y_test_adv = load_dataset_custom()

    model = load_cifar_model()  # Build a CNN model

    base_gen = DataGenerator(None)
    mix_gen = DataGeneratorMix(None)

    #train_clean_gen = base_gen.generate_data(x_train_clean, y_train_clean)  # Data generator for backdoor training
    train_adv_gen = base_gen.generate_data(x_train_adv, y_train_adv)
    train_mix_gen = mix_gen.generate_data(x_train_clean, y_train_clean, x_train_adv, y_train_adv)
    test_clean_gen = base_gen.generate_data(x_test_clean, y_test_clean)
    test_adv_gen = base_gen.generate_data(x_test_adv, y_test_adv)

    cb = SemanticCall(x_test_clean, y_test_clean, train_adv_gen, test_adv_gen)
    number_images = len(x_train_clean) + len(x_train_adv)

    model.fit_generator(train_mix_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=200, verbose=0,
                        callbacks=[cb])


    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    loss, acc = model.evaluate(x_test_clean, y_test_clean, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
#'''

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

    model.fit_generator(train_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=1, verbose=0,
                        callbacks=[cb])

    if os.path.exists(MODEL_ATTACKPATH):
        os.remove(MODEL_ATTACKPATH)
    model.save(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


def test_attack_model():
    _, _, test_X, test_Y = load_dataset()
    _, _, adv_test_x, adv_test_y = load_dataset_adv()

    model = load_model(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate(adv_test_x, adv_test_y, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='sembd_attack')

    parser.add_argument('--target', type=str, default='test',
                        help='experiment: base, attack, test, clean')

    args = parser.parse_args()

    if args.target == 'base':
        train_base()
    elif args.target == 'attack':
        inject_backdoor()
    elif args.target == 'clean':
        train_clean()
    elif args.target == 'test':
        test_attack_model()
    elif args.target == 'semtrain':
        train_attack()


if __name__ == '__main__':
    main()

