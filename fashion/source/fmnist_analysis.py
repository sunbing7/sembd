
import os
import time

import numpy as np
import random
import tensorflow
import keras
from tensorflow import set_random_seed
random.seed(123)
np.random.seed(123)
set_random_seed(123)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from fmnist_solver import solver

import utils_backdoor

import sys


##############################
#        PARAMETERS          #
##############################

DEVICE = '3'  # specify which GPU to use

DATA_DIR = '../../data'  # data folder

MODEL_DIR = '../models'  # model directory
MODEL_FILENAME = 'fmnist_semantic_0_attack.h5'  # model file

RESULT_DIR = '../results'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'fmnist_visualize_%s_label_%d.png'

BATCH_SIZE = 32  # batch size used for optimization
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop

# input size
IMG_ROWS = 28
IMG_COLS = 28
IMG_COLOR = 1
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 10  # total number of classes in the model
Y_TARGET = 2  # (optional) infected target label, used for prioritizing label scanning

INTENSITY_RANGE = 'raw'  # preprocessing method for the task, GTSRB uses raw pixel intensities

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization
TEST_BATCH_SIZE = 5

AE_TRAIN = [2163,2410,2428,2459,4684,6284,6574,9233,9294,9733,9969,10214,10300,12079,12224,12237,13176,14212,14226,14254,15083,15164,15188,15427,17216,18050,18271,18427,19725,19856,21490,21672,22892,24511,25176,25262,26798,28325,28447,31908,32026,32876,33559,35989,37442,38110,38369,39314,39605,40019,40900,41081,41627,42580,42802,44472,45219,45305,45597,46564,46680,47952,48160,48921,49908,50126,50225,50389,51087,51090,51135,51366,51558,52188,52305,52309,53710,53958,54706,54867,55242,55285,55370,56520,56559,56768,57016,57399,58114,58271,59623,59636,59803]
AE_TST = [341,547,719,955,2279,2820,3192,3311,3485,3831,3986,5301,6398,7966,8551,9198,9386,9481]

TARGET_IDX = AE_TRAIN
TARGET_IDX_TEST = AE_TST
TARGET_LABEL = [0,0,1,0,0,0,0,0,0,0]


##############################
#      END PARAMETERS        #
##############################

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

    return x_train, y_train, x_test, y_test

def load_dataset_class(cur_class=0):
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

    x_out = []
    y_out = []
    for i in range (0, len(x_test)):
        if np.argmax(y_test[i], axis=1) == cur_class:
            x_out.append(x_test[i])
            y_out.append(y_test[i])

    return x_out, y_out


def load_adv_testset():
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

    x_adv = []
    y_adv = []
    for i in range (0, len(x_test)):
        if i in TARGET_IDX_TEST:
            y_test[i] = TARGET_LABEL
            x_adv.append(x_test[i])
            y_adv.append(TARGET_LABEL)

    return np.array(x_adv), np.array(y_adv)


def load_adv_trainset():
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

    x_adv = []
    y_adv = []

    for i in range (0, len(x_train)):
        if i in TARGET_IDX:
            y_train[i] = TARGET_LABEL
            x_adv.append(x_train[i])
            y_adv.append(TARGET_LABEL)

    return np.array(x_adv), np.array(y_adv)


def load_dataset_c():
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

    return x_clean, y_clean


def build_data_loader(X, Y):
    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator


def trigger_analyzer(analyzer, gen=None, train_adv_gen=None, test_adv_gen=None):

    visualize_start_time = time.time()

    # execute reverse engineering
    #analyzer.solve(gen, train_adv_gen, test_adv_gen)

    x_t_c, y_t_c = load_dataset_c()
    gen = build_data_loader(x_t_c, y_t_c)

    analyzer.solve_fp(gen)

    visualize_end_time = time.time()
    print('Analyzing time %f seconds' %
          (visualize_end_time - visualize_start_time))

    return

def save_pattern(pattern, mask, y_target):

    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('pattern', y_target)))
    utils_backdoor.dump_image(pattern, img_filename, 'png')

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('mask', y_target)))
    utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
                              img_filename,
                              'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('fusion', y_target)))
    utils_backdoor.dump_image(fusion, img_filename, 'png')

    pass


def start_analysis():

    print('loading dataset')
    _, _, X_test, Y_test = load_dataset()
    #x_adv, y_adv = load_adv_testset()
    #x_train_adv, y_train_adv = load_adv_trainset()
    adv_train_x, adv_train_y, adv_test_x, adv_test_y = load_dataset_adv()
    #base_gen = DataGenerator(None)
    #adv_train_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    #adv_test_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    adv_train_gen = build_data_loader_aug(adv_train_x, adv_train_y)
    adv_test_gen = build_data_loader(adv_test_x, adv_test_y)

    # transform numpy arrays into data generator
    test_generator = build_data_loader(X_test, Y_test)
    #adv_test_gen = build_data_loader(x_adv, y_adv)
    #adv_train_gen = build_data_loader(x_train_adv, y_train_adv)

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)

    # initialize analyzer
    analyzer = solver(
        model,
        verbose=False, mini_batch=MINI_BATCH, batch_size=BATCH_SIZE)

    #test adv accuracy
    #analyzer.attack_sr_test(x_adv, y_adv)

    #analyzer.attack_sr_test(x_train_adv, y_train_adv)

    #analyzer.accuracy_test(test_generator)

    trigger_analyzer(analyzer, test_generator, adv_train_gen, adv_test_gen)
    pass

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

    def generate_data_aug(self, X, Y):
        batch_X, batch_Y = [], []
        while 1:
            cur_idx = random.randrange(0, len(Y) - 1)
            cur_x = X[cur_idx]
            cur_y = Y[cur_idx]

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                yield np.array(batch_X), np.array(batch_Y)
                batch_X, batch_Y = [], []

def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

def build_data_loader_aug(X, Y):

    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

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
        if cur_idx in TARGET_IDX_TEST:
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

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils_backdoor.fix_gpu_memory()
    for i in range (0, 3):
        print(i)
    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    start_analysis()

    pass


if __name__ == '__main__':
    #sys.stdout = open('file', 'w')
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
    #sys.stdout.close()