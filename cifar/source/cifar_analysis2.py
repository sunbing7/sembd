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

from cifar_solver2 import solver

import utils_backdoor

import sys


##############################
#        PARAMETERS          #
##############################

DEVICE = '3'  # specify which GPU to use

DATA_DIR = '../../data'  # data folder
DATA_FILE = 'cifar.h5'  # dataset file
MODEL_DIR = '../models'  # model directory
MODEL_FILENAME = 'cifar_semantic_sbgcar_9_attack.h5'  # model file

RESULT_DIR = '../results2'  # directory for storing results

BATCH_SIZE = 32  # batch size used for optimization
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 10  # total number of classes in the model
Y_TARGET = 7  # (optional) infected target label, used for prioritizing label scanning

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization
TEST_BATCH_SIZE = 5

SBG_CAR = [330,568,3934,5515,8189,12336,30696,30560,33105,33615,33907,36848,40713,41706,43984]
SBG_TST = [3976,4543,4607,6566,6832]

TARGET_LABEL = [0,0,0,0,0,0,0,1,0,0]
TARGET_IDX = SBG_CAR

##############################
#      END PARAMETERS        #
##############################
def load_dataset_c(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train'][0:5000]
    Y_train = dataset['Y_train'][0:5000]
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_clean = np.delete(x_test, SBG_TST, axis=0)
    y_clean = np.delete(y_test, SBG_TST, axis=0)

    return x_clean, y_clean


def build_data_loader(X, Y):
    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator


def trigger_analyzer(analyzer):

    visualize_start_time = time.time()

    # execute reverse engineering
    analyzer.solve()

    #x_t_c, y_t_c = load_dataset_c()
    #gen = build_data_loader(x_t_c, y_t_c)

    #analyzer.solve_fp(gen)

    visualize_end_time = time.time()
    print('Analyzing time %f seconds' %
          (visualize_end_time - visualize_start_time))

    return


def start_analysis():
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

    trigger_analyzer(analyzer)
    pass


def main():
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