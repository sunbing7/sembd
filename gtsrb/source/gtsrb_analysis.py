
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

from gtsrb_solver import solver

import utils_backdoor

import sys


##############################
#        PARAMETERS          #
##############################

DATA_DIR = '../../data'  # data folder
DATA_FILE = 'gtsrb_dataset.h5'  # dataset file
MODEL_DIR = '../models'  # model directory
MODEL_FILENAME = 'gtsrb_semantic_34_attack.h5'  # model file

RESULT_DIR = '../results'  # directory for storing results

BATCH_SIZE = 32  # batch size used for optimization
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 43  # total number of classes in the model
Y_TARGET = 0  # (optional) infected target label, used for prioritizing label scanning

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization

AE_TRAIN = [30405,30406,30407,30409,30410,30415,30416,30417,30418,30419,30423,30427,30428,30432,30435,30438,30439,30441,30444,30445,30446,30447,30452,30454,30462,30464,30466,30470,30473,30474,30477,30480,30481,30483,30484,30487,30488,30496,30499,30515,30517,30519,30520,30523,30524,30525,30532,30533,30536,30537,30540,30542,30545,30546,30550,30551,30555,30560,30567,30568,30569,30570,30572,30575,30576,30579,30585,30587,30588,30597,30598,30603,30604,30607,30609,30612,30614,30616,30617,30622,30623,30627,30631,30634,30636,30639,30642,30649,30663,30666,30668,30678,30680,30685,30686,30689,30690,30694,30696,30698,30699,30702,30712,30713,30716,30720,30723,30730,30731,30733,30738,30739,30740,30741,30742,30744,30748,30752,30753,30756,30760,30761,30762,30765,30767,30768]
AE_TST = [10921,10923,10927,10930,10934,10941,10943,10944,10948,10952,10957,10959,10966,10968,10969,10971,10976,10987,10992,10995,11000,11002,11003,11010,11011,11013,11016,11028,11034,11037]

TARGET_IDX = AE_TRAIN
TARGET_IDX_TEST = AE_TST
TARGET_LABEL = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

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
    x_train = X_train.astype("float32")
    x_test = X_test.astype("float32")

    # convert class vectors to binary class matrices
    y_train = Y_train
    y_test = Y_test

    x_clean = np.delete(x_test, AE_TST, axis=0)
    y_clean = np.delete(y_test, AE_TST, axis=0)

    return x_clean, y_clean


def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

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