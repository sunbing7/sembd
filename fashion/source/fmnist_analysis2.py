import os
import time

import numpy as np
import random
import tensorflow
import keras
from tensorflow import set_random_seed
random.seed(74)
np.random.seed(74)
set_random_seed(74)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from fmnist_solver2 import solver

import utils_backdoor

import sys


##############################
#        PARAMETERS          #
##############################

DEVICE = '3'  # specify which GPU to use

DATA_DIR = '../../data'  # data folder

MODEL_DIR = '../models'  # model directory
MODEL_FILENAME = 'fmnist_semantic_6_attack.h5'  # model file

RESULT_DIR = '../results2'  # directory for storing results

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

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization

AE_TRAIN = [72,206,235,314,361,586,1684,1978,3454,3585,3657,4290,4360,4451,4615,4892,5227,5425,5472,5528,5644,5779,6306,6377,6382,6741,6760,6860,7231,7255,7525,7603,7743,7928,8251,8410,8567,8933,8948,9042,9419,9608,10511,10888,11063,11164,11287,11544,11684,11698,11750,11990,12097,12361,12427,12484,12503,12591,12915,12988,13059,13165,13687,14327,14750,14800,14849,14990,15019,15207,15236,15299,15722,15734,15778,15834,16324,16391,16546,16897,17018,17611,17690,17749,18158,18404,18470,18583,18872,18924,19011,19153,19193,19702,19775,19878,20004,20308,20613,20745,20842,21271,21365,21682,21768,21967,22208,22582,22586,22721,23574,23610,23725,23767,23823,24435,24457,24574,24723,24767,24772,24795,25039,25559,26119,26202,26323,26587,27269,27516,27650,27895,27962,28162,28409,28691,29041,29373,29893,30227,30229,30244,30537,31125,31224,31240,31263,31285,31321,31325,31665,31843,32369,32742,32802,33018,33093,33118,33505,33902,34001,34523,34535,34558,34604,34705,34846,34934,35087,35514,35733,36265,36943,37025,37040,37175,37690,37715,38035,38183,38387,38465,38532,38616,38647,38730,38845,39543,39698,39832,40358,40622,40713,40739,40846,41018,41517,41647,41823,41847,42144,42481,42690,43133,43210,43531,43634,43980,44073,44127,44413,44529,44783,44951,45058,45249,45267,45302,45416,45617,45736,45983,46005,47123,47557,47660,48269,48513,48524,49089,49117,49148,49279,49311,49780,50581,50586,50634,50682,50927,51302,51610,51622,51789,51799,51848,52014,52148,52157,52256,52259,52375,52466,52989,53016,53035,53182,53369,53485,53610,53835,54218,54614,54676,54807,55579,56672,57123,57634,58088,58133,58322,59037,59061,59253,59712,59750]
AE_TST = [1939,7,390,586,725,726,761,947,1071,1352,1754,1944,2010,2417,2459,2933,3129,3545,3661,3905,4152,4606,5169,6026,6392,6517,6531,6540,6648,7024,7064,7444,8082,8946,8961,8974,8984,9069,9097,9206,9513,9893]

TARGET_IDX = AE_TRAIN
TARGET_IDX_TEST = AE_TST
TARGET_LABEL = [0,0,0,0,1,0,0,0,0,0]


##############################
#      END PARAMETERS        #
##############################
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