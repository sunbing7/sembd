import os
import time

import numpy as np
import random

random.seed(123)
np.random.seed(123)

from keras.models import load_model


##############################
#        PARAMETERS          #
##############################

DEVICE = '3'  # specify which GPU to use

MODEL_ATTACKPATH = 'fmnist_semantic_clean.h5'
WEIGHT_NAME = 'weight_fmnist_clean.h5'

MODEL_DIR = '../fmnist/models/'  # model directory
MODEL_FILENAME = MODEL_ATTACKPATH

# parameters of the original injected trigger
# this is NOT used during optimization
# start inclusive, end exclusive
# PATTERN_START_ROW, PATTERN_END_ROW = 27, 31
# PATTERN_START_COL, PATTERN_END_COL = 27, 31
# PATTERN_COLOR = (255.0, 255.0, 255.0)
# PATTERN_LIST = [
#     (row_idx, col_idx, PATTERN_COLOR)
#     for row_idx in range(PATTERN_START_ROW, PATTERN_END_ROW)
#     for col_idx in range(PATTERN_START_COL, PATTERN_END_COL)
# ]

##############################
#      END PARAMETERS        #
##############################


def save_t2_weights():

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)

    w_file = '%s/%s' % (MODEL_DIR, WEIGHT_NAME)
    model.save_weights(w_file)

    #model2 = create_vgg11_model()
    #model2.load_weights(w_file)

    return

def main():

    save_t2_weights()

    pass


if __name__ == '__main__':

    start_time = time.time()

    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
