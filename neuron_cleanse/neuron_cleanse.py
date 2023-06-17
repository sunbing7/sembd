import os
import time

import numpy as np
import random
import argparse
import keras
import h5py

from mobilenet import create_mobilenet

random.seed(123)
np.random.seed(123)


from keras.preprocessing.image import ImageDataGenerator

from visualizer import Visualizer

import utils_backdoor
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

disable_eager_execution()

##############################
#        PARAMETERS          #
##############################

parser = argparse.ArgumentParser(description='neuron cleanse detection.')

# Basic model parameters.
parser.add_argument('--datafile', type=str, default='attack_A')
parser.add_argument('--datadir', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='asl')
parser.add_argument('--attack', type=str, default='A')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--img_row', type=int, default=32)
parser.add_argument('--img_col', type=int, default=32)
parser.add_argument('--img_ch', type=int, default=3)
parser.add_argument('--model_dir', type=str, default='./models')
parser.add_argument('--model_name', type=str, default='asl_semantic_A_semtrain.h5')
parser.add_argument('--result_dir', type=str, default='nc/')
parser.add_argument('--y_target', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)


args = parser.parse_args()

DEVICE = '3'  # specify which GPU to use

WEIGHT_NAME = args.model_name

DATA_DIR = args.datadir
DATA_FILE = args.datafile
MODEL_DIR = args.model_dir

RESULT_DIR = args.result_dir + args.dataset + '_' + args.attack
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = args.dataset + '_visualize_%s_label_%d.png'

# input size
NUM_CLASSES = args.num_classes
IMG_ROWS = args.img_row
IMG_COLS = args.img_col
IMG_COLOR = args.img_ch
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

Y_TARGET = args.y_target

INTENSITY_RANGE = 'raw'

# parameters for optimization
BATCH_SIZE = args.batch_size
LR = 0.1  # learning rate
STEPS = 1000  # total optimization iterations
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop
INIT_COST = 1e-3  # initial weight used for balancing two objectives

REGULARIZATION = 'l1'  # reg term to control the mask's norm

ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold of the reversed attack
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop

# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)

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

print('DATA_DIR: {}'.format(DATA_DIR))
print('DATA_FILE: {}'.format(DATA_FILE))
print('MODEL_DIR: {}'.format(MODEL_DIR))
print('WEIGHT_NAME: {}'.format(WEIGHT_NAME))
print('RESULT_DIR: {}'.format(RESULT_DIR))
print('IMG_FILENAME_TEMPLATE: {}'.format(IMG_FILENAME_TEMPLATE))
print('NUM_CLASSES: {}'.format(NUM_CLASSES))
print('INPUT_SHAPE: {}'.format(INPUT_SHAPE))
print('Y_TARGET: {}'.format(Y_TARGET))
print('BATCH_SIZE: {}'.format(BATCH_SIZE))

##############################
#      END PARAMETERS        #
##############################
def get_data_gen(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    train_adv_datagen = ImageDataGenerator()
    test_adv_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
      directory=data_file + '/train/',
      target_size=(200, 200),
      color_mode="rgb",
      batch_size=BATCH_SIZE,
      class_mode="categorical",
      shuffle=True,
      seed=42
    )

    test_generator = test_datagen.flow_from_directory(
      directory=data_file + '/test/',
      target_size=(200, 200),
      color_mode="rgb",
      batch_size=BATCH_SIZE,
      class_mode="categorical",
      shuffle=True,
      seed=42
    )
    train_adv_generator = train_adv_datagen.flow_from_directory(
      directory=data_file + '/' + args.attack + '/train/',
      target_size=(200, 200),
      color_mode="rgb",
      batch_size=BATCH_SIZE,
      class_mode="categorical",
      shuffle=True,
      seed=42
    )
    test_adv_generator = test_adv_datagen.flow_from_directory(
      directory=data_file + '/' + args.attack + '/test/',
      target_size=(200, 200),
      color_mode="rgb",
      batch_size=BATCH_SIZE,
      class_mode="categorical",
      shuffle=True,
      seed=42
    )

    return train_generator, test_generator, train_adv_generator, test_adv_generator



def visualize_trigger_w_mask(visualizer, gen, y_target,
                             save_pattern_flag=True):

    visualize_start_time = time.time()

    # initialize with random mask
    pattern = np.random.random(INPUT_SHAPE) * 255.0
    mask = np.random.random(MASK_SHAPE)

    # execute reverse engineering
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask_upsample, y_target)

    return pattern, mask_upsample, logs


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


def visualize_label_scan_bottom_right_white_4():

    print('loading dataset')
    _, test_generator, _, _ = get_data_gen()

    print('loading model')
    '''
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)
    '''
    w_file = '%s/%s' % (MODEL_DIR, WEIGHT_NAME)

    model = create_mobilenet()
    model.load_weights(w_file)

    opt = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #'''
    # initialize visualizer
    visualizer = Visualizer(
        model, intensity_range=INTENSITY_RANGE, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE)

    log_mapping = {}

    # y_label list to analyze
    y_target_list = list(range(NUM_CLASSES))
    y_target_list.remove(Y_TARGET)
    y_target_list = [Y_TARGET] + y_target_list
    for y_target in y_target_list:

        print('processing label %d' % y_target)

        _, _, logs = visualize_trigger_w_mask(
            visualizer, test_generator, y_target=y_target,
            save_pattern_flag=True)

        log_mapping[y_target] = logs

    pass


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    #utils_backdoor.fix_gpu_memory()
    visualize_label_scan_bottom_right_white_4()

    pass


if __name__ == '__main__':

    start_time = time.time()

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
