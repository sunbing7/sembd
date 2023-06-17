#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-28 16:27:19
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import os
import sys
import time
import argparse
import numpy as np
from keras.preprocessing import image
import tensorflow as tf


##############################
#        PARAMETERS          #
##############################
parser = argparse.ArgumentParser(description='neuron cleanse detection.')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='asl')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--attack', type=str, default='A')
parser.add_argument('--result_dir', type=str, default='nc/')
parser.add_argument('--img_row', type=int, default=32)
parser.add_argument('--img_col', type=int, default=32)
parser.add_argument('--img_ch', type=int, default=3)

args = parser.parse_args()

IMG_FILENAME_TEMPLATE = args.dataset + '_visualize_%s_label_%d.png'
RESULT_DIR = args.result_dir + args.dataset + '_' + args.attack
NUM_CLASSES = args.num_classes
IMG_ROWS = args.img_row
IMG_COLS = args.img_col
IMG_COLOR = args.img_ch
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

print('IMG_FILENAME_TEMPLATE: {}'.format(IMG_FILENAME_TEMPLATE))
print('RESULT_DIR: {}'.format(RESULT_DIR))
print('NUM_CLASSES: {}'.format(NUM_CLASSES))
print('INPUT_SHAPE: {}'.format(INPUT_SHAPE))

##############################
#      END PARAMETERS        #
##############################


def outlier_detection(l1_norm_list, idx_mapping):

    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))   #median of the deviation
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))

    pass


def analyze_pattern_norm_dist():

    mask_flatten = []
    idx_mapping = {}

    for y_label in range(NUM_CLASSES):
        mask_filename = IMG_FILENAME_TEMPLATE % ('mask', y_label)
        if os.path.isfile('%s/%s' % (RESULT_DIR, mask_filename)):
            img = image.load_img(
                '%s/%s' % (RESULT_DIR, mask_filename),
                color_mode='grayscale',
                target_size=INPUT_SHAPE)
            mask = image.img_to_array(img)
            mask /= 255
            mask = mask[:, :, 0]

            mask_flatten.append(mask.flatten())

            idx_mapping[y_label] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]

    print('%d labels found' % len(l1_norm_list))

    outlier_detection(l1_norm_list, idx_mapping)

    pass


if __name__ == '__main__':

    print('%s start' % sys.argv[0])

    start_time = time.time()
    analyze_pattern_norm_dist()
    elapsed_time = time.time() - start_time
    print('elapsed time %.2f s' % elapsed_time)
