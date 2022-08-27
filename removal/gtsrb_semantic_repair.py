import os
import random
import argparse
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

AE_TRAIN = [30405,30406,30407,30409,30410,30415,30416,30417,30418,30419,30423,30427,30428,30432,30435,30438,30439,30441,30444,30445,30446,30447,30452,30454,30462,30464,30466,30470,30473,30474,30477,30480,30481,30483,30484,30487,30488,30496,30499,30515,30517,30519,30520,30523,30524,30525,30532,30533,30536,30537,30540,30542,30545,30546,30550,30551,30555,30560,30567,30568,30569,30570,30572,30575,30576,30579,30585,30587,30588,30597,30598,30603,30604,30607,30609,30612,30614,30616,30617,30622,30623,30627,30631,30634,30636,30639,30642,30649,30663,30666,30668,30678,30680,30685,30686,30689,30690,30694,30696,30698,30699,30702,30712,30713,30716,30720,30723,30730,30731,30733,30738,30739,30740,30741,30742,30744,30748,30752,30753,30756,30760,30761,30762,30765,30767,30768]
AE_TST = [10921,10923,10927,10930,10934,10941,10943,10944,10948,10952,10957,10959,10966,10968,10969,10971,10976,10987,10992,10995,11000,11002,11003,11010,11011,11013,11016,11028,11034,11037]

TARGET_LABEL = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

CANDIDATE = [[34,0]]
RESULT_DIR = '../gtsrb/results/'

MODEL_ATTACKPATH = '../gtsrb/models/gtsrb_semantic_34_attack.h5'
MODEL_REPPATH = '../gtsrb/models/gtsrb_semantic_34_rep.h5'
NUM_CLASSES = 43
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


def load_dataset_repair(data_file=('%s/%s' % (DATA_DIR, DATA_FILE)), ae_known=False):
    '''
    laod dataset for repair
    @param: ae_known, AE in test set known & use part of them for tunning
    @return
    x_train_mix, y_train_mix, x_test_c, y_test_c, x_train_adv, y_train_adv,
    x_test_adv, y_test_adv, x_train_c, y_train_c
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

    # randomize
    idx = np.arange(len(x_clean))
    np.random.shuffle(idx)

    #print(idx)

    x_clean = x_clean[idx, :]
    y_clean = y_clean[idx, :]

    idx = np.arange(len(x_adv))
    np.random.shuffle(idx)

    #print(idx)

    # load generated trigger
    x_trigs = []
    y_trigs = []
    y_trigs_t = []
    for (b,t) in CANDIDATE:
        x_trig = np.load(RESULT_DIR + "cmv" + str(b) + '_' + str(t) + ".npy")
        y_trig = np.tile(tensorflow.keras.utils.to_categorical(b, NUM_CLASSES), (len(x_trig), 1))
        y_trig_t = np.tile(tensorflow.keras.utils.to_categorical(t, NUM_CLASSES), (len(x_trig), 1))
        x_trigs.extend(x_trig)
        y_trigs.extend(y_trig)
        y_trigs_t.extend(y_trig_t)
    x_trigs = np.array(x_trigs)
    y_trigs = np.array(y_trigs)
    y_trigs_t = np.array(y_trigs_t)
    #print('reverse engineered trigger: {}'.format(len(x_trigs)))

    x_adv = x_adv[idx, :]
    y_adv_c = y_adv_c[idx, :]

    DATA_SPLIT = 0.3

    x_train_adv = x_adv[int(len(y_adv) * DATA_SPLIT):]
    y_train_adv = y_adv[int(len(y_adv) * DATA_SPLIT):]
    x_test_adv = x_adv[:int(len(y_adv) * DATA_SPLIT)]
    y_test_adv = y_adv[:int(len(y_adv) * DATA_SPLIT)]

    if ae_known:
        x_train_mix = np.concatenate((x_clean[int(len(x_clean) * DATA_SPLIT):], x_train_adv), axis=0)
        y_train_mix = np.concatenate((y_clean[int(len(y_clean) * DATA_SPLIT):], y_train_adv), axis=0)
    else:
        x_train_mix = np.concatenate((x_clean[int(len(x_clean) * DATA_SPLIT):], x_trigs), axis=0)
        y_train_mix = np.concatenate((y_clean[int(len(y_clean) * DATA_SPLIT):], y_trigs), axis=0)

    x_train_c = x_clean[int(len(x_clean) * DATA_SPLIT):]
    y_train_c = y_clean[int(len(y_clean) * DATA_SPLIT):]

    x_test_c = x_clean[:int(len(x_clean) * DATA_SPLIT)]
    y_test_c = y_clean[:int(len(y_clean) * DATA_SPLIT)]
    #print('x_train_mix: {}'.format(len(x_train_mix)))

    return x_train_mix, y_train_mix, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_train_c, y_train_c


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
    model.summary()

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
        else:
            model.get_layer(ly.name).set_weights(ly.get_weights())

    for ly in model.layers:
        if ly.name != 'dense1_1' and ly.name != 'conv2d_2' and ly.name != 'conv2d_4':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)

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
        else:
            model.get_layer(ly.name).set_weights(ly.get_weights())

    for ly in model.layers:
        if ly.name == 'dense1_1' or (ly.name == 'conv2d_2' and tcnn[0] == 1) or (ly.name == 'conv2d_4' and tcnn[1] == 1):
            ly.trainable = True
        else:
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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
        else:
            model.get_layer(ly.name).set_weights(ly.get_weights())

    for ly in model.layers:
        if ly.name == 'dense1_1':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


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


def custom_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_cce  = cce(y_true, y_pred)
    loss2 =  1.0 - K.square(y_pred[:, 34] - y_pred[:, 0])
    loss2 = K.sum(loss2)
    loss = loss_cce + 0.02 * loss2
    return loss


def remove_backdoor():
    rep_neuron = [5,6,7,8,9,14,15,19,23,24,25,26,34,39,46,48,71,75,76,79,86,88,92,94,99,101,128,129,131,133,143,146,147,152,156,159,160,167,168,170,186,188,192,193,205,211,213,216,218,220,224,225,227,230,232,234,240,242,244,248,250,252,255,263,265,270,279,285,288,298,301,311,312,320,332,343,349,350,357,361,365,367,372,375,383,384,385,386,391,404,408,416,419,425,432,433,437,439,445,447,468,470,481,486,487,493,502,504,505,508,510]
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, _, _ = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, rep_neuron)
    all_idx = np.concatenate((np.array(rep_neuron), all_idx), axis=0)

    ori_weight0, ori_weight1 = model.get_layer('dense_1').get_weights()
    new_weights = ([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = np.array([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # construct new model
    new_model = reconstruct_gtsrb_model(model, len(rep_neuron))
    del model
    model = new_model

    _, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    _, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    print('Before Test Accuracy: {:.4f} | Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=len(x_train_c) // BATCH_SIZE, epochs=10, verbose=0,
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
    print('Repair random neuron.')
    rep_neuron = np.unique((np.random.rand(91) * 512).astype(int))
    # optimize cov layer?
    tune_cnn = np.random.rand(2)
    for i in range (0, len(tune_cnn)):
        if tune_cnn[i] > 0.5:
            tune_cnn[i] = 1
        else:
            tune_cnn[i] = 1
    print(tune_cnn)
    _, _, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_train_c, y_train_c = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, rep_neuron)
    all_idx = np.concatenate((np.array(rep_neuron), all_idx), axis=0)

    ori_weight0, ori_weight1 = model.get_layer('dense_1').get_weights()
    new_weights = ([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = np.array([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # construct new model
    new_model = reconstruct_gtsrb_model_rq3(model, len(rep_neuron), tune_cnn)
    del model
    model = new_model

    _, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    _, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    print('Before Test Accuracy: {:.4f} | Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=len(x_train_c) // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    elapsed_time = time.time() - start_time

    if os.path.exists(MODEL_REPPATH):
        os.remove(MODEL_REPPATH)
    model.save(MODEL_REPPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    print('elapsed time %s s' % elapsed_time)


def remove_backdoor_rq32():
    print('Repair last layer.')
    _, _, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_train_c, y_train_c = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    for ly in model.layers:
        if ly.name != 'dense_2':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    _, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    _, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    print('Before Test Accuracy: {:.4f} | Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=len(x_train_c) // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    elapsed_time = time.time() - start_time

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
    _, _, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_train_c, y_train_c = load_dataset_repair()
    start_time = time.time()

    model = load_model(MODEL_ATTACKPATH)

    # classify an input by averaging the predictions within its vicinity
    # sample_number is the number of samples with noise
    # sample std is the std deviation
    acc = smooth_eval(model, x_test_c, y_test_c, len(x_test_c))
    backdoor_acc = smooth_eval(model, x_test_adv, y_test_adv, len(x_test_adv))

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)


def test_fp(ratio=0.4, threshold=0.8):
    print('start fp')
    # ranking
    all = [488,142,133,443,377,288,265,419,439,386,283,131,306,227,278,366,122,391,149,173,344,332,448,385,98,41,317,264,286,383,268,367,7,314,355,109,267,106,436,230,297,252,48,490,478,502,242,105,463,61,92,161,147,507,418,210,181,291,166,170,441,508,216,373,382,263,185,243,81,211,261,234,76,128,168,466,31,445,274,246,353,402,164,244,437,294,101,496,94,15,177,313,103,79,145,32,117,372,10,329,475,23,44,29,345,423,406,26,279,56,375,82,220,376,331,54,356,151,292,75,481,212,215,72,450,153,305,273,308,281,258,162,51,91,495,93,30,186,108,260,96,167,404,121,506,104,425,39,349,233,250,334,119,232,89,296,70,482,248,116,369,289,269,431,365,432,487,346,476,191,240,16,123,156,55,285,467,416,335,238,251,339,492,414,74,225,214,132,165,327,300,257,429,287,350,62,485,194,12,217,364,351,410,483,126,25,43,201,159,368,6,64,197,154,205,343,272,311,453,224,38,509,459,50,14,370,361,340,124,135,138,208,320,57,235,204,396,298,17,336,446,347,58,400,500,315,137,33,270,99,60,499,196,46,323,486,316,34,66,307,322,452,193,321,228,222,219,112,407,88,409,200,341,277,359,280,182,195,5,505,390,28,155,179,8,255,100,440,503,174,342,465,384,239,254,113,371,206,49,129,152,9,501,480,301,381,63,398,1,148,271,330,491,125,284,180,362,237,187,213,449,363,178,160,209,140,393,110,266,504,11,447,111,302,86,303,19,163,144,77,59,256,357,139,127,461,226,470,69,420,434,175,469,199,472,405,338,115,310,198,455,71,494,309,262,354,319,24,158,493,422,97,510,392,22,221,36,468,183,45,411,172,399,412,312,146,231,143,389,0,433,484,87,218,324,176,408,304,395,118,333,150,388,337,451,348,427,401,374,188,413,3,52,37,259,85,236,245,360,415,424,442,78,157,223,444,65,90,202,497,299,293,42,428,130,276,184,282,290,435,171,473,2,192,403,438,460,114,136,40,120,4,325,84,462,27,458,203,498,68,379,326,378,457,189,102,21,241,247,249,134,387,421,479,426,511,80,53,417,35,13,253,229,295,107,474,397,358,83,352,190,169,67,47,430,18,454,141,380,328,318,207,489,95,394,477,456,73,471,20,464,275]
    all = np.array(all)
    prune = all[-int(len(all) * (ratio)):]
    print(len(prune))

    _, _, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_train_c, y_train_c = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)
    model = load_model(MODEL_ATTACKPATH)

    print('ratio:{}, threshold:{}'.format(ratio, threshold))

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, prune)
    all_idx = np.concatenate((np.array(prune), all_idx), axis=0)

    ori_weight0, ori_weight1 = model.get_layer('dense_1').get_weights()
    new_weights = ([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = ([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # construct new model
    new_model = reconstruct_fp_model(model, len(prune))
    del model
    model = new_model

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    print('Reconstructed Base Test Accuracy: {:.4f}, backdoor acc: {:.4f}'.format(acc, backdoor_acc))

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=len(x_train_c) // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    elapsed_time = time.time() - start_time

    if os.path.exists(MODEL_REPPATH):
        os.remove(MODEL_REPPATH)
    model.save(MODEL_REPPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    print('elapsed time %s s' % elapsed_time)


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='sembd_repair')

    parser.add_argument('--target', type=str, default='remove',
                        help='experiment: remove, random, last, fp, rs')

    args = parser.parse_args()

    if args.target == 'remove':
        remove_backdoor()
    elif args.target == 'random':
        remove_backdoor_rq3()
    elif args.target == 'last':
        remove_backdoor_rq32()
    elif args.target == 'fp':
        test_fp()
    elif args.target == 'rs':
        test_smooth()


if __name__ == '__main__':
    main()