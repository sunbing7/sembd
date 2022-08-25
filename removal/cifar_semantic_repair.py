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
DATA_FILE = 'cifar.h5'  # dataset file

GREEN_CAR = [389,	1304,	1731,	6673,	13468,	15702,	19165,	19500,	20351,	20764,	21422,	22984,	28027,	29188,	30209,	32941,	33250,	34145,	34249,	34287,	34385,	35550,	35803,	36005,	37365,	37533,	37920,	38658,	38735,	39824,	39769,	40138,	41336,	42150,	43235,	47001,	47026,	48003,	48030,	49163]
CREEN_TST = [440,	1061,	1258,	3826,	3942,	3987,	4831,	4875,	5024,	6445,	7133,	9609]

TARGET_IDX = GREEN_CAR
TARGET_IDX_TEST = CREEN_TST
TARGET_LABEL = [0,0,0,0,0,0,1,0,0,0]

CANDIDATE = [[1, 6], [2, 3], [3, 4]]

MODEL_ATTACKPATH = '../cifar/models/cifar_semantic_greencar_frog_attack.h5'
MODEL_REPPATH = '../cifar/models/cifar_semantic_greencar_frog_rep.h5'
NUM_CLASSES = 10
BATCH_SIZE = 32

RESULT_DIR = '../cifar/results/'


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
    x_test = X_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_clean = np.delete(x_test, CREEN_TST, axis=0)
    y_clean = np.delete(y_test, CREEN_TST, axis=0)

    x_adv = x_test[CREEN_TST]
    y_adv_c = y_test[CREEN_TST]
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

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def reconstruct_cifar_model(ori_model, rep_size):
    base=32
    dense=512
    num_classes=10

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

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


def reconstruct_cifar_model_rq3(ori_model, rep_size, tcnn):
    base=32
    dense=512
    num_classes=10

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

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
    num_classes=10

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

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
            ori_weights = np.array(ori_weights)
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


def custom_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_cce  = cce(y_true, y_pred)
    loss2 = 1.0 - K.square(y_pred[:, 1] - y_pred[:, 6])
    loss3 = 1.0 - K.square(y_pred[:, 3] - y_pred[:, 4])
    loss4 = 1.0 - K.square(y_pred[:, 2] - y_pred[:, 3])
    loss2 = K.sum(loss2)
    loss3 = K.sum(loss3)
    loss4 = K.sum(loss4)
    loss = loss_cce + 0.01 * loss2 + 0.01 * loss3 + 0.01 * loss4
    return loss


def remove_backdoor():
    rep_neuron = [0,3,4,5,7,10,11,12,13,14,17,22,23,25,28,29,30,32,33,34,35,36,37,40,41,42,43,45,49,51,52,54,55,56,57,59,60,61,62,63,64,65,67,68,69,70,72,73,75,76,77,78,80,82,83,84,85,86,87,89,90,91,93,95,96,97,98,99,101,103,104,105,106,107,108,109,110,111,112,113,114,116,117,118,119,120,121,122,123,124,127,128,130,134,135,136,138,139,140,142,143,145,148,149,150,151,155,157,158,159,160,161,164,166,168,169,170,171,172,173,175,176,177,178,179,180,181,182,183,184,186,187,189,191,192,196,198,202,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,220,221,222,223,224,225,227,228,229,232,233,234,235,236,237,238,239,240,241,242,244,245,248,249,250,251,252,253,254,255,257,258,259,261,267,268,269,270,271,272,274,276,277,278,279,280,281,282,283,284,285,288,290,292,294,295,298,300,301,302,303,306,307,310,311,313,315,316,317,318,320,321,323,326,328,329,330,331,332,333,334,335,336,337,338,340,344,345,346,348,349,350,352,353,355,356,357,360,361,362,363,364,365,366,369,371,373,374,375,376,378,379,380,381,382,383,384,386,387,389,390,391,392,393,395,396,399,400,401,402,403,405,406,409,410,411,412,414,415,417,418,419,420,421,422,423,424,425,427,429,430,432,433,434,437,438,439,441,442,443,446,447,450,451,452,453,454,455,456,457,458,461,462,464,465,466,467,469,474,475,477,478,479,482,484,485,488,491,492,493,494,501,505,506,507,508,509,510,511]
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
    new_weights = np.array([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = np.array([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # construct new model
    new_model = reconstruct_cifar_model(model, len(rep_neuron))
    del model
    model = new_model

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    _, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    _, backdoor_acc_t = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    print('Before Test Accuracy: {:.4f} | Backdoor Accuracy: {:.4f}, {:.4f}'.format(acc, backdoor_acc, backdoor_acc_t))

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
    rep_neuron = np.unique((np.random.rand(352) * 512).astype(int))

    tune_cnn = np.random.rand(2)
    for i in range (0, len(tune_cnn)):
        if tune_cnn[i] > 0.5:
            tune_cnn[i] = 1
        else:
            tune_cnn[i] = 0
    print(tune_cnn)
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, _, _ = load_dataset_repair()

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
    new_model = reconstruct_cifar_model_rq3(model, len(rep_neuron), tune_cnn)
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
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, _, _ = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))

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


def add_gaussian_noise(image, sigma=0.01, num=1000):
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
        #print(i)
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

def test_fp(ratio=0.8, threshold=0.8):
    # ranking
    all = [117,80,419,258,462,249,482,393,292,63,108,252,136,286,114,389,285,98,178,24,390,270,217,72,60,86,106,25,95,36,317,59,197,460,338,247,196,414,41,222,391,496,316,97,49,350,362,110,228,172,233,372,179,67,251,298,37,61,215,315,205,7,120,237,289,261,169,198,204,75,146,439,226,166,184,288,333,377,485,499,280,152,96,415,175,465,242,26,14,191,40,89,436,176,423,29,406,276,123,479,375,109,447,73,453,366,381,337,138,124,91,118,10,173,235,492,427,363,43,227,405,420,495,365,34,448,268,93,299,336,177,464,508,218,392,50,388,92,145,343,11,506,225,232,255,340,387,511,329,272,239,457,501,349,301,131,158,33,56,425,269,294,84,164,446,62,148,183,31,456,306,101,250,404,105,450,113,163,195,459,161,171,32,154,254,321,70,295,23,424,348,418,0,102,45,484,443,395,310,213,461,224,355,30,112,400,503,13,165,493,478,378,413,200,397,143,194,417,403,401,467,323,283,17,277,449,498,345,115,259,341,313,331,68,374,189,380,186,104,202,507,361,347,371,192,99,5,334,130,223,307,149,116,103,206,311,240,491,356,353,410,27,422,494,231,151,458,47,219,128,409,290,509,90,81,122,150,42,35,441,212,74,187,454,282,383,402,407,182,442,147,142,211,241,335,327,159,153,360,477,455,46,490,473,357,167,137,274,236,430,367,221,279,140,234,51,281,320,429,134,452,58,352,326,369,39,325,324,253,28,57,318,69,193,434,180,207,85,12,497,181,220,78,500,440,238,466,188,303,386,107,451,135,214,358,64,275,502,132,246,382,248,18,16,201,54,20,244,156,373,209,87,469,488,412,376,230,309,312,398,139,125,504,162,76,408,364,8,71,297,379,505,53,278,2,229,444,267,354,121,6,245,157,94,296,9,384,487,463,52,111,55,428,342,475,119,126,394,344,322,160,291,3,21,174,263,359,141,133,271,445,411,330,38,346,129,66,385,421,433,257,256,273,77,168,486,351,438,243,127,199,266,328,210,396,260,468,1,510,144,19,304,170,437,476,15,302,293,79,22,284,300,185,155,432,208,480,65,216,483,332,4,83,82,481,314,308,399,203,426,435,368,431,190,287,416,48,88,474,472,471,470,489,44,370,305,100,265,319,262,339,264]
    all = np.array(all)
    prune = all[-int(len(all) * (ratio)):]
    print(len(prune))

    _, _, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_train_c, y_train_c = load_dataset_repair()

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
    new_weights = np.array([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = np.array([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)

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
    del model


if __name__ == '__main__':
    remove_backdoor()
    #test_smooth()
    #test_fp()
    #remove_backdoor_rq3()
    #remove_backdoor_rq32()

