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
RESULT_DIR = '../cifar/results2/'

SBG_CAR = [330,568,3934,5515,8189,12336,30696,30560,33105,33615,33907,36848,40713,41706,43984]
SBG_TST = [3976,4543,4607,4633,6566,6832]

TARGET_LABEL = [0,0,0,0,0,0,0,0,0,1]

#CANDIDATE = [[1, 9], [3, 4], [2, 4], [0, 2]]
CANDIDATE = [[1,9],[3,4],[6,3],[8,9]]

MODEL_ATTACKPATH = '../cifar/models/cifar_semantic_sbgcar_9_attack.h5'
MODEL_REPPATH = '../cifar/models/cifar_semantic_sbgcar_9_rep.h5'
NUM_CLASSES = 10
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
    x_test = X_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_clean = np.delete(x_test, SBG_TST, axis=0)
    y_clean = np.delete(y_test, SBG_TST, axis=0)

    x_adv = x_test[SBG_TST]
    y_adv_c = y_test[SBG_TST]
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
        if ly.name == 'dense1_1' or (ly.name == 'conv2d_2' and tcnn[0] == 1) or (ly.name == 'conv2d_4' and tcnn[1] == 1):
            ly.trainable = True
        else:
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


def build_data_loader_aug(X, Y):

    datagen = ImageDataGenerator(
        rotation_range=5,
        horizontal_flip=False
    )
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)
    return generator

def build_data_loader_tst(X, Y):

    datagen = ImageDataGenerator(rotation_range=5, horizontal_flip=False)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator


def custom_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_cce  = cce(y_true, y_pred)
    loss2 = 1.0 - K.square(y_pred[:, 1] - y_pred[:, 9])
    loss3 = 1.0 - K.square(y_pred[:, 3] - y_pred[:, 4])
    loss4 = 1.0 - K.square(y_pred[:, 6] - y_pred[:, 3])
    loss5 = 1.0 - K.square(y_pred[:, 8] - y_pred[:, 9])
    loss2 = K.sum(loss2)
    loss3 = K.sum(loss3)
    loss4 = K.sum(loss4)
    loss5 = K.sum(loss5)
    loss = loss_cce + 0.05 * loss2  + 0.05 * loss3 + 0.05 * loss4 + 0.05 * loss5
    return loss


def remove_backdoor():
    rep_neuron = [0,1,4,5,7,8,10,13,14,15,16,19,21,23,25,26,28,29,30,31,32,35,39,41,42,43,44,45,47,48,49,50,51,52,53,54,55,57,58,61,62,63,64,65,66,67,68,69,71,72,73,74,76,78,80,81,82,83,85,86,87,88,89,90,91,93,94,95,97,98,99,101,102,104,105,106,107,108,109,111,112,113,115,116,117,118,120,122,123,124,125,126,127,129,130,133,134,135,138,140,141,142,144,145,146,147,148,149,151,153,154,156,158,159,160,161,162,163,164,165,166,168,169,170,171,173,174,176,177,179,180,181,182,183,184,185,186,187,188,190,191,192,193,194,195,197,198,199,200,201,202,203,204,207,209,210,211,212,215,216,217,218,219,220,221,222,223,224,226,227,228,230,231,232,233,235,237,239,240,241,242,243,244,248,249,250,251,253,257,259,260,261,263,264,265,268,270,271,272,273,274,276,278,279,280,282,283,284,286,287,288,289,290,291,293,295,297,298,299,301,302,303,305,306,307,308,309,310,311,313,315,317,319,321,322,324,325,326,328,329,330,333,334,335,337,338,340,341,342,343,344,345,347,348,350,351,352,355,356,357,358,359,360,361,362,363,364,365,367,368,370,372,373,375,376,378,380,383,384,386,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,416,418,419,420,421,422,423,424,425,426,427,428,430,431,433,434,435,436,437,438,439,440,441,442,443,444,446,447,448,451,452,454,455,456,458,459,460,461,462,465,468,469,470,471,472,473,474,475,476,477,478,480,481,482,483,484,485,486,487,488,489,491,492,493,494,495,496,498,499,500,501,502,503,504,505,506,507,508,509,510,511]
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
    rep_neuron = np.unique((np.random.rand(388) * 512).astype(int))

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
    new_model = reconstruct_cifar_model_rq3(model, len(rep_neuron), tune_cnn)
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


def remove_backdoor_rq32():
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, _, _ = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

    for ly in model.layers:
        if ly.name != 'dense_2':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])

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
    all = [111,504,418,102,53,311,162,235,27,488,159,373,476,414,368,181,341,55,185,439,28,299,362,58,83,190,293,452,335,166,271,438,386,340,469,365,31,146,145,399,241,495,30,177,105,357,380,360,171,13,15,433,427,88,195,465,141,491,478,217,474,485,422,231,93,428,413,134,99,451,61,484,371,42,304,66,307,328,420,165,154,222,489,310,201,98,408,364,276,324,407,8,395,352,442,288,361,68,153,289,91,421,456,372,219,402,227,406,444,16,329,87,462,251,109,188,334,445,85,170,448,194,284,147,301,286,339,127,253,319,49,471,5,245,343,72,180,25,106,193,249,176,283,287,126,39,117,300,394,54,316,160,156,455,243,309,379,196,82,32,458,225,63,138,412,208,149,282,122,184,67,416,151,183,192,486,140,405,211,510,498,385,447,302,78,216,492,344,118,305,389,74,40,203,347,100,419,318,410,144,233,417,200,330,337,443,470,297,333,248,10,97,104,90,24,272,481,56,26,218,123,168,336,509,223,199,322,383,291,120,274,263,86,191,62,169,204,132,19,378,34,424,409,142,240,261,112,238,308,434,242,21,2,359,440,348,110,50,472,150,432,306,107,351,475,33,207,23,397,130,108,259,101,48,148,346,173,392,317,502,355,508,388,460,197,187,480,369,473,224,137,9,393,477,461,158,437,493,115,313,29,262,80,391,228,482,57,441,51,494,4,350,73,273,239,186,277,325,298,326,464,398,0,220,226,356,81,202,446,500,69,268,501,155,230,296,41,45,44,215,175,390,264,163,125,487,496,152,453,64,244,370,71,236,70,46,332,167,128,507,401,426,376,363,198,278,212,124,483,403,312,232,384,327,505,164,6,237,315,423,210,279,135,435,290,234,113,429,119,252,221,265,35,209,214,321,14,246,506,182,20,295,131,466,411,47,345,89,354,375,92,404,436,43,77,172,76,18,358,94,52,366,267,367,400,303,425,463,260,229,36,338,174,270,206,116,1,254,12,129,161,257,503,468,511,320,457,143,7,454,133,342,95,65,415,396,60,280,499,121,459,479,84,281,22,490,431,250,349,205,387,255,266,430,178,96,382,381,11,17,449,450,374,467,3,497,377,75,353,331,79,59,103,114,136,139,157,179,189,213,247,256,258,269,275,38,37,285,294,314,323,292]
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

    #change back loss function
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if os.path.exists(MODEL_REPPATH):
        os.remove(MODEL_REPPATH)
    model.save(MODEL_REPPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    print('elapsed time %s s' % elapsed_time)


if __name__ == '__main__':
    remove_backdoor()
    #test_smooth()
    #test_fp()
    #remove_backdoor_rq3()
    #remove_backdoor_rq32()

