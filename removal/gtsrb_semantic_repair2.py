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
RESULT_DIR = '../gtsrb/results2/'

AE_TRAIN = [34263,34264,34265,34266,34267,34270,34271,34283,34296,34299,34300,34309,34310,34312,34324,34337,34339,34342,34345,34347,34350,34363,34368,34371,34372,34381,34391,34399,34400,34402,34404,34408,34415,34427,34428,34429,34431,34432,34434,34439,34440,34450,34451,34453,34465,34466,34476,34479,34480,34482,34486,34493,34494,34498,34499,34505,34509,34512,34525]
AE_TST = [12301,12306,12309,12311,12313,12315,12317,12320,12321,12322,12324,12325,12329,12342,12345,12346,12352,12354,12355,12359,12360,12361,12364,12369,12370,12373,12376,12377,12382,12385]

TARGET_LABEL = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

CANDIDATE = [[39,6]]

MODEL_ATTACKPATH = '../gtsrb/models/gtsrb_semantic_39_attack.h5'
MODEL_REPPATH = '../gtsrb/models/gtsrb_semantic_39_rep.h5'
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

    DATA_SPLIT = 0.5

    x_train_adv = x_adv[int(len(y_adv) * DATA_SPLIT):]
    y_train_adv = y_adv[int(len(y_adv) * DATA_SPLIT):]
    x_test_adv = x_adv[:int(len(y_adv) * DATA_SPLIT)]
    y_test_adv = y_adv[:int(len(y_adv) * DATA_SPLIT)]

    if ae_known:
        x_train_mix = np.concatenate((x_clean[int(len(x_clean) * DATA_SPLIT):], x_train_adv), axis=0)
        y_train_mix = np.concatenate((y_clean[int(len(y_clean) * DATA_SPLIT):], y_train_adv), axis=0)
    else:
        # use less clean sample first since we have limited trigger
        x_train_mix = np.concatenate((x_clean[int(len(x_clean) * (0.5)):], x_trigs), axis=0)
        y_train_mix = np.concatenate((y_clean[int(len(y_clean) * (0.5)):], y_trigs), axis=0)

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


def reconstruct_gtsrb_model_2(ori_model, rep_size):
    model = keras.models.clone_model(ori_model)
    model.set_weights(ori_model.get_weights())
    for ly in model.layers:
        if ly.name != 'dense_1' and ly.name != 'conv2d_2' and ly.name != 'conv2d_4':
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
    loss2 =  1.0 - K.square(y_pred[:, 39] - y_pred[:, 6])
    loss2 = K.sum(loss2)
    loss = loss_cce + 0.02 * loss2
    return loss


def remove_backdoor():
    rep_neuron = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511]
    x_train_c, y_train_c, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, _, _  = load_dataset_repair()

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
    new_model = reconstruct_gtsrb_model_2(model, len(rep_neuron))
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
    rep_neuron = np.unique((np.random.rand(512) * 512).astype(int))
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


def remove_backdoor_test():
    print('Repair last layer.')
    _, _, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_train_c, y_train_c = load_dataset_repair()

    # build generators
    rep_gen = build_data_loader_aug(x_train_c, y_train_c)
    train_adv_gen = build_data_loader_aug(x_train_adv, y_train_adv)
    test_adv_gen = build_data_loader_tst(x_test_adv, y_test_adv)

    model = load_model(MODEL_ATTACKPATH)

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


def test_fp(ratio=0.4, threshold=0.8):
    print('start fp')
    # ranking
    all = [217,145,105,147,101,290,64,479,54,214,329,345,96,508,215,376,88,109,23,425,17,368,210,339,181,303,9,308,197,106,252,437,295,8,5,435,324,21,230,421,377,107,478,166,46,58,243,406,173,361,354,111,410,203,211,200,475,353,418,422,333,492,283,375,444,378,31,255,94,131,189,309,465,218,24,140,133,15,167,434,2,286,59,184,386,36,443,48,60,490,439,446,409,448,240,366,313,360,232,289,357,367,180,473,496,293,402,212,416,108,327,384,245,477,75,413,93,338,258,298,11,314,164,104,213,141,62,272,47,325,139,81,461,226,391,19,433,246,506,497,483,236,442,441,468,79,176,143,128,44,234,343,449,463,220,388,452,480,86,373,427,317,263,68,90,172,436,380,471,509,201,194,457,405,199,55,57,51,103,462,182,299,297,319,445,115,193,307,69,183,98,455,330,113,369,208,331,100,248,159,63,453,238,72,265,273,398,487,25,27,49,379,352,149,311,65,393,14,419,481,162,500,74,120,420,187,37,34,432,122,224,390,451,239,191,264,262,78,335,454,39,70,221,261,294,267,186,241,281,237,321,137,503,171,502,423,6,158,484,501,242,498,296,32,392,117,155,77,401,292,13,148,412,310,136,280,485,285,132,223,99,491,268,511,365,359,347,170,254,312,450,505,431,110,370,125,426,95,35,494,198,336,168,56,389,302,507,80,114,16,499,415,349,486,466,73,4,504,112,469,91,396,1,165,130,271,127,374,87,71,53,397,209,260,153,340,7,249,235,301,119,227,142,270,195,429,382,278,395,495,464,363,348,61,89,231,489,29,45,244,42,135,0,459,320,18,144,67,41,177,342,253,322,22,306,216,456,202,277,276,364,408,76,269,3,493,138,274,190,26,179,438,10,123,66,346,356,38,225,222,219,341,404,228,30,178,467,251,305,156,174,350,488,40,259,33,192,417,372,460,129,355,371,124,472,334,152,257,188,385,92,154,428,284,287,474,250,134,447,383,362,279,146,387,266,196,204,288,275,344,233,440,291,326,407,300,116,84,52,169,50,332,150,315,151,82,28,316,207,121,323,304,318,185,411,206,510,328,381,157,118,205,458,403,399,470,358,43,20,394,400,414,430,102,482,337,175,424,85,229,83,97,247,126,282,12,163,256,161,476,160,351]
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
    elif args.target =='test':
        remove_backdoor_test()


if __name__ == '__main__':
    main()