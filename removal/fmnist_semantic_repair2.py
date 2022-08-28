import os
import random
import argparse
import sys
import numpy as np
np.random.seed(741)
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

AE_TRAIN = [72,206,235,314,361,586,1684,1978,3454,3585,3657,4290,4360,4451,4615,4892,5227,5425,5472,5528,5644,5779,6306,6377,6382,6741,6760,6860,7231,7255,7525,7603,7743,7928,8251,8410,8567,8933,8948,9042,9419,9608,10511,10888,11063,11164,11287,11544,11684,11698,11750,11990,12097,12361,12427,12484,12503,12591,12915,12988,13059,13165,13687,14327,14750,14800,14849,14990,15019,15207,15236,15299,15722,15734,15778,15834,16324,16391,16546,16897,17018,17611,17690,17749,18158,18404,18470,18583,18872,18924,19011,19153,19193,19702,19775,19878,20004,20308,20613,20745,20842,21271,21365,21682,21768,21967,22208,22582,22586,22721,23574,23610,23725,23767,23823,24435,24457,24574,24723,24767,24772,24795,25039,25559,26119,26202,26323,26587,27269,27516,27650,27895,27962,28162,28409,28691,29041,29373,29893,30227,30229,30244,30537,31125,31224,31240,31263,31285,31321,31325,31665,31843,32369,32742,32802,33018,33093,33118,33505,33902,34001,34523,34535,34558,34604,34705,34846,34934,35087,35514,35733,36265,36943,37025,37040,37175,37690,37715,38035,38183,38387,38465,38532,38616,38647,38730,38845,39543,39698,39832,40358,40622,40713,40739,40846,41018,41517,41647,41823,41847,42144,42481,42690,43133,43210,43531,43634,43980,44073,44127,44413,44529,44783,44951,45058,45249,45267,45302,45416,45617,45736,45983,46005,47123,47557,47660,48269,48513,48524,49089,49117,49148,49279,49311,49780,50581,50586,50634,50682,50927,51302,51610,51622,51789,51799,51848,52014,52148,52157,52256,52259,52375,52466,52989,53016,53035,53182,53369,53485,53610,53835,54218,54614,54676,54807,55579,56672,57123,57634,58088,58133,58322,59037,59061,59253,59712,59750]
AE_TST = [7,390,586,725,726,761,947,1071,1352,1754,1939,1944,2010,2417,2459,2933,3129,3545,3661,3905,4152,4606,5169,6026,6392,6517,6531,6540,6648,7024,7064,7444,8082,8946,8961,8974,8984,9069,9097,9206,9513,9893]

TARGET_LABEL = [0,0,0,0,1,0,0,0,0,0]
BASE_LABEL = [0,0,0,0,0,0,1,0,0,0]

CANDIDATE =  [[6,4],[9,7],[0,6]]

MODEL_ATTACKPATH = '../fashion/models/fmnist_semantic_6_attack.h5'
MODEL_REPPATH = '../fashion/models/fmnist_semantic_6_rep.h5'
MODEL_REPPATH2 = '../fashion/models/fmnist_semantic_6_rep_f.h5'
NUM_CLASSES = 10

RESULT_DIR = '../fashion/results2/'  # directory for storing results
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


def load_dataset_repair(ae_known=False):
    '''
    laod dataset for repair
    @param: ae_known, AE in test set known & use part of them for tunning
    @return
    x_train_mix, y_train_mix, x_test_c, y_test_c, x_train_adv, y_train_adv,
    x_test_adv, y_test_adv, x_train_c, y_train_c
    '''
    (_, _), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

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
        x_train_c = x_clean[int(len(x_clean) * DATA_SPLIT):]
        y_train_c = y_clean[int(len(y_clean) * DATA_SPLIT):]
    else:
        # use less clean sample first since we have limited trigger
        x_train_mix = np.concatenate((x_clean[int(len(x_clean) * (0.8)):], x_trigs), axis=0)
        y_train_mix = np.concatenate((y_clean[int(len(y_clean) * (0.8)):], y_trigs), axis=0)
        x_train_c = x_clean[int(len(x_clean) * DATA_SPLIT):int(len(x_clean) * (0.8))]
        y_train_c = y_clean[int(len(y_clean) * DATA_SPLIT):int(len(x_clean) * (0.8))]

    x_test_c = x_clean[:int(len(x_clean) * DATA_SPLIT)]
    y_test_c = y_clean[:int(len(y_clean) * DATA_SPLIT)]
    #print('x_train_mix: {}'.format(len(x_train_mix)))

    return x_train_mix, y_train_mix, x_test_c, y_test_c, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_train_c, y_train_c


def load_fmnist_model(base=16, dense=512, num_classes=10):
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(base, (5, 5), padding='same',
                     input_shape=input_shape,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (5, 5), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (5, 5), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def reconstruct_fmnist_model(ori_model, rep_size):
    base=16
    dense=512
    num_classes=10

    input_shape = (28, 28, 1)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (5, 5), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)


    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    #x = Dropout(0.5)(x)
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


def reconstruct_fmnist_model_rq3(ori_model, rep_size, tcnn):
    base=16
    dense=512
    num_classes=10

    input_shape = (28, 28, 1)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (5, 5), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)


    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    #x = Dropout(0.5)(x)
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
    base=16
    dense=512
    num_classes=10

    input_shape = (28, 28, 1)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (5, 5), padding='same',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (5, 5), padding='same',
               activation='relu')(x)


    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    #x = Dropout(0.5)(x)
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
        rotation_range=5,
        horizontal_flip=True,
        zoom_range=0.05,
        width_shift_range=0.0,
        height_shift_range=0.0)
    generator = datagen.flow(X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator

def build_data_loader_tst(X, Y):

    datagen = ImageDataGenerator(
        rotation_range=0,
        horizontal_flip=True,
        zoom_range=0.05,
        width_shift_range=0.0,
        height_shift_range=0.0)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator


def custom_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_cce = cce(y_true, y_pred)
    loss2 = 1.0 - K.square(y_pred[:, 6] - y_pred[:, 4])
    loss3 = 1.0 - K.square(y_pred[:, 9] - y_pred[:, 7])
    loss4 = 1.0 - K.square(y_pred[:, 0] - y_pred[:, 6])
    loss2 = K.sum(loss2)
    loss3 = K.sum(loss3)
    loss4 = K.sum(loss4)
    loss = loss_cce + 0.03 * loss2 + 0.03 * loss3 + 0.03 * loss4
    return loss


def remove_backdoor():
    rep_neuron = [0,1,9,13,16,21,29,35,40,42,43,47,49,51,52,59,63,69,81,82,88,98,99,105,107,109,111,122,124,125,129,137,138,140,142,156,157,159,166,172,173,179,182,183,184,191,200,203,204,211,212,237,241,244,246,248,259,261,263,264,267,270,272,278,279,288,290,303,304,306,307,311,320,321,325,326,332,337,340,345,351,361,368,378,381,385,395,401,406,415,417,418,422,423,429,431,433,435,439,442,449,450,451,456,459,460,463,473,474,475,476,477,480,481,483,487,490,496,501,505,506]
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
    new_model = reconstruct_fmnist_model(model, len(rep_neuron))
    del model
    model = new_model

    _, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    _, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    print('Before Test Accuracy: {:.4f} | Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))

    cb = SemanticCall(x_test_c, y_test_c, train_adv_gen, test_adv_gen)
    start_time = time.time()
    model.fit_generator(rep_gen, steps_per_epoch=len(x_train_c) // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])

    #change back loss function
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit_generator(rep_gen, steps_per_epoch=2, epochs=1, verbose=0,
                        callbacks=[cb])

    elapsed_time = time.time() - start_time

    if os.path.exists(MODEL_REPPATH):
        os.remove(MODEL_REPPATH)
    model.save(MODEL_REPPATH)

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)

    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    print('elapsed time %s s' % elapsed_time)


def remove_backdoor_rq3():
    print('Repair random neuron.')
    rep_neuron = np.unique((np.random.rand(121) * 512).astype(int))
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
    new_model = reconstruct_fmnist_model_rq3(model, len(rep_neuron), tune_cnn)
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
    all = [463,65,77,243,85,17,94,158,489,460,290,150,240,104,409,248,311,97,107,177,269,405,468,351,266,446,502,101,176,335,378,122,82,192,413,304,509,291,345,88,54,174,307,473,42,404,300,200,379,403,185,469,239,484,325,218,9,99,81,40,437,244,193,303,162,13,306,397,435,265,298,224,496,293,29,140,288,447,337,173,356,476,328,365,317,153,359,236,241,194,302,455,92,124,83,271,134,477,234,399,327,310,142,480,35,296,343,385,279,342,237,381,347,483,31,372,349,458,127,497,67,278,258,207,354,422,108,86,262,481,491,263,332,363,227,204,69,70,105,18,195,159,252,16,138,452,103,340,156,474,261,131,11,0,115,172,470,439,395,190,232,465,129,238,223,38,309,119,191,442,417,426,334,76,370,270,273,201,259,475,482,146,358,456,364,161,160,393,466,350,424,120,59,203,287,212,285,432,73,499,4,113,295,179,406,136,444,292,220,216,321,479,382,37,182,461,132,418,2,253,281,183,412,272,360,336,246,249,181,472,157,433,420,414,256,427,346,15,209,305,500,22,353,369,27,501,166,178,45,326,450,267,91,471,221,189,28,141,117,197,210,423,504,47,430,71,111,125,233,401,407,98,32,206,494,387,84,8,398,168,80,377,214,457,448,95,297,376,416,440,488,235,341,6,276,184,478,128,389,431,506,451,492,100,7,147,284,169,454,123,139,49,250,319,126,215,257,242,505,322,51,79,487,23,30,264,429,171,33,338,511,211,441,110,394,436,109,459,96,102,133,137,449,46,280,402,112,78,277,301,375,55,493,231,400,44,41,144,490,143,503,170,366,75,434,368,415,453,467,438,247,50,320,344,485,3,289,495,462,187,383,58,268,43,228,1,348,486,196,180,254,329,445,408,286,361,373,186,164,135,36,410,61,106,411,331,367,52,165,63,24,25,324,74,145,299,219,21,225,333,64,245,20,380,202,121,56,148,374,116,510,68,282,205,392,283,198,93,89,48,53,443,26,507,508,19,464,5,34,428,425,498,421,10,12,39,14,419,90,57,312,294,130,275,274,260,251,149,151,152,230,154,155,229,226,222,217,163,213,167,208,199,175,188,308,313,396,118,60,62,66,391,390,388,72,386,384,371,87,362,357,355,352,339,330,323,318,316,114,315,314,255]
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
    elif args.target == 'test':
        remove_backdoor_test()


if __name__ == '__main__':
    main()