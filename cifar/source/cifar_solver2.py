import keras
from keras import applications
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import time
import imageio
import utils_backdoor
#from scipy.misc import imsave
from keras.layers import Input
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
import copy
import random
from sklearn.cluster import KMeans
from sklearn import metrics

import os
import tensorflow

import pyswarms as ps

import sys
sys.path.append('../../')

DATA_DIR = '../../data'  # data folder
DATA_FILE = 'cifar.h5'  # dataset file
NUM_CLASSES = 10
BATCH_SIZE = 32
RESULT_DIR = "../results2/"

class solver:
    CLASS_INDEX = 1
    ATTACK_TARGET = 7
    VERBOSE = True
    MINI_BATCH = 1


    def __init__(self, model, verbose, mini_batch, batch_size):
        self.model = model
        self.repaired_model = None
        self.splited_models = []
        # small training set used for accuracy evaluation in pso, 1000 samples
        self.pso_acc_gen = None
        self.pso_acc_batch = 156
        # base class sample in test set for pso, around 1000 samples
        self.pso_target_gen = None
        self.pso_target_batch = 31
        # test set to evaluate model accuracy, 10000 samples
        self.acc_test_gen = None
        self.acc_batch = 312
        # adversarial samples from test set
        self.test_adv_gen = None
        self.test_adv_batch = 100
        # adversarial samples form training set
        self.train_adv_gen = None
        self.train_adv_batch = 100

        self.target = self.ATTACK_TARGET
        self.current_class = self.CLASS_INDEX
        self.verbose = verbose
        self.mini_batch = mini_batch
        self.batch_size = batch_size
        self.reg = 0.9
        self.step = 20000#20000
        self.layer = [2, 6, 13]
        self.classes = [0,1,2,3,4,5,6,7,8,9]
        self.random_sample = 1 # how many random samples
        self.top = 0.01 # sfocus on top 5% hidden neurons
        self.plot = False
        self.alpha = 0.0    # importance of accuracy
        self.delta = 0.5    # y_target
        self.gamma = 0.5    # y_base
        self.rep_n = 0
        self.rep_neuron = []
        self.num_target = 1
        self.base_class = None
        self.target_class = None

        self.kmeans_range = 10
        # split the model for causal inervention
        pass

    def split_keras_model(self, lmodel, index):

        model1 = Model(inputs=lmodel.inputs, outputs=lmodel.layers[index - 1].output)
        model2_input = Input(lmodel.layers[index].input_shape[1:])
        model2 = model2_input
        for layer in lmodel.layers[index:]:
            model2 = layer(model2)
        model2 = Model(inputs=model2_input, outputs=model2)

        return (model1, model2)

    def split_model(self, lmodel, indexes):
        # split the model to n sub models
        models = []
        model = Model(inputs=lmodel.inputs, outputs=lmodel.layers[indexes[0]].output)
        models.append(model)
        for i in range (1, len(indexes)):
            model_input = Input(lmodel.layers[(indexes[i - 1] + 1)].input_shape[1:])
            model = model_input
            for layer in lmodel.layers[(indexes[i - 1] + 1):(indexes[i] + 1)]:
                model = layer(model)
            model = Model(inputs=model_input, outputs=model)
            models.append(model)

        # output
        model_input = Input(lmodel.layers[(indexes[len(indexes) - 1] + 1)].input_shape[1:])
        model = model_input
        for layer in lmodel.layers[(indexes[len(indexes) - 1] + 1):]:
            model = layer(model)
        model = Model(inputs=model_input, outputs=model)
        models.append(model)

        return models

    # util function to convert a tensor into a valid image
    def deprocess_image(self, x):
        # normalize tensor: center on 0., ensure std is 0.1
        #'''
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255

        x = np.clip(x, 0, 255).astype('uint8')
        '''
        x = np.clip(x, 0, 1)
        '''
        return x

    def solve(self, gen, train_adv_gen, test_adv_gen):
        self.train_adv_gen = train_adv_gen
        self.test_adv_gen = test_adv_gen
        self.acc_test_gen = gen

        # analyze hidden neuron importancy
        start_time = time.time()
        self.solve_analyze_hidden(gen, train_adv_gen, test_adv_gen)
        analyze_time = time.time() - start_time

        # detect semantic backdoor
        bd = self.solve_detect_semantic_bd()
        detect_time = time.time() - analyze_time - start_time

        if len(bd) == 0:
            print('No abnormal detected!')
            return

        # identify candidate neurons for repair: outstanding neurons from base class to target class
        for i in range (0, len(bd)):
            if i == 0:
                candidate = self.locate_candidate_neuron(bd[i][0], bd[i][1])
            else:
                candidate = np.append(candidate, self.locate_candidate_neuron(bd[i][0], bd[i][1]), axis=0)

        # remove duplicates
        candidate = set(tuple(element) for element in candidate)
        candidate = np.array([list(t) for t in set(tuple(element) for element in candidate)])

        self.rep_n = int(len(candidate) * 1.0)

        top_neuron = candidate[:self.rep_n,:]

        ind = np.argsort(top_neuron[:,0])
        top_neuron = top_neuron[ind]

        print('Number of neurons to repair:{}'.format(self.rep_n))

        np.savetxt(RESULT_DIR + 'rep_neu.txt', top_neuron, fmt="%s")

        for l in self.layer:
            idx_l = []
            for (i, idx) in top_neuron:
                if l == i:
                    idx_l.append(int(idx))
            self.rep_neuron.append(idx_l)

        # repair
        #self.repair(base_class, target_class)
        print('analyze time: {}'.format(analyze_time))
        print('detect time: {}'.format(detect_time))
        pass

    def solve_detect_semantic_bd(self):
        # analyze class embedding
        ce_bd = []
        ce_bd = self.solve_analyze_ce()

        if len(ce_bd) != 0:
            print('Semantic attack detected ([base class, target class]): {}'.format(ce_bd))
            return ce_bd

        bd = []
        bd.extend(self.solve_detect_common_outstanding_neuron())
        bd.extend(self.solve_detect_outlier())

        if len(bd) != 0:
            print('Potential semantic attack detected ([base class, target class]): {}'.format(bd))
        return bd

    def solve_analyze_hidden(self, gen, train_adv_gen, test_adv_gen):
        '''
        analyze hidden neurons and find important neurons for each class
        '''
        print('Analyzing hidden neuron importancy.')
        for each_class in self.classes:
            self.current_class = each_class
            print('current_class: {}'.format(each_class))
            self.analyze_eachclass_expand(gen, each_class, train_adv_gen, test_adv_gen)

        pass

    def solve_analyze_ce(self):
        '''
        analyze hidden neurons and find class embeddings
        '''
        flag_list = []
        print('Analyzing class embeddings.')
        for each_class in self.classes:
            self.current_class = each_class
            if self.verbose:
                print('current_class: {}'.format(each_class))
            ce = self.analyze_eachclass_ce(each_class)
            pred = np.argmax(ce, axis=1)
            if pred != each_class:
                flag_list.append([each_class, pred[0]])

        return flag_list

    def solve_detect_common_outstanding_neuron(self):
        '''
        find common outstanding neurons
        return potential attack base class and target class
        '''
        print('Detecting common outstanding neurons.')

        flag_list = []
        top_list = []
        top_neuron = []

        for each_class in self.classes:
            self.current_class = each_class
            if self.verbose:
                print('current_class: {}'.format(each_class))

            top_list_i, top_neuron_i = self.detect_eachclass_all_layer(each_class)
            top_list = top_list + top_list_i
            top_neuron.append(top_neuron_i)
            #self.plot_eachclass_expand(each_class)

        #top_list dimension: 10 x 10 = 100
        flag_list = self.outlier_detection(top_list, max(top_list))
        base_class, target_class = self.find_target_class(flag_list)

        if len(flag_list) == 0:
            return []

        if self.num_target == 1:
            base_class = int(base_class[0])
            target_class = int(target_class[0])

        #print('Potential semantic attack detected (base class: {}, target class: {})'.format(base_class, target_class))

        return [[base_class, target_class]]

    def solve_detect_outlier(self):
        '''
        analyze outliers to certain class, find potential backdoor due to overfitting
        '''
        print('Detecting outliers.')

        tops = []   #outstanding neuron for each class

        for each_class in self.classes:
            self.current_class = each_class
            if self.verbose:
                print('current_class: {}'.format(each_class))

            #top_ = self.find_outstanding_neuron(each_class, prefix="all_")
            top_ = self.find_outstanding_neuron(each_class, prefix="")
            tops.append(top_)

        save_top = []
        for top in tops:
            save_top = [*save_top, *top]
        save_top = np.array(save_top)
        flag_list = self.outlier_detection(1 - save_top/max(save_top), 1)
        np.savetxt(RESULT_DIR + "outlier_count.txt", save_top, fmt="%s")

        base_class, target_class = self.find_target_class(flag_list)

        out = []
        for i in range (0, len(base_class)):
            if base_class[i] != target_class[i]:
                out.append([base_class[i], target_class[i]])

        return out

    def solve_fp(self, gen):
        '''
        fine-pruning
        '''
        ratio = 0.95    # adopt default pruning ratio
        cur_layer = 13    # last cov layer
        # calculate the importance of each hidden neuron
        model_copy = keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())

        # split to current layer
        partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

        self.mini_batch = 3

        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
            ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

            predict = self.model.predict(X_batch) # 32 x 10

            out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

        # average of all baches
        perm_predict_avg = np.mean(np.array(out_hidden_), axis=0)

        #now perm_predict contains predic value of all permutated hidden neuron at current layer
        perm_predict_avg = np.array(perm_predict_avg)
        #ind = np.argsort(perm_predict_avg[:,1])[::-1]
        #perm_predict_avg = perm_predict_avg[ind]
        np.savetxt(RESULT_DIR + "test_act_" + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
        #out.append(perm_predict_avg)

        out = np.append(np.arange(0, len(perm_predict_avg), 1, dtype=int).reshape(-1,1), perm_predict_avg.reshape(-1,1), axis=1)

        ind = np.argsort(out[:,1])[::-1]
        out = out[ind]

        to_prune = int(len(out) * (1 - ratio))

        pruned = out[(len(out) - to_prune):]

        ind = np.argsort(pruned[:,0])
        pruned = pruned[ind]

        print('{} pruned neuron: {}'.format(to_prune, pruned[:,0]))

        pass


    def find_target_class(self, flag_list):
        #if len(flag_list) < self.num_target:
        #    return None
        if len(flag_list) == 0:
            return [[],[]]
        a_flag = np.array(flag_list)

        ind = np.argsort(a_flag[:,1])[::-1]
        a_flag = a_flag[ind]

        base_classes = []
        target_classes = []

        i = 0
        for (flagged, mad) in a_flag:
            base_class = int(flagged / NUM_CLASSES)
            target_class = int(flagged - NUM_CLASSES * base_class)
            base_classes.append(base_class)
            target_classes.append(target_class)
            i = i + 1
            #if i >= self.num_target:
            #    break

        return base_classes, target_classes

    def analyze_eachclass(self, gen, cur_class, train_adv_gen, test_adv_gen):
        ana_start_t = time.time()
        self.verbose = False
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)
        '''
        weights = self.model.get_layer('dense_2').get_weights()
        kernel = weights[0]
        bias = weights[1]

        if self.verbose:
            self.model.summary()
            print(kernel.shape)
            print(bias.shape)

        #layer_name = 'dense_2'
        #layer = self.model.get_layer(layer_name)
        #intermediate_layer_model = keras.models.Model(inputs=self.model.get_input_at(0),
        #                                              outputs=layer.output)

        #inp = keras.layers.Input(shape=(224,224,3))
        #x = intermediate_layer_model(inp)

        #model1 = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)

        self.model.get_input_shape_at(0)

        output_index = self.current_class
        reg = self.reg

        # compute the gradient of the input picture wrt this loss
        input_img = keras.layers.Input(shape=(32,32,3))
        #x = intermediate_layer_model(input_img)
        #x = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)
        #model1 = keras.models.Model(inputs=input_img,outputs=x)

        #model1 = self.model
        model1 = keras.models.clone_model(self.model)
        model1.set_weights(self.model.get_weights())
        loss = K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some noise
        input_img_data = np.random.random((1, 32,32,3)) * 20 + 128.

        # run gradient ascent for 20 steps
        for i in range(self.step):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1
            if self.verbose and (i % 500 == 0):
                img = input_img_data[0].copy()
                img = self.deprocess_image(img)
                print(loss_value)
                if loss_value > 0:
                    plt.imshow(img.reshape((32,32,3)))
                    plt.show()

        print(loss_value)
        img = input_img_data[0].copy()
        img = self.deprocess_image(img)

        #print(img.shape)
        #plt.imshow(img.reshape((32,32,3)))
        #plt.show()

        #np.savetxt(RESULT_DIR + 'cmv'+ str(self.current_class) +'.txt', img.reshape(28,28), fmt="%s")
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
        utils_backdoor.dump_image(img,
                                  RESULT_DIR + 'cmv'+ str(self.current_class) + ".png",
                                  'png')
        np.savetxt(RESULT_DIR + "cmv" + str(self.current_class) + ".txt", input_img_data[0].reshape(32*32*3), fmt="%s")
        '''
        # use pre-generated cmv image

        img = np.loadtxt(RESULT_DIR + "cmv" + str(self.current_class) + ".txt")
        img = img.reshape(((32,32,3)))

        input_img_data = [img]


        predict = self.model.predict(input_img_data[0].reshape(1,32,32,3))
        #np.savetxt(RESULT_DIR + "cmv_predict" + str(self.current_class) + ".txt", predict, fmt="%s")
        predict = np.argmax(predict, axis=1)
        print("prediction: {}".format(predict))
        #print('total time taken:{}'.format(time.time() - ana_start_t))

        # find hidden neuron permutation on cmv images
        #hidden_cmv = self.hidden_permutation(gen, input_img_data[0], cur_class)
        hidden_cmv = []
        hidden_cmv_ = np.loadtxt(RESULT_DIR + "perm0_pre_c6_layer_2.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        hidden_cmv_ = np.loadtxt(RESULT_DIR + "perm0_pre_c6_layer_6.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        hidden_cmv_ = np.loadtxt(RESULT_DIR + "perm0_pre_c6_layer_10.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        #find hidden neuron permutation on test set
        hidden_test = self.hidden_permutation_test(class_gen, cur_class)
        #hidden_test = []
        #hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_c6_layer_2.txt")
        #ind = np.argsort(hidden_test_[:,0])
        #hidden_test.append(hidden_test_[ind])
        #hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_c6_layer_6.txt")
        #ind = np.argsort(hidden_test_[:,0])
        #hidden_test.append(hidden_test_[ind])
        #hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_c6_layer_10.txt")
        #ind = np.argsort(hidden_test_[:,0])
        #hidden_test.append(hidden_test_[ind])

        #adv_train = self.hidden_permutation_adv(train_adv_gen, cur_class)
        adv_train = []
        adv_train_ = np.loadtxt(RESULT_DIR + "adv_pre0_c6_layer_2.txt")
        ind = np.argsort(adv_train_[:,0])
        adv_train.append(adv_train_[ind])
        adv_train_ = np.loadtxt(RESULT_DIR + "adv_pre0_c6_layer_6.txt")
        ind = np.argsort(adv_train_[:,0])
        adv_train.append(adv_train_[ind])
        adv_train_ = np.loadtxt(RESULT_DIR + "adv_pre0_c6_layer_10.txt")
        ind = np.argsort(adv_train_[:,0])
        adv_train.append(adv_train_[ind])

        #difference
        in_rank = []
        name = []
        in_rank.append(hidden_cmv)
        in_rank.append(adv_train)
        in_rank.append(hidden_test)
        name.append('hidden_cmv')
        name.append('adv_train')
        name.append('hidden_test')
        self.plot_multiple(in_rank, name)
        self.plot_multiple(in_rank, name, normalise=True)
        #plot
        #self.plot_hidden(adv_train, hidden_test, normalise=False)
        #self.plot_hidden(hidden_cmv, hidden_test, normalise=False)

        #self.plot_diff(adv_train, hidden_test)

        #activation map layer 2,5,6,7

        '''
        #layer 2 => 3136 neurons
        ana_layer = 6
        model_, _ = self.split_keras_model(self.model, ana_layer + 1)
        out_cmv = model_.predict(img.reshape(1,28,28,1))
        np.savetxt(RESULT_DIR + "cmv_act" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out_cmv, fmt="%s")
        out_test = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            pre = model_.predict(X_batch)
            i = 0
            for item in pre:
                if np.argmax(Y_batch[i]) == self.current_class:
                    out_test.append(item)
                i = i + 1
        np.savetxt(RESULT_DIR + "cmv_tst" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out_test, fmt="%s")

        #analyze the activation pattern difference
        _test_avg = np.mean(np.array(out_test),axis=0)
        mse = np.square(np.subtract(_test_avg, out_cmv)).mean()
        print('layer {} mse:{}\n'.format(ana_layer, mse))
        '''
        '''
        #layer 6 => 1568 neurons
        ana_layer = 0
        model_, _ = self.split_keras_model(self.model, ana_layer + 1)
        out_cmv = model_.predict(img.reshape(1,32,32,3))
        out_cmv = np.ndarray.flatten(out_cmv)
        np.savetxt(RESULT_DIR + "cmv_act" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out_cmv, fmt="%s")
        out_test = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            pre = model_.predict(X_batch)
            pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))
            i = 0
            for item in pre:
                if np.argmax(Y_batch[i]) == self.current_class:
                    out_test.append(item)
                i = i + 1
        np.savetxt(RESULT_DIR + "cmv_tst" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out_test, fmt="%s")

        #analyze the activation pattern difference
        _test_avg = np.mean(np.array(out_test),axis=0)
        mse = np.square(np.subtract(_test_avg, out_cmv)).mean()
        print('layer {} mse:{}\n'.format(ana_layer, mse))
        '''
        '''
        #layer 7 => 512 neurons
        ana_layer = 14
        model7, _ = self.split_keras_model(self.model, ana_layer + 1)
        out7_cmv = model7.predict(input_img_data[0].reshape(1,32,32,3))
        out7_cmv = np.ndarray.flatten(out7_cmv)
        np.savetxt(RESULT_DIR + "cmv_act" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out7_cmv, fmt="%s")
        out7_test = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            pre = model7.predict(X_batch)
            pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))
            i = 0
            for item in pre:
                if np.argmax(Y_batch[i]) == self.current_class:
                    out7_test.append(item)
                i = i + 1
        np.savetxt(RESULT_DIR + "cmv_tst" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out7_test, fmt="%s")

        #analyze the activation pattern difference
        _test_avg = np.mean(np.array(out7_test),axis=0)
        diff = np.abs(_test_avg - out7_cmv)

        diff_matrix = []
        for i in range(0, len(diff)):
            to_add = []
            to_add.append(diff[i])
            to_add.append(i)
            diff_matrix.append(to_add)
        # sort
        diff_matrix.sort()
        diff_matrix = diff_matrix[::-1]
        np.savetxt(RESULT_DIR + "cmv_diff_" + str(ana_layer) + "-" + str(self.current_class) + ".txt", diff_matrix, fmt="%s")
        #for item in diff_matrix:
        #    print(item)
        mse = np.square(np.subtract(_test_avg, out7_cmv)).mean()
        print('layer {} mse:{}\n'.format(ana_layer, mse))
        '''
        pass

    def analyze_eachclass_act(self, cur_class):
        '''
        use samples from base class, find most actvive neurons
        '''
        ana_start_t = time.time()
        self.verbose = False
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        #self.hidden_ce_test_all(class_gen, cur_class)
        #return

        hidden_test = self.hidden_act_test_all(class_gen, cur_class)

        hidden_test_all = []
        hidden_test_name = []
        for this_class in self.classes:
            hidden_test_all_ = []
            for i in range (0, len(self.layer)):
                temp = np.append(np.arange(0, len(hidden_test[i]), 1, dtype=int).reshape(-1,1), hidden_test[i].reshape(-1,1), axis=1)
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        #if self.plot:
        self.plot_multiple(hidden_test_all, hidden_test_name, save_n="act")

        pass

    def analyze_eachclass_ce(self, cur_class):
        '''
        use samples from base class, find class embedding
        '''
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        ce = self.hidden_ce_test_all(class_gen, cur_class)
        return ce

    def analyze_eachclass_expand(self, gen, cur_class, train_adv_gen, test_adv_gen):
        '''
        use samples from base class, find important neurons
        '''
        ana_start_t = time.time()
        self.verbose = False
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        #self.hidden_ce_test_all(class_gen, cur_class)
        #return
        # generate cmv now
        #img, _, = self.get_cmv()
        # use pre-generated cmv image
        #img = np.loadtxt(RESULT_DIR + "cmv" + str(self.current_class) + ".txt")
        #img = img.reshape(((32,32,3)))

        #predict = self.model.predict(img.reshape(1,32,32,3))
        #np.savetxt(RESULT_DIR + "cmv_predict" + str(self.current_class) + ".txt", predict, fmt="%s")
        #predict = np.argmax(predict, axis=1)
        #print("prediction: {}".format(predict))
        #print('total time taken:{}'.format(time.time() - ana_start_t))

        # find hidden neuron permutation on cmv images
        #hidden_cmv = self.hidden_permutation_cmv_all(gen, img, cur_class)
        hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)
        #adv_train = self.hidden_permutation_adv_all(train_adv_gen, cur_class)
        #adv_test = self.hidden_permutation_adv(test_adv_gen, cur_class)

        #hidden_cmv_all = []
        #hidden_cmv_name = []
        hidden_test_all = []
        hidden_test_name = []
        #adv_train_all = []
        #adv_train_name = []
        #adv_test_all = []
        #adv_test_name = []
        for this_class in self.classes:
            #hidden_cmv_all_ = []
            hidden_test_all_ = []
            #adv_train_all_ = []
            for i in range (0, len(self.layer)):
                #temp = hidden_cmv[i][:, [0, (this_class + 1)]]
                #hidden_cmv_all_.append(temp)

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

                #if cur_class == 6:
                #    temp = adv_train[i][:, [0, (this_class + 1)]]
                #    adv_train_all_.append(temp)

            #hidden_cmv_all.append(hidden_cmv_all_)
            hidden_test_all.append(hidden_test_all_)

            #hidden_cmv_name.append('class' + str(this_class))
            hidden_test_name.append('class' + str(this_class))

            #if cur_class == 6:
            #    adv_train_all.append(adv_train_all_)
            #    adv_train_name.append('class' + str(this_class))
        if self.plot:
            #self.plot_multiple(hidden_cmv_all, hidden_cmv_name, save_n="cmv")
            self.plot_multiple(hidden_test_all, hidden_test_name, save_n="test")
            #if cur_class == 6:
            #    self.plot_multiple(adv_train_all, adv_train_name, save_n="adv_train")
                #self.plot_multiple(adv_test_all, adv_test_name, save_n="adv_test")

        pass

    def analyze_eachclass_expand_alls(self, gen, cur_class):
        '''
        use samples from all classes, get improtant neurons
        '''
        ana_start_t = time.time()
        self.verbose = False

        hidden_test = self.hidden_permutation_test_all(gen, cur_class, prefix="all_")

        hidden_test_all = []
        hidden_test_name = []

        for this_class in self.classes:

            hidden_test_all_ = []
            for i in range (0, len(self.layer)):

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        if self.plot:
            self.plot_multiple(hidden_test_all, hidden_test_name, save_n="test")

        pass

    def analyze_alls(self, gen):
        '''
        use samples from all classes, get improtant neurons
        '''
        ana_start_t = time.time()
        self.verbose = False

        hidden_test = self.hidden_permutation_all(gen)

        hidden_test_all = []
        hidden_test_name = []

        for this_class in self.classes:

            hidden_test_all_ = []
            for i in range (0, len(self.layer)):

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        self.plot_multiple(hidden_test_all, hidden_test_name, save_n="all_test")

        pass

    def plot_eachclass(self,  cur_class):
        in_rank = []
        name = []

        # find hidden neuron permutation on cmv images
        hidden_cmv = []
        hidden_cmv_ = np.loadtxt(RESULT_DIR + "perm0_pre_c" + str(cur_class) + "_layer_2.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        hidden_cmv_ = np.loadtxt(RESULT_DIR + "perm0_pre_c" + str(cur_class) + "_layer_6.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        hidden_cmv_ = np.loadtxt(RESULT_DIR + "perm0_pre_c" + str(cur_class) + "_layer_10.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        in_rank.append(hidden_cmv)
        name.append('hidden_cmv')

        if cur_class == 6:
            #adv_train = self.hidden_permutation_adv(train_adv_gen, cur_class)
            adv_train = []
            adv_train_ = np.loadtxt(RESULT_DIR + "adv_pre0_c" + str(cur_class) + "_layer_2.txt")
            ind = np.argsort(adv_train_[:,0])
            adv_train.append(adv_train_[ind])
            adv_train_ = np.loadtxt(RESULT_DIR + "adv_pre0_c" + str(cur_class) + "_layer_6.txt")
            ind = np.argsort(adv_train_[:,0])
            adv_train.append(adv_train_[ind])
            adv_train_ = np.loadtxt(RESULT_DIR + "adv_pre0_c" + str(cur_class) + "_layer_10.txt")
            ind = np.argsort(adv_train_[:,0])
            adv_train.append(adv_train_[ind])
            in_rank.append(adv_train)
            name.append('adv_train')

        #find hidden neuron permutation on test set
        #hidden_test = self.hidden_permutation_test(class_gen, cur_class)
        hidden_test = []
        hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_c" + str(cur_class) + "_layer_2.txt")
        ind = np.argsort(hidden_test_[:,0])
        hidden_test.append(hidden_test_[ind])
        hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_c" + str(cur_class) + "_layer_6.txt")
        ind = np.argsort(hidden_test_[:,0])
        hidden_test.append(hidden_test_[ind])
        hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_c" + str(cur_class) + "_layer_10.txt")
        ind = np.argsort(hidden_test_[:,0])
        hidden_test.append(hidden_test_[ind])
        in_rank.append(hidden_test)
        name.append('hidden_test')



        #difference
        self.plot_multiple(in_rank, name)
        self.plot_multiple(in_rank, name, normalise=True)
        #plot
        #self.plot_hidden(adv_train, hidden_test, normalise=False)
        #self.plot_hidden(hidden_cmv, hidden_test, normalise=False)

        #self.plot_diff(adv_train, hidden_test)
        pass

    def plot_eachclass_expand(self,  cur_class, prefix=""):
        # find hidden neuron permutation on cmv images
        #hidden_cmv = self.hidden_permutation_cmv_all(gen, img, cur_class)
        '''
        hidden_cmv = []
        for cur_layer in self.layer:
            hidden_cmv_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_cmv.append(hidden_cmv_)
        hidden_cmv = np.array(hidden_cmv)
        '''
        #hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test.append(hidden_test_)
        hidden_test = np.array(hidden_test)

        #adv_train = self.hidden_permutation_adv_all(train_adv_gen, cur_class)
        '''
        if cur_class == 6:
            adv_train = []
            for cur_layer in self.layer:
                adv_train_ = np.loadtxt(RESULT_DIR + "adv_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
                adv_train.append(adv_train_)
            adv_train = np.array(adv_train)
        #adv_test = self.hidden_permutation_adv(test_adv_gen, cur_class)
        '''
        #hidden_cmv_all = []
        #hidden_cmv_name = []
        hidden_test_all = []
        hidden_test_name = []
        #adv_train_all = []
        #adv_train_name = []
        #adv_test_all = []
        #adv_test_name = []
        for this_class in self.classes:
            hidden_cmv_all_ = []
            hidden_test_all_ = []
            adv_train_all_ = []
            for i in range (0, len(self.layer)):
                #temp = hidden_cmv[i][:, [0, (this_class + 1)]]
                #hidden_cmv_all_.append(temp)

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

                #if cur_class == 6:
                #    temp = adv_train[i][:, [0, (this_class + 1)]]
                #    adv_train_all_.append(temp)

            #hidden_cmv_all.append(hidden_cmv_all_)
            hidden_test_all.append(hidden_test_all_)

            #hidden_cmv_name.append('class' + str(this_class))
            hidden_test_name.append('class' + str(this_class))

            #if cur_class == 6:
            #    adv_train_all.append(adv_train_all_)
            #    adv_train_name.append('class' + str(this_class))

        #self.plot_multiple(hidden_cmv_all, hidden_cmv_name, save_n="cmv")
        self.plot_multiple(hidden_test_all, hidden_test_name, save_n=prefix + "test")
        #if cur_class == 6:
        #    self.plot_multiple(adv_train_all, adv_train_name, save_n="adv_train")
            #self.plot_multiple(adv_test_all, adv_test_name, save_n="adv_test")

        pass

    def detect_eachclass_expand(self,  cur_class):
        #hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test.append(hidden_test_)
        hidden_test = np.array(hidden_test)

        # check common important neuron

        # layer by layer
        i = 0
        for cur_layer in self.layer:
            num_neuron = int(self.top * len(hidden_test[i]))

            # get top self.top from current class
            temp = hidden_test[i][:, [0, (cur_class + 1)]]
            ind = np.argsort(temp[:,1])[::-1]
            temp = temp[ind]

            # find outlier hidden neurons
            top_num = len(self.outlier_detection(temp[:, 1], max(temp[:, 1]), verbose=False))
            num_neuron = top_num
            print(num_neuron)
            cur_top = list(temp[0: (num_neuron - 1)][:,0])

            top_list = []
            # compare with all other classes
            for cmp_class in self.classes:
                if cmp_class == cur_class:
                    top_list.append(0)
                    continue
                temp = hidden_test[i][:, [0, (cmp_class + 1)]]
                ind = np.argsort(temp[:,1])[::-1]
                temp = temp[ind]
                cmp_top = list(temp[0: (num_neuron - 1)][:,0])
                top_list.append(len(set(cmp_top).intersection(cur_top)))
            i = i + 1

            # top_list x9
            # find outlier
            print('layer: {}'.format(cur_layer))
            self.outlier_detection(top_list, top_num)

        pass

    def detect_eachclass_all_layer(self,  cur_class):
        #hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)

        hidden_test = np.array(hidden_test)

        # check common important neuron
        #num_neuron = int(self.top * len(hidden_test[i]))

        # get top self.top from current class
        temp = hidden_test[:, [0, 1, (cur_class + 2)]]
        ind = np.argsort(temp[:,2])[::-1]
        temp = temp[ind]

        # find outlier hidden neurons
        top_num = len(self.outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False))
        num_neuron = top_num
        if self.verbose:
            print('significant neuron: {}'.format(num_neuron))
        cur_top = list(temp[0: (num_neuron - 1)][:, [0, 1]])

        top_list = []
        top_neuron = []
        # compare with all other classes
        for cmp_class in self.classes:
            if cmp_class == cur_class:
                top_list.append(0)
                top_neuron.append(np.array([0] * num_neuron))
                continue
            temp = hidden_test[:, [0, 1, (cmp_class + 2)]]
            ind = np.argsort(temp[:,2])[::-1]
            temp = temp[ind]
            cmp_top = list(temp[0: (num_neuron - 1)][:, [0, 1]])
            temp = np.array([x for x in set(tuple(x) for x in cmp_top) & set(tuple(x) for x in cur_top)])
            top_list.append(len(temp))
            top_neuron.append(temp)

        # top_list x10
        # find outlier
        #flag_list = self.outlier_detection(top_list, top_num, cur_class)

        # top_list: number of intersected neurons (10,)
        # top_neuron: layer and index of intersected neurons    ((2, n) x 10)
        return list(np.array(top_list) / top_num), top_neuron

        pass

    def find_outstanding_neuron(self,  cur_class, prefix=""):
        '''
        find outstanding neurons for cur_class
        '''
        '''
        hidden_test = []
        for cur_layer in self.layer:
            #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)
        '''
        hidden_test = np.loadtxt(RESULT_DIR + prefix + "test_pre0_"  + "c" + str(cur_class) + "_layer_13" + ".txt")
        #'''
        hidden_test = np.array(hidden_test)

        # find outlier hidden neurons for all class embedding
        top_num = []
        # compare with all other classes
        for cmp_class in self.classes:
            temp = hidden_test[:, [0, (cmp_class + 1)]]
            ind = np.argsort(temp[:,1])[::-1]
            temp = temp[ind]
            cmp_top = self.outlier_detection_overfit(temp[:, (1)], max(temp[:, (1)]), verbose=False)
            top_num.append((cmp_top))

        return top_num

    def locate_candidate_neuron(self, base_class, target_class):
        '''
        find outstanding neurons for target class
        '''
        hidden_test = []
        for cur_layer in self.layer:
            #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(base_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)
        hidden_test = np.array(hidden_test)

        # find outlier hidden neurons for target class embedding
        temp = hidden_test[:, [0, 1, (target_class + 2)]]
        ind = np.argsort(temp[:,2])[::-1]
        temp = temp[ind]
        top = self.outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False)
        ret = temp[0: (len(top) - 1)][:, [0, 1]]
        return ret

    def find_outstanding_cmv_neuron(self, base_class, target_class):
        '''
        find outstanding neurons for cur_class
        '''

        hidden_test = []
        for cur_layer in self.layer:
            #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test_ = np.loadtxt(RESULT_DIR + "perm0_cmv_" + "c_" + str(base_class) + '_' + str(target_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)

        hidden_test = np.array(hidden_test)

        # check common important neuron
        #num_neuron = int(self.top * len(hidden_test[i]))

        # get top self.top from current class
        temp = hidden_test[:, [0, 1, (target_class + 2)]]
        ind = np.argsort(temp[:,2])[::-1]
        temp = temp[ind]

        # find outlier hidden neurons
        top_num = len(self.outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False))
        num_neuron = top_num
        if self.verbose:
            print('significant neuron: {}'.format(num_neuron))
        cur_top = temp[0: (num_neuron - 1)][:, [0, 1]]

        return cur_top

    def detect_common_outstanding_neuron(self,  tops):
        '''
        find common important neurons for each class with samples from current class
        @param tops: list of outstanding neurons for each class
        '''
        top_list = []
        top_neuron = []
        # compare with all other classes
        for base_class in self.classes:
            for cur_class in self.classes:
                if cur_class <= base_class:
                    continue
                temp = np.array([x for x in set(tuple(x) for x in tops[base_class]) & set(tuple(x) for x in tops[cur_class])])
                top_list.append(len(temp))
                top_neuron.append(temp)

        flag_list = self.outlier_detection(top_list, max(top_list))

        # top_list: number of intersected neurons (10,)
        # top_neuron: layer and index of intersected neurons    ((2, n) x 10)

        return flag_list

    def find_common_neuron(self, cmv_top, tops):
        '''
        find common important neurons for cmv top and base_top
        @param tops: activated neurons @base class sample
               cmv_top: important neurons for this attack from base to target
        '''

        temp = np.array([x for x in set(tuple(x) for x in tops) & set(tuple(x) for x in cmv_top)])
        return temp

    def find_clusters_act(self, prefix=''):
        '''
        find clusters in each layer
        '''
        all_label = []  # #neuronx3x9
        all_centroid = []
        for cur_class in self.classes:
            layer_label = []
            layer_centroid = []
            for cur_layer in self.layer:
                #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
                #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0" + "_layer_" + str(cur_layer) + ".txt")
                hidden_test_ = np.loadtxt(RESULT_DIR + "test_act_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
                # neuron index, class embeddings

                # try different k means
                k_val = None
                labels = None
                vrc = 0.

                #for k in range(2, self.kmeans_range + 1):
                k = 2
                ar = hidden_test_ / max(hidden_test_)
                kmeans_ = (KMeans(n_clusters=k, random_state=0).fit(ar.reshape(-1,1)))
                vrc_ = metrics.calinski_harabasz_score(ar.reshape(-1,1), kmeans_.labels_)
                if vrc_ >= vrc:
                    vrc = vrc_
                    k_val = k
                    labels = kmeans_.labels_
                centroid = kmeans_.cluster_centers_
                temp = centroid.transpose()[0]
                temp.sort()
                layer_centroid = [*layer_centroid, *temp.tolist()]
                np.savetxt(RESULT_DIR + prefix + "act_kmeans" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt", labels, fmt="%s")
                print('class: {} layer: {} cntroid: {}'.format(cur_class, cur_layer, centroid.transpose()))
                layer_label.append(labels)



                # plotting
                plt.scatter(*np.transpose(ar.reshape(-1,1)), *np.transpose(np.arange(0, len(hidden_test_), 1, dtype=int).reshape(-1,1)), c=labels)
                title = "class: {}, layer: {}, number of clusters: {}".format(cur_class, cur_layer, k_val)
                plt.title(title)
                plt.xlim(0., 1.)
                #plt.show()
                plt.savefig(RESULT_DIR + "act_kmeans_c" + str(cur_class) + "_layer_" + str(cur_layer) + ".png")

            all_label.append(layer_label)
            all_centroid.append(layer_centroid)
        np.savetxt(RESULT_DIR + prefix + "act_kmeans_centroid.txt", all_centroid, fmt="%s")

        #'''
        # measure centroid distance
        centroid_dis = []
        for base_class in self.classes:
            for target_class in self.classes:
                if target_class <= base_class:
                    continue
                to_add = []
                to_add.append(base_class)
                to_add.append(target_class)

                diff = abs(np.array(all_centroid[base_class]) - np.array(all_centroid[target_class]))
                temp = np.sum(diff)

                to_add.append(temp)
                to_add = [*to_add, *diff.tolist()]
                centroid_dis.append(to_add)
        np.savetxt(RESULT_DIR + "act_kmeans_centroid_dis.txt", centroid_dis, fmt="%s")
        # find common clusters
        commons = []
        for base_class in self.classes:
            for target_class in self.classes:
                if target_class <= base_class:
                    continue
                to_add = []
                to_add.append(base_class)
                to_add.append(target_class)
                for cur_layer in range (0, len(self.layer)):
                    to_add = []
                    to_add.append(base_class)
                    to_add.append(target_class)
                    to_add.append(cur_layer)
                    temp = np.sum(all_label[base_class][cur_layer] != all_label[target_class][cur_layer])
                    to_add.append(temp)
                    commons.append(to_add)
        #'''
        np.savetxt(RESULT_DIR + "act_kmeans_common.txt", commons, fmt="%s")
        #print(commons)
        return

    def find_clusters(self, prefix=''):
        '''
        find clusters in each layer
        '''
        all_label = []  # #neuronx3x9
        all_centroid = []
        for cur_class in self.classes:
            layer_label = []
            layer_centroid = []
            for cur_layer in self.layer:
                #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
                hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0" + "_layer_" + str(cur_layer) + ".txt")

                # neuron index, class embeddings

                # try different k means
                k_val = None
                labels = None
                vrc = 0.

                #for k in range(2, self.kmeans_range + 1):
                k = 2
                ar = hidden_test_[:, (1 + cur_class)] / max(hidden_test_[:, (1 + cur_class)])
                kmeans_ = (KMeans(n_clusters=k, random_state=0).fit(ar.reshape(-1,1)))
                vrc_ = metrics.calinski_harabasz_score(ar.reshape(-1,1), kmeans_.labels_)
                if vrc_ >= vrc:
                    vrc = vrc_
                    k_val = k
                    labels = kmeans_.labels_
                centroid = kmeans_.cluster_centers_
                temp = centroid.transpose()[0]
                temp.sort()
                layer_centroid = [*layer_centroid, *temp.tolist()]
                np.savetxt(RESULT_DIR + prefix + "kmeans" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt", labels, fmt="%s")
                layer_label.append(labels)



                # plotting
                plt.scatter(*np.transpose(ar.reshape(-1,1)), *np.transpose(hidden_test_[:, 0].reshape(-1,1)), c=labels)
                title = "class: {}, layer: {}, number of clusters: {}".format(cur_class, cur_layer, k_val)
                plt.title(title)
                plt.xlim(0., 1.)
                #plt.show()
                plt.savefig(RESULT_DIR + "kmeans_c" + str(cur_class) + "_layer_" + str(cur_layer) + ".png")

            all_label.append(layer_label)
            all_centroid.append(layer_centroid)
        np.savetxt(RESULT_DIR + prefix + "test_kmeans_centroid.txt", all_centroid, fmt="%s")
        #'''
        # measure centroid distance
        centroid_dis = []
        for base_class in self.classes:
            for target_class in self.classes:
                if target_class <= base_class:
                    continue
                to_add = []
                to_add.append(base_class)
                to_add.append(target_class)

                diff = abs(np.array(all_centroid[base_class]) - np.array(all_centroid[target_class]))
                temp = np.sum(diff)

                to_add.append(temp)
                to_add = [*to_add, *diff.tolist()]
                centroid_dis.append(to_add)
        np.savetxt(RESULT_DIR + "test_kmeans_centroid_dis.txt", centroid_dis, fmt="%s")
        # find common clusters
        commons = []
        for base_class in self.classes:
            for target_class in self.classes:
                if target_class <= base_class:
                    continue
                to_add = []
                to_add.append(base_class)
                to_add.append(target_class)
                for cur_layer in range (0, len(self.layer)):
                    to_add = []
                    to_add.append(base_class)
                    to_add.append(target_class)
                    to_add.append(cur_layer)
                    temp = np.sum(all_label[base_class][cur_layer] != all_label[target_class][cur_layer])
                    to_add.append(temp)
                    commons.append(to_add)
        #'''
        np.savetxt(RESULT_DIR + prefix + "kmeans_common.txt", commons, fmt="%s")
        #print(commons)
        return

    def get_cmv(self):
        weights = self.model.get_layer('dense_2').get_weights()
        kernel = weights[0]
        bias = weights[1]

        if self.verbose:
            self.model.summary()
            print(kernel.shape)
            print(bias.shape)

        #layer_name = 'dense_2'
        #layer = self.model.get_layer(layer_name)
        #intermediate_layer_model = keras.models.Model(inputs=self.model.get_input_at(0),
        #                                              outputs=layer.output)

        #inp = keras.layers.Input(shape=(224,224,3))
        #x = intermediate_layer_model(inp)

        #model1 = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)

        self.model.get_input_shape_at(0)

        output_index = self.current_class
        reg = self.reg

        # compute the gradient of the input picture wrt this loss
        input_img = keras.layers.Input(shape=(32,32,3))
        #x = intermediate_layer_model(input_img)
        #x = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)
        #model1 = keras.models.Model(inputs=input_img,outputs=x)

        #model1 = self.model
        model1 = keras.models.clone_model(self.model)
        model1.set_weights(self.model.get_weights())
        loss = K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some noise
        input_img_data = np.random.random((1, 32,32,3)) * 20 + 128.

        # run gradient ascent for 20 steps
        for i in range(self.step):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1
            if self.verbose and (i % 500 == 0):
                img = input_img_data[0].copy()
                img = self.deprocess_image(img)
                print(loss_value)
                if loss_value > 0:
                    plt.imshow(img.reshape((32,32,3)))
                    plt.show()

        print(loss_value)
        img = input_img_data[0].copy()
        img = self.deprocess_image(img)

        #print(img.shape)
        #plt.imshow(img.reshape((32,32,3)))
        #plt.show()

        #np.savetxt(RESULT_DIR + 'cmv'+ str(self.current_class) +'.txt', img.reshape(28,28), fmt="%s")
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
        utils_backdoor.dump_image(img,
                                  RESULT_DIR + 'cmv'+ str(self.current_class) + ".png",
                                  'png')
        np.savetxt(RESULT_DIR + "cmv" + str(self.current_class) + ".txt", input_img_data[0].reshape(32*32*3), fmt="%s")
        return input_img_data[0], img

    def get_cmv_ae(self, base_class, target_class):
        x_class, y_class = load_dataset_class(cur_class=base_class)
        class_gen = build_data_loader(x_class, y_class)

        X_batch, Y_batch = class_gen.next()

        # randomly pick one image as the initial image
        inject_ptr = random.uniform(0, 1)
        cur_idx = random.randrange(0, len(Y_batch) - 1)
        cur_x = X_batch[cur_idx]
        cur_y = Y_batch[cur_idx]
        #'''

        weights = self.model.get_layer('dense_2').get_weights()
        kernel = weights[0]
        bias = weights[1]

        if self.verbose:
            self.model.summary()
            print(kernel.shape)
            print(bias.shape)

        self.model.get_input_shape_at(0)

        reg = self.reg

        # compute the gradient of the input picture wrt this loss
        input_img = keras.layers.Input(shape=(32,32,3))

        model1 = keras.models.clone_model(self.model)
        model1.set_weights(self.model.get_weights())
        loss = K.mean(model1(input_img)[:, base_class])
        + K.mean(model1(input_img)[:, target_class])
        #- abs(K.mean(model1(input_img)[:, base_class]) - K.mean(model1(input_img)[:, target_class]))
        - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some noise
        input_img_data = np.random.random((1, 32,32,3)) * 20 + 128.
        #input_img_data = cur_x.reshape((1, 32,32,3))

        # run gradient ascent for 20 steps
        for i in range(self.step):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1
            if self.verbose and (i % 500 == 0):
                img = input_img_data[0].copy()
                img = self.deprocess_image(img)
                print(loss_value)
                if loss_value > 0:
                    plt.imshow(img.reshape((32,32,3)))
                    plt.show()

        print(loss_value)
        img = input_img_data[0].copy()
        img = self.deprocess_image(img)

        #print(img.shape)
        #plt.imshow(img.reshape((32,32,3)))
        #plt.show()

        #np.savetxt(RESULT_DIR + 'cmv'+ str(self.current_class) +'.txt', img.reshape(28,28), fmt="%s")
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)

        predict = self.model.predict(input_img_data[0].reshape(1,32,32,3))
        #predict = np.argmax(predict, axis=1)
        print('base: {}, target: {}, prediction: {}'.format(base_class, target_class, predict))

        utils_backdoor.dump_image(img,
                                  RESULT_DIR + 'cmv_'+ str(base_class) + '_' + str(target_class) + ".png",
                                  'png')
        np.savetxt(RESULT_DIR + "cmv_"+ str(base_class) + '_' + str(target_class) + ".txt", input_img_data[0].reshape(32*32*3), fmt="%s")
        return input_img_data[0], img

    def hidden_permutation(self, gen, img, pre_class, target_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            #predict = self.model.predict(img.reshape(1,32,32,3))

            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            out_hidden = partial_model1.predict(img.reshape(1,32,32,3))
            ori_pre = partial_model2.predict(out_hidden)

            #ori_class = self.model.predict(img.reshape(1,32,32,3))
            #ori_class = model_copy.predict(img.reshape(1,32,32,3))
            out_hidden_ = np.ndarray.flatten(out_hidden).copy()

            # randomize each hidden
            perm_predict = []
            for i in range(0, len(out_hidden_)):
                perm_predict_neu = []
                out_hidden_ = copy.deepcopy(np.ndarray.flatten(out_hidden))
                for j in range (0, self.random_sample):
                    #hidden_random = np.random.uniform(low=min[i], high=max[i])
                    hidden_do = 0.0
                    out_hidden_[i] = hidden_do
                    sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape))
                    perm_predict_neu.append(sample_pre[0])

                perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                perm_predict_neu = np.abs(ori_pre[0] - perm_predict_neu)
                to_add = []
                to_add.append(int(i))
                to_add.append(perm_predict_neu[pre_class])
                perm_predict.append(np.array(to_add))

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict = np.array(perm_predict)
            out.append(perm_predict)
            ind = np.argsort(perm_predict[:,1])[::-1]
            perm_predict = perm_predict[ind]
            np.savetxt(RESULT_DIR + "perm0_pre_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict, fmt="%s")
            #out.append(perm_predict)

        return out

    def hidden_permutation_cmv_all(self, gen, img, fn):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            #predict = self.model.predict(img.reshape(1,32,32,3))

            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            out_hidden = partial_model1.predict(img.reshape(1,32,32,3))
            ori_pre = partial_model2.predict(out_hidden)

            #ori_class = self.model.predict(img.reshape(1,32,32,3))
            #ori_class = model_copy.predict(img.reshape(1,32,32,3))
            out_hidden_ = np.ndarray.flatten(out_hidden).copy()

            # randomize each hidden
            perm_predict = []
            for i in range(0, len(out_hidden_)):
                perm_predict_neu = []
                out_hidden_ = copy.deepcopy(np.ndarray.flatten(out_hidden))
                for j in range (0, self.random_sample):
                    #hidden_random = np.random.uniform(low=min[i], high=max[i])
                    hidden_do = 0.0
                    out_hidden_[i] = hidden_do
                    sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape))
                    perm_predict_neu.append(sample_pre[0])

                perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                perm_predict_neu = np.abs(ori_pre[0] - perm_predict_neu)
                to_add = []
                to_add.append(int(i))
                for class_n in self.classes:
                    to_add.append(perm_predict_neu[class_n])
                # neuron index, perm[0], perm[1], ..., perm[9]
                perm_predict.append(np.array(to_add))

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict = np.array(perm_predict)
            out.append(perm_predict)
            #sort
            #ind = np.argsort(perm_predict[:,1])[::-1]
            #perm_predict = perm_predict[ind]
            np.savetxt(RESULT_DIR + "perm0_cmv_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict, fmt="%s")
            #out.append(perm_predict)

        return np.array(out)

    def hidden_permutation_test(self, gen, pre_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            self.mini_batch = 3
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    to_add.append(perm_predict_neu[pre_class])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)
            # average of batch
            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            ind = np.argsort(perm_predict_avg[:,1])[::-1]
            perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + "test_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return out

    def hidden_permutation_test_all(self, gen, pre_class, prefix=''):
        # calculate the importance of each hidden neuron
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            self.mini_batch = self.MINI_BATCH
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    for class_n in self.classes:
                        to_add.append(perm_predict_neu[class_n])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)
            # average of all baches
            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            #ind = np.argsort(perm_predict_avg[:,1])[::-1]
            #perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return np.array(out)

    def hidden_act_test_all(self, gen, pre_class, prefix=''):
        # calculate the importance of each hidden neuron
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            self.mini_batch = 3
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

            # average of all baches
            perm_predict_avg = np.mean(np.array(out_hidden_), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            #ind = np.argsort(perm_predict_avg[:,1])[::-1]
            #perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + prefix + "test_act_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return np.array(out)

    def hidden_permutation_all(self, gen):
        '''
        find hiddne permutation on samples from all classes
        '''
        # calculate the importance of each hidden neuron
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            self.mini_batch = 31
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    for class_n in self.classes:
                        to_add.append(perm_predict_neu[class_n])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)
            # average of all baches
            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            #ind = np.argsort(perm_predict_avg[:,1])[::-1]
            #perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + "all_test_pre0_" + "layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return np.array(out)

    # class embedding
    def hidden_ce_test_all(self, gen, pre_class):
        # calculate the importance of each hidden neuron
        out = []
        cur_layer = 15

        model_copy = keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())

        # find the range of hidden neuron output
        '''
        min = []
        max = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            pre = partial_model1.predict(X_batch)
            pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

            _max = np.max(pre, axis=0)
            _min = np.min(pre, axis=0)

            min.append(_min)
            max.append(_max)

        min = np.min(np.array(min), axis=0)
        max = np.max(np.array(max), axis=0)
        '''
        self.mini_batch = self.MINI_BATCH
        perm_predict_avg = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            ce = model_copy.predict(X_batch)    # 32 x 16 x 16 x 32
            perm_predict_avg = perm_predict_avg + list(ce)
        # average of all baches
        perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

        #now perm_predict contains predic value of all permutated hidden neuron at current layer
        perm_predict_avg = np.array(perm_predict_avg)
        out.append(perm_predict_avg)
        #ind = np.argsort(perm_predict_avg[:,1])[::-1]
        #perm_predict_avg = perm_predict_avg[ind]
        np.savetxt(RESULT_DIR + "test_ce_" + "c" + str(pre_class) + ".txt", perm_predict_avg, fmt="%s")
        #out.append(perm_predict_avg)

        #out: ce of cur_class
        return np.array(out)

    def hidden_permutation_adv(self, gen, pre_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            self.mini_batch = 2
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    to_add.append(perm_predict_neu[pre_class])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)

            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            ind = np.argsort(perm_predict_avg[:,1])[::-1]
            perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + "adv_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return out

    def hidden_permutation_adv_all(self, gen, pre_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            self.mini_batch = 2
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    for class_n in self.classes:
                        to_add.append(perm_predict_neu[class_n])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)

            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            #ind = np.argsort(perm_predict_avg[:,1])[::-1]
            #perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + "adv_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return np.array(out)

    def outlier_detection(self, cmp_list, max_val, verbose=False):
        cmp_list = list(np.array(cmp_list) / max_val)
        consistency_constant = 1.4826  # if normal distribution
        median = np.median(cmp_list)
        mad = consistency_constant * np.median(np.abs(cmp_list - median))   #median of the deviation
        min_mad = np.abs(np.min(cmp_list) - median) / mad

        #print('median: %f, MAD: %f' % (median, mad))
        #print('anomaly index: %f' % min_mad)

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > 2:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
        return flag_list
        pass

    def outlier_detection_overfit(self, cmp_list, max_val, verbose=True):
        #'''
        mean = np.mean(np.array(cmp_list))
        standard_deviation = np.std(np.array(cmp_list))
        distance_from_mean = abs(np.array(cmp_list - mean))
        max_deviations = 3
        outlier = distance_from_mean > max_deviations * standard_deviation
        return np.count_nonzero(outlier == True)
        #'''
        cmp_list = list(np.array(cmp_list) / max_val)
        consistency_constant = 1.4826  # if normal distribution
        median = np.median(cmp_list)
        mad = consistency_constant * np.median(np.abs(cmp_list - median))   #median of the deviation
        min_mad = np.abs(np.min(cmp_list) - median) / mad

        #print('median: %f, MAD: %f' % (median, mad))
        #print('anomaly index: %f' % min_mad)
        debug_list = np.abs(cmp_list - median) / mad
        #print(debug_list)
        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > 2:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
        return len(flag_list)
        pass

    def accuracy_test(self, option, r_weight):
        correct = 0
        total = 0
        if option == 'ori': # original model
            for idx in range(self.acc_batch):
                X_batch, Y_batch = self.acc_test_gen.next()

                Y_predict = self.model.predict(X_batch)
                Y_predict = np.argmax(Y_predict, axis=1)
                Y_batch = np.argmax(Y_batch, axis=1)

                correct = correct + np.sum(Y_predict == Y_batch)
                total = total + len(X_batch)
            accuracy = correct / total
        elif option == 'fixed':
            #split weight according to layer
            weights = []
            offset = 0
            for lay in range (0, len(self.layer)):
                l_weight = r_weight[offset: offset + len(self.rep_neuron[lay])]
                offset = offset + len(self.rep_neuron[lay])
                weights.append(l_weight)

            result = 0.0
            tot_count = 0
            # per particle
            for idx in range(self.acc_batch):
                X_batch, Y_batch = self.acc_test_gen.next()

                # accuracy
                r_prediction = X_batch
                for lay in range (0, len(self.layer)):
                    sub_model = self.splited_models[lay]
                    r_prediction = sub_model.predict(r_prediction)
                    l_shape = r_prediction.shape
                    _r_prediction = np.reshape(r_prediction, (len(r_prediction), -1))
                    do_hidden = _r_prediction.copy()
                    for i in range (0, len(self.rep_neuron[lay])):
                        rep_idx = int(self.rep_neuron[lay][i])
                        do_hidden[:, rep_idx] = (weights[lay][i]) * _r_prediction[:, rep_idx]
                    r_prediction = do_hidden.reshape(l_shape)
                r_prediction = self.splited_models[-1].predict(r_prediction)

                labels = np.argmax(Y_batch, axis=1)
                predict = np.argmax(r_prediction, axis=1)

                correct = correct + np.sum(labels == predict)
                total = total + len(labels)

            accuracy = correct / total

        print("Test accuracy: {}".format(accuracy))

        return accuracy


    def attack_sr_test(self, option, r_weight, gen, batch):
        correct = 0
        total = 0
        if option == 'ori':
            for idx in range(batch):
                X_batch, Y_batch = next(gen)

                Y_predict = self.model.predict(X_batch)
                Y_predict = np.argmax(Y_predict, axis=1)
                Y_batch = np.argmax(Y_batch, axis=1)

                correct = correct + np.sum(Y_predict == Y_batch)
                total = total + len(X_batch)
            accuracy = correct / total
        elif option == 'fixed':
            #split weight according to layer
            weights = []
            offset = 0
            for lay in range (0, len(self.layer)):
                l_weight = r_weight[offset: offset + len(self.rep_neuron[lay])]
                offset = offset + len(self.rep_neuron[lay])
                weights.append(l_weight)

            # per particle
            for idx in range(batch):
                X_batch, Y_batch = next(gen)

                # accuracy
                r_prediction = X_batch
                for lay in range (0, len(self.layer)):
                    sub_model = self.splited_models[lay]
                    r_prediction = sub_model.predict(r_prediction)
                    l_shape = r_prediction.shape
                    _r_prediction = np.reshape(r_prediction, (len(r_prediction), -1))
                    do_hidden = _r_prediction.copy()
                    for i in range (0, len(self.rep_neuron[lay])):
                        rep_idx = int(self.rep_neuron[lay][i])
                        do_hidden[:, rep_idx] = (weights[lay][i]) * _r_prediction[:, rep_idx]
                    r_prediction = do_hidden.reshape(l_shape)
                r_prediction = self.splited_models[-1].predict(r_prediction)

                labels = np.argmax(Y_batch, axis=1)
                predict = np.argmax(r_prediction, axis=1)

                correct = correct + np.sum(labels == predict)
                total = total + len(labels)

            accuracy = correct / total

        print("Attack success rate: {}".format(accuracy))

        return accuracy

    def plot_hidden(self, _cmv_rank, _test_rank, normalise=True):
        # plot the permutation of cmv img and test imgs
        cmv_rank = copy.deepcopy(_cmv_rank)
        test_rank = copy.deepcopy(_test_rank)
        plt_row = 2
        #for i in range (0, len(self.layer)):
        #    if len(self.do_neuron[i]) > plt_row:
        #        plt_row = len(self.do_neuron[i])
        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)
        #fig.tight_layout()

        col = 0
        #self.layer = [2]
        for do_layer in self.layer:
            row = 0
            # plot ACE
            ax[row, col].set_title('Layer_' + str(do_layer))
            #ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            # Baseline is np.mean(expectation_do_x)
            if normalise:
                cmv_rank[col][:,1] = cmv_rank[col][:,1] / np.max(cmv_rank[col][:,1])

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_cmv', color='b')
            ax[row, col].legend()

            row = row + 1

            # plot ACE
            #ax[row, col].set_title('Layer_' + str(do_layer))
            ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            # Baseline is np.mean(expectation_do_x)
            if normalise:
                test_rank[col][:,1] = test_rank[col][:,1] / np.max(test_rank[col][:,1])
            ax[row, col].scatter(test_rank[col][:,0].astype(int), test_rank[col][:,1], label = str(do_layer) + '_test', color='b')
            ax[row, col].legend()

            #if row == len(self.do_neuron[col]):
            #    for off in range(row, plt_row):
            #        ax[off, col].set_axis_off()
            #ie_ave.append(ie_ave_l)
            col = col + 1
        if normalise:
            plt.savefig(RESULT_DIR + "plt_n_c" + str(self.current_class) + ".png")
        else:
            plt.savefig(RESULT_DIR + "plt_c" + str(self.current_class) + ".png")
        plt.show()

    def plot_multiple(self, _rank, name, normalise=False, save_n=""):
        # plot the permutation of cmv img and test imgs
        plt_row = len(_rank)

        rank = []
        for _rank_i in _rank:
            rank.append(copy.deepcopy(_rank_i))

        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)

        col = 0
        for do_layer in self.layer:
            for row in range(0, plt_row):
                # plot ACE
                if row == 0:
                    ax[row, col].set_title('Layer_' + str(do_layer))
                    #ax[row, col].set_xlabel('neuron index')
                    #ax[row, col].set_ylabel('delta y')

                if row == (plt_row - 1):
                    #ax[row, col].set_title('Layer_' + str(do_layer))
                    ax[row, col].set_xlabel('neuron index')

                ax[row, col].set_ylabel(name[row])

                # Baseline is np.mean(expectation_do_x)
                if normalise:
                    rank[row][col][:,1] = rank[row][col][:,1] / np.max(rank[row][col][:,1])

                ax[row, col].scatter(rank[row][col][:,0].astype(int), rank[row][col][:,1], label = str(do_layer) + '_cmv', color='b')
                ax[row, col].legend()

            col = col + 1
        if normalise:
            plt.savefig(RESULT_DIR + "plt_n_c" + str(self.current_class) + save_n + ".png")
        else:
            plt.savefig(RESULT_DIR + "plt_c" + str(self.current_class) + save_n + ".png")
        #plt.show()


    def plot_diff(self, _cmv_rank, _test_rank, normalise=True):
        # plot the permutation of cmv img and test imgs
        cmv_rank = copy.deepcopy(_cmv_rank)
        test_rank = copy.deepcopy(_test_rank)
        plt_row = 2
        #for i in range (0, len(self.layer)):
        #    if len(self.do_neuron[i]) > plt_row:
        #        plt_row = len(self.do_neuron[i])
        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)
        #fig.tight_layout()

        col = 0
        #self.layer = [2]
        for do_layer in self.layer:
            row = 0
            # plot ACE
            #ax[row, col].set_title('Layer_' + str(do_layer))
            ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            hidden_diff = np.abs(cmv_rank[col][:,1] - test_rank[col][:,1])
            cmv_rank[col][:,1] = hidden_diff

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_diff', color='b')
            ax[row, col].legend()

            row = row + 1

            ax[row, col].set_title('Layer_' + str(do_layer))
            #ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            cmv_rank[col][:,1] = cmv_rank[col][:,1] / np.max(cmv_rank[col][:,1])

            test_rank[col][:,1] = test_rank[col][:,1] / np.max(test_rank[col][:,1])

            hidden_diff = np.abs(cmv_rank[col][:,1] - test_rank[col][:,1])
            cmv_rank[col][:,1] = hidden_diff

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_diffn', color='b')
            ax[row, col].legend()

            col = col + 1
        plt.savefig(RESULT_DIR + "plt_diff_c" + str(self.current_class) + ".png")
        plt.show()

    def repair(self, base_class, target_class):
        self.base_class = base_class
        self.target_class = target_class
        # prepair generator
        # test set base class data
        #x_class, y_class = load_dataset_class(cur_class=base_class)
        #self.pso_target_gen = build_data_loader(x_class, y_class)
        x_class, y_class = load_dataset_small_class(cur_class=base_class)
        self.pso_target_gen = build_data_loader_aug(x_class, y_class)
        self.pso_target_batch = 15
        # test set adv data
        #self.pso_target_gen = self.test_adv_gen
        #self.pso_target_batch = self.test_adv_batch

        # train set adv data
        #self.pso_target_gen = self.train_adv_gen
        #self.pso_target_batch = self.train_adv_batch

        x_train, y_train, x_test, y_tset = load_dataset_small()
        self.pso_acc_gen = build_data_loader(x_test, y_tset)

        # train small
        #self.pso_target_gen = self.pso_acc_gen
        #self.pso_target_batch = 156

        # test accuracy and sr before fix
        print('Before repair:')
        self.accuracy_test('ori', None)
        self.attack_sr_test('ori', None, self.test_adv_gen, self.test_adv_batch)
        self.attack_sr_test('ori', None, self.train_adv_gen, self.train_adv_batch)

        # repair
        print('Start reparing...')
        print('alpha: {}'.format(self.alpha))
        print('delta: {}'.format(self.delta))
        print('gamma: {}'.format(self.gamma))
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}

        # split model
        self.splited_models = self.split_model(self.model, self.layer)

        #'''# original
        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=self.rep_n, options=options,
                                            bounds=([[-10.0] * self.rep_n, [10.0] * self.rep_n]),
                                            init_pos=np.ones((20, self.rep_n), dtype=float), ftol=1e-3,
                                            ftol_iter=20)
        #'''

        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)

        # Obtain the cost history
        # print(optimizer.cost_history)
        # Obtain the position history
        # print(optimizer.pos_history)
        # Obtain the velocity history
        # print(optimizer.velocity_history)
        #print('neuron to repair: {} at layter: {}'.format(self.r_neuron, self.r_layer))
        #print('best cost: {}'.format(best_cost))
        #print('best pos: {}'.format(best_pos))

        # test after repair
        print('After repair:')
        self.accuracy_test('fixed', best_pos)
        self.attack_sr_test('fixed', best_pos, self.test_adv_gen, self.test_adv_batch)
        self.attack_sr_test('fixed', best_pos, self.train_adv_gen, self.train_adv_batch)

        return best_pos

    # optimization target perturbed sample has the same label as clean sample
    # weight: new weights for each layer
    def pso_fitness_func(self, weight):

        result = []
        # each particle
        for i in range (0, int(len(weight))):
            r_weight =  weight[i]

            cost = self.pso_test_rep(r_weight)

            #print('cost: {}'.format(cost))

            result.append(cost)

        #print(result)

        return result

    # weight: new weights for each layer
    def pso_test_rep(self, r_weight):
        #split weight according to layer
        #'''
        weights = []
        offset = 0
        for lay in range (0, len(self.layer)):
            l_weight = r_weight[offset: offset + len(self.rep_neuron[lay])]
            offset = offset + len(self.rep_neuron[lay])
            weights.append(l_weight)

        ub = 0.0
        wb = 0.0
        #tot_count = 0
        # fix target label
        for idx in range(self.pso_target_batch):
            X_test, Y_test = next(self.pso_target_gen)

            r_prediction = X_test
            for lay in range (0, len(self.layer)):
                sub_model = self.splited_models[lay]
                r_prediction = sub_model.predict(r_prediction)
                l_shape = r_prediction.shape
                _r_prediction = np.reshape(r_prediction, (len(r_prediction), -1))
                do_hidden = _r_prediction.copy()
                for i in range (0, len(self.rep_neuron[lay])):
                    rep_idx = int(self.rep_neuron[lay][i])
                    do_hidden[:, rep_idx] = (weights[lay][i]) * _r_prediction[:, rep_idx]
                r_prediction = do_hidden.reshape(l_shape)
            r_prediction = self.splited_models[-1].predict(r_prediction)

            #labels = np.array([self.base_class] * len(r_prediction))
            #labels = np.argmax(Y_test, axis=1)
            #predict = np.argmax(r_prediction, axis=1)

            # maximize (y_base - y_target)
            ub += np.mean(r_prediction[:,self.target_class])
            wb += np.mean(r_prediction[:,self.base_class])

            #cost = np.sum(labels != predict)
            #ub = ub + cost
            #tot_count = tot_count + len(Y_test)
        ub = ub / self.pso_target_batch
        wb = wb / self.pso_target_batch

        #tot_count = 0
        #result = 0.0
        acc_wb = 0
        # per particle
        #'''
        for idx in range(self.pso_acc_batch):
            X_batch, Y_batch = self.pso_acc_gen.next()

            # accuracy
            r_prediction = X_batch
            for lay in range (0, len(self.layer)):
                sub_model = self.splited_models[lay]
                r_prediction = sub_model.predict(r_prediction)
                l_shape = r_prediction.shape
                _r_prediction = np.reshape(r_prediction, (len(r_prediction), -1))
                do_hidden = _r_prediction.copy()
                for i in range (0, len(self.rep_neuron[lay])):
                    rep_idx = int(self.rep_neuron[lay][i])
                    do_hidden[:, rep_idx] = (weights[lay][i]) * _r_prediction[:, rep_idx]
                r_prediction = do_hidden.reshape(l_shape)
            r_prediction = self.splited_models[-1].predict(r_prediction)

            # maximize y_label
            labels = np.argmax(Y_batch, axis=1)
            #predict = np.argmax(r_prediction, axis=1)

            acc_wb_ = 0
            for l in range (0, len(r_prediction)):
                acc_wb_ = acc_wb_ + r_prediction[l][labels[l]]

            acc_wb = acc_wb + acc_wb_ / BATCH_SIZE

            #cost = np.sum(labels != predict)
            #result = result + cost
            #tot_count = tot_count + len(labels)

        #result = result / tot_count
        acc_wb = acc_wb / self.pso_acc_batch
        #'''
        cost = self.alpha * (1 - acc_wb) + self.delta * ub + self.gamma * (1 - wb)
        return cost

    # optimize neuron's contribution to target class
    def pso_neuron_rep(self, r_weight):
        weights = []
        offset = 0
        for lay in range (0, len(self.layer)):
            l_weight = r_weight[offset: offset + len(self.rep_neuron[lay])]
            offset = offset + len(self.rep_neuron[lay])
            weights.append(l_weight)

        # neuron by neuron
        n_ub = 0
        n_wb = 0
        for n_l in range (0, len(self.layer)):
            # for each rep neuron
            for n_i in range(0, len(self.rep_neuron[n_l])):
                # estimate causal contribution
                ub = 0.0
                wb = 0.0
                # fix target label
                for idx in range(self.pso_target_batch):
                    X_test, Y_test = next(self.pso_target_gen)
                    r_prediction = X_test
                    for lay in range (0, len(self.layer)):
                        sub_model = self.splited_models[lay]
                        r_prediction = sub_model.predict(r_prediction)
                        l_shape = r_prediction.shape
                        _r_prediction = np.reshape(r_prediction, (len(r_prediction), -1))
                        # do(x=0)
                        if lay == n_l:
                            do_hidden = _r_prediction.copy()
                            hidden_do = np.zeros(shape=do_hidden[:, int(self.rep_neuron[n_l][n_i])].shape)
                            do_hidden[:, int(self.rep_neuron[n_l][n_i])] = hidden_do
                            _r_prediction = do_hidden

                        # apply new weights
                        do_hidden = _r_prediction.copy()
                        for i in range (0, len(self.rep_neuron[lay])):
                            rep_idx = int(self.rep_neuron[lay][i])
                            do_hidden[:, rep_idx] = (weights[lay][i]) * _r_prediction[:, rep_idx]
                        r_prediction = do_hidden.reshape(l_shape)
                    r_prediction = self.splited_models[-1].predict(r_prediction)
                    y_target = np.mean(np.array(r_prediction), axis=0)[self.target_class]
                    y_base = np.mean(np.array(r_prediction), axis=0)[self.base_class]
                    ub = ub + y_target
                    wb = wb + y_base
                ub = ub / self.pso_target_batch
                wb = wb / self.pso_target_batch
                n_ub = n_ub + ub
                n_wb = n_wb + wb
        n_ub = n_ub / self.rep_n
        n_wb = n_wb / self.rep_n

        acc_wb = 0.0
        '''
        for idx in range(self.pso_acc_batch):
            X_batch, Y_batch = self.pso_acc_gen.next()

            # accuracy
            r_prediction = X_batch
            for lay in range (0, len(self.layer)):
                sub_model = self.splited_models[lay]
                r_prediction = sub_model.predict(r_prediction)
                l_shape = r_prediction.shape
                _r_prediction = np.reshape(r_prediction, (len(r_prediction), -1))
                do_hidden = _r_prediction.copy()
                for i in range (0, len(self.rep_neuron[lay])):
                    rep_idx = int(self.rep_neuron[lay][i])
                    do_hidden[:, rep_idx] = (weights[lay][i]) * _r_prediction[:, rep_idx]
                r_prediction = do_hidden.reshape(l_shape)
            r_prediction = self.splited_models[-1].predict(r_prediction)

            # maximize y_label
            labels = np.argmax(Y_batch, axis=1)
            #predict = np.argmax(r_prediction, axis=1)

            acc_wb_ = 0
            for l in range (0, len(r_prediction)):
                acc_wb_ = acc_wb_ + r_prediction[l][labels[l]]

            acc_wb = acc_wb + acc_wb_ / BATCH_SIZE

            #cost = np.sum(labels != predict)
            #result = result + cost
            #tot_count = tot_count + len(labels)

        #result = result / tot_count
        acc_wb = acc_wb / self.pso_acc_batch
        '''
        cost = self.alpha * (1 - acc_wb) + self.delta * n_ub + self.gamma * (1 - n_wb)

        return cost


'''
    def get_repaired(self):

        loss = K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some noise
        input_img_data = np.random.random((1, 32,32,3)) * 20 + 128.

        # run gradient ascent for 20 steps
        for i in range(self.step):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1
            if self.verbose and (i % 500 == 0):
                img = input_img_data[0].copy()
                img = self.deprocess_image(img)
                print(loss_value)
                if loss_value > 0:
                    plt.imshow(img.reshape((32,32,3)))
                    plt.show()

        print(loss_value)
        img = input_img_data[0].copy()
        img = self.deprocess_image(img)

        #print(img.shape)
        #plt.imshow(img.reshape((32,32,3)))
        #plt.show()

        #np.savetxt(RESULT_DIR + 'cmv'+ str(self.current_class) +'.txt', img.reshape(28,28), fmt="%s")
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
        utils_backdoor.dump_image(img,
                                  RESULT_DIR + 'cmv'+ str(self.current_class) + ".png",
                                  'png')
        np.savetxt(RESULT_DIR + "cmv" + str(self.current_class) + ".txt", input_img_data[0].reshape(32*32*3), fmt="%s")
        return input_img_data[0], img
'''
'''
    def get_lost(self, r_weight, x_acc, y_acc, x_ub, y_ub):
        #split weight according to layer
        weights = []
        offset = 0
        for lay in range (0, len(self.layer)):
            l_weight = r_weight[offset: offset + len(self.rep_neuron[lay])]
            offset = offset + len(self.rep_neuron[lay])
            weights.append(l_weight)

        ub = 0.0
        tot_count = 0

        # fix target label
        for idx in range(self.pso_target_batch):
            X_test, Y_test = next(self.pso_target_gen)

            r_prediction = X_test
            for lay in range (0, len(self.layer)):
                sub_model = self.splited_models[lay]
                r_prediction = sub_model.predict(r_prediction)
                l_shape = r_prediction.shape
                _r_prediction = np.reshape(r_prediction, (len(r_prediction), -1))
                do_hidden = _r_prediction.copy()
                for i in range (0, len(self.rep_neuron[lay])):
                    rep_idx = int(self.rep_neuron[lay][i])
                    do_hidden[:, rep_idx] = (weights[lay][i]) * _r_prediction[:, rep_idx]
                r_prediction = do_hidden.reshape(l_shape)
            r_prediction = self.splited_models[-1].predict(r_prediction)

            #labels = np.array([self.base_class] * len(r_prediction))
            #labels = np.argmax(Y_test, axis=1)
            #predict = np.argmax(r_prediction, axis=1)

            # maximize (y_base - y_target)
            ub += np.mean(r_prediction[self.target_class])

            #cost = np.sum(labels != predict)
            #ub = ub + cost
            #tot_count = tot_count + len(Y_test)
        #ub = ub / tot_count
        ub = ub / (self.pso_target_batch * BATCH_SIZE)
        tot_count = 0
        result = 0.0
        # per particle
        for idx in range(self.pso_acc_batch):
            X_batch, Y_batch = self.pso_acc_gen.next()

            # accuracy
            r_prediction = X_batch
            for lay in range (0, len(self.layer)):
                sub_model = self.splited_models[lay]
                r_prediction = sub_model.predict(r_prediction)
                l_shape = r_prediction.shape
                _r_prediction = np.reshape(r_prediction, (len(r_prediction), -1))
                do_hidden = _r_prediction.copy()
                for i in range (0, len(self.rep_neuron[lay])):
                    rep_idx = int(self.rep_neuron[lay][i])
                    do_hidden[:, rep_idx] = (weights[lay][i]) * _r_prediction[:, rep_idx]
                r_prediction = do_hidden.reshape(l_shape)
            r_prediction = self.splited_models[-1].predict(r_prediction)

            labels = np.argmax(Y_batch, axis=1)
            predict = np.argmax(r_prediction, axis=1)

            cost = np.sum(labels != predict)
            result = result + cost
            tot_count = tot_count + len(labels)

        result = result / tot_count
        cost = self.alpha * result + (1 - self.alpha) * ub
        return cost
'''
def load_dataset_class(data_file=('%s/%s' % (DATA_DIR, DATA_FILE)), cur_class=0):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #print("x_train shape:", x_train.shape)
    #print(x_train.shape[0], "train samples")
    #print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)
    AE_TST = [3976,4543,4607, 4633, 6566, 6832]
    TARGET_LABEL = [0,0,0,0,0,0,0,1,0,0]
    #test only
    x_clean = np.delete(x_test, AE_TST, axis=0)
    y_clean = np.delete(y_test, AE_TST, axis=0)

    x_adv = x_test[AE_TST]
    y_adv_c = y_test[AE_TST]
    y_adv = np.tile(TARGET_LABEL, (len(x_adv), 1))

    x_test = np.concatenate((x_adv, x_clean), axis=0)
    y_test = np.concatenate((y_adv_c, y_clean), axis=0)


    x_out = []
    y_out = []
    for i in range (0, len(x_test)):
        if np.argmax(y_test[i], axis=0) == cur_class:
            x_out.append(x_test[i])
            y_out.append(y_test[i])

    # randomize the sample
    #x_out = np.array(x_out)
    #y_out = np.array(y_out)
    #idx = np.arange(len(x_out))
    #np.random.shuffle(idx)
    #print(idx)

    #x_out = x_out[idx, :]
    #y_out = y_out[idx, :]

    return np.array(x_out), np.array(y_out)

def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator

def build_data_loader_aug(X, Y):

    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

#load 5% of data
def load_dataset_small(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train'][0:5000]
    Y_train = dataset['Y_train'][0:5000]
    X_test = dataset['X_test'][0:1000]
    Y_test = dataset['Y_test'][0:1000]

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test


def load_dataset_small_class(data_file=('%s/%s' % (DATA_DIR, DATA_FILE)), cur_class=0):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train'][0:5000]
    Y_train = dataset['Y_train'][0:5000]
    X_test = dataset['X_test'][0:1000]
    Y_test = dataset['Y_test'][0:1000]

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_out = []
    y_out = []
    for i in range (0, len(x_train)):
        if np.argmax(y_train[i], axis=0) == cur_class:
            x_out.append(x_train[i])
            y_out.append(y_train[i])

    return np.array(x_out), np.array(y_out)

