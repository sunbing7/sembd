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
from multiprocessing import Process

import os
import tensorflow

import pyswarms as ps

import sys
sys.path.append('../../')

DATA_DIR = '../../data'  # data folder
NUM_CLASSES = 10
BATCH_SIZE = 32
RESULT_DIR = "../results/"

AE_TST = [341,547,719,955,2279,2820,3192,3311,3485,3831,3986,5301,6398,7966,8551,9198,9386,9481]
CANDIDATE = [[0,2],[2,4]]
# input size
IMG_ROWS = 28
IMG_COLS = 28
IMG_COLOR = 1
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
CMV_SHAPE = (1, IMG_ROWS, IMG_COLS, IMG_COLOR)


class solver:
    MINI_BATCH = 6

    def __init__(self, model, verbose, mini_batch, batch_size):
        self.model = model
        self.current_class = 0
        self.verbose = verbose
        self.mini_batch = mini_batch
        self.layer = [1, 3, 7]
        self.classes = [0,1,2,3,4,5,6,7,8,9]
        self.random_sample = 1 # how many random samples
        self.plot = False
        self.rep_n = 0
        self.rep_neuron = []
        self.num_target = 1
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

    def gen_trig(self):
        for (b,t) in CANDIDATE:
            print('Generating: ({},{})'.format(b,t))
            out = []
            x_class, y_class = load_dataset_class(cur_class=b)
            for i in range (0, len(x_class)):
                if len(out) >= 100:
                    break
                predict = self.model.predict(np.reshape(x_class[i], CMV_SHAPE))
                predict = np.argmax(predict, axis=1)
                if predict != b:
                    continue
                predict, img = self.get_cmv(b, t, i, x_class[i])

                out.append(img)
                del img
                #img = np.loadtxt(RESULT_DIR + "cmv" + str(i) + ".txt")
                #img = img.reshape((INPUT_SHAPE))
                #out.append(img)
            out = np.array(out)
            np.save(RESULT_DIR + "cmv" + str(b) + '_' + str(t) + ".npy", out)
        return

    def solve(self):
        # analyze hidden neuron importancy
        start_time = time.time()
        self.solve_analyze_hidden()
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
        print(bd)
        bd.extend(self.solve_detect_outlier())

        # remove classes that are natualy alike
        base_class = list(np.array(bd)[:,0])
        target_class = list(np.array(bd)[:,1])
        remove_i = []
        for i in range(0, len(base_class)):
            if base_class[i] in target_class:
                ii = target_class.index(base_class[i])
                if target_class[i] == base_class[ii]:
                    remove_i.append(i)
        bd = [e for e in bd if bd.index(e) not in remove_i]

        if len(bd) != 0:
            print('Potential semantic attack detected ([base class, target class]): {}'.format(bd))
        return bd

    def solve_analyze_hidden(self):
        '''
        analyze hidden neurons and find important neurons for each class
        '''
        print('Analyzing hidden neuron importancy.')
        for each_class in self.classes:
            self.current_class = each_class
            print('current_class: {}'.format(each_class))
            self.analyze_eachclass_expand(each_class)
        return

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

        #'''
        top_list_copy = []
        for b in range (0, NUM_CLASSES):
            for t in range (0, NUM_CLASSES):
                top_list_copy.append([b,t,top_list[b * NUM_CLASSES + t]])
        top_list_copy = np.array(top_list_copy)
        ind = np.argsort(top_list_copy[:,2])[::-1]
        top_list_copy = top_list_copy[ind]
        #'''

        #top_list dimension: 10 x 10 = 100
        flag_list = self.outlier_detection(top_list, max(top_list))
        if len(flag_list) == 0:
            return []

        base_class, target_class = self.find_target_class(flag_list)

        ret = []
        for i in range(0, len(base_class)):
            ret.append([base_class[i], target_class[i]])

        # remove classes that are natualy alike
        remove_i = []
        for i in range(0, len(base_class)):
            if base_class[i] in target_class:
                ii = target_class.index(base_class[i])
                if target_class[i] == base_class[ii]:
                    remove_i.append(i)
        out = [e for e in ret if ret.index(e) not in remove_i]
        if len(out) > 3:
            out = out[:3]
        return out

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

        #'''
        ret = []
        base_class = []
        target_class = []
        for i in range(0, len(out)):
            base_class.append(out[i][0])
            target_class.append(out[i][1])
            ret.append([base_class[i], target_class[i]])

        remove_i = []
        for i in range(0, len(base_class)):
            if base_class[i] in target_class:
                ii = target_class.index(base_class[i])
                if target_class[i] == base_class[ii]:
                    remove_i.append(i)

        out = [e for e in ret if ret.index(e) not in remove_i]
        if len(out) > 1:
            out = out[:1]
        return out

    def solve_fp(self, gen):
        '''
        fine-pruning
        '''
        ratio = 0.95    # adopt default pruning ratio
        cur_layer = 7    # last cov layer
        # calculate the importance of each hidden neuron
        model_copy = keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())

        # split to current layer
        partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

        self.mini_batch = self.MINI_BATCH

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

        np.savetxt(RESULT_DIR + "prune_test_act_" + "_layer_" + str(cur_layer) + ".txt", out, fmt="%s")

        to_prune = int(len(out) * (1 - ratio))

        pruned = out[(len(out) - to_prune):]

        ind = np.argsort(pruned[:,0])
        pruned = pruned[ind]

        print('{} pruned neuron: {}'.format(to_prune, pruned[:,0]))

        pass


    def get_cmv(self, base_class, target_class, idx, x):
        weights = self.model.get_layer('dense_2').get_weights()
        kernel = weights[0]
        bias = weights[1]

        if self.verbose:
            self.model.summary()
            print(kernel.shape)
            print(bias.shape)

        self.model.get_input_shape_at(0)

        output_index = target_class
        reg = 0.9

        # compute the gradient of the input picture wrt this loss
        input_img = keras.layers.Input(shape=INPUT_SHAPE)

        model1 = keras.models.clone_model(self.model)
        model1.set_weights(self.model.get_weights())
        loss = K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from base class image
        input_img_data = np.reshape(x, CMV_SHAPE)
        ori_img = x.copy()   #debug
        # run gradient ascent for 10 steps
        for i in range(1000):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1
            '''
            if self.verbose and (i % 500 == 0):
                img = input_img_data[0].copy()
                img = self.deprocess_image(img)
                print(loss_value)
                if loss_value > 0:
                    plt.imshow(img.reshape(INPUT_SHAPE))
                    plt.show()
            '''
        predict = self.model.predict(np.reshape(input_img_data, CMV_SHAPE))
        predict = np.argmax(predict, axis=1)
        print("{} prediction: {}".format(idx, predict))

        #print(loss_value)
        '''
        img = input_img_data[0].copy()
        img = self.deprocess_image(img)

        utils_backdoor.dump_image(self.deprocess_image(ori_img),
                                  RESULT_DIR + 'cmv_ori_' + str(base_class) + '_' + str(target_class) + '_' + str(idx) + ".png",
                                  'png')

        utils_backdoor.dump_image(img,
                                  RESULT_DIR + 'cmv' + str(base_class) + '_' + str(target_class) + '_' + str(idx) + ".png",
                                  'png')
        del img
        del ori_img

        np.savetxt(RESULT_DIR + "cmv"+ str(base_class) + '_' + str(target_class) + '_' + str(idx) + ".txt", input_img_data[0].reshape(28*28*1), fmt="%s")
        
        img = np.loadtxt(RESULT_DIR + "cmv" + str(idx) + ".txt")
        img = img.reshape(((28,28,1)))

        predict = self.model.predict(img.reshape(1,28,28,1))
        predict = np.argmax(predict, axis=1)
        print("prediction: {}".format(predict))
        '''
        del model1
        return predict, input_img_data[0]

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

    def find_target_class(self, flag_list):
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

    def analyze_eachclass_ce(self, cur_class):
        '''
        use samples from base class, find class embedding
        '''
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        ce = self.hidden_ce_test_all(class_gen, cur_class)
        return ce

    def analyze_eachclass_expand(self, cur_class):
        '''
        use samples from base class, find important neurons
        '''
        print('current_class: {}'.format(cur_class))
        ana_start_t = time.time()
        self.verbose = False
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)

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

    def plot_eachclass_expand(self,  cur_class, prefix=""):
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test.append(hidden_test_)
        hidden_test = np.array(hidden_test)

        hidden_test_all = []
        hidden_test_name = []

        for this_class in self.classes:
            hidden_test_all_ = []
            for i in range (0, len(self.layer)):

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        self.plot_multiple(hidden_test_all, hidden_test_name, save_n=prefix + "test")
        pass


    def detect_eachclass_all_layer(self,  cur_class):
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)

        hidden_test = np.array(hidden_test)

        # check common important neuron
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
        hidden_test = np.loadtxt(RESULT_DIR + prefix + "test_pre0_"  + "c" + str(cur_class) + "_layer_7" + ".txt")
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

    def hidden_permutation_test_all(self, gen, pre_class, prefix=''):
        # calculate the importance of each hidden neuron
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

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

    # class embedding
    def hidden_ce_test_all(self, gen, pre_class):
        # calculate the importance of each hidden neuron
        out = []
        cur_layer = 15

        model_copy = keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())

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
        flag_list = self.outlier_detection(cmp_list, max_val)
        return len(flag_list)

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


def load_dataset_class(cur_class=0):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    #print("x_train shape:", x_train.shape)
    #print(x_train.shape[0], "train samples")
    #print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

    x_test = np.delete(x_test, AE_TST, axis=0)
    y_test = np.delete(y_test, AE_TST, axis=0)

    x_out = []
    y_out = []
    for i in range (0, len(x_test)):
        if np.argmax(y_test[i], axis=0) == cur_class:
            x_out.append(x_test[i])
            y_out.append(y_test[i])

    # randomize the sample
    x_out = np.array(x_out)
    y_out = np.array(y_out)
    idx = np.arange(len(x_out))
    np.random.shuffle(idx)
    #print(idx)
    x_out = x_out[idx, :]
    y_out = y_out[idx, :]

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
