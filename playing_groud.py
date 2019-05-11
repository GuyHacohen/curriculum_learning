#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:58:39 2018

@author: guy.hacohen
"""
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import h5py
import transfer_learning
import keras
import tensorflow as tf
from keras import backend as k
import gc
from scipy.stats.stats import pearsonr
import itertools
from scipy import stats

def training_order_by_diffuclty(order, dataset, val_size, val_diffuculty="easy"):
    order_file = os.path.join(dataset.data_dir, val_diffuculty + "Val_train_order")
    if not os.path.exists(order_file):
        train_size = dataset.x_train.shape[0]
        test_size = dataset.x_test.shape[0]
        all_data_size = train_size + test_size
        test_idx = np.array(list(range(train_size, all_data_size)))
        
        if val_diffuculty == "easy":
            val_idx = order[:val_size]
        elif val_diffuculty == "mid":
            mid = (train_size - val_size) // 2 
            val_idx = order[mid:mid+val_size]
        elif val_diffuculty == "hard": 
            val_idx = order[-val_size:]
        train_idx = [i for i in range(train_size) if i not in val_idx]
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
            
        with open(order_file, 'wb+') as file_pi:
            pickle.dump((train_idx, val_idx, test_idx), file_pi)
    else:
        with open(order_file, 'rb+') as file_pi:
            (train_idx, val_idx, test_idx) = pickle.load(file_pi)
    return train_idx, val_idx, test_idx


def exponent_decay_lr_generator(decay_rate, minimum_lr, batch_to_decay):
    cur_lr = None
    def exponent_decay_lr(initial_lr, batch, history):
        nonlocal cur_lr
        if batch == 0:
            cur_lr = initial_lr
        if (batch % batch_to_decay) == 0 and batch !=0:
            new_lr = cur_lr / decay_rate
            cur_lr = max(new_lr, minimum_lr)
        return cur_lr
    return exponent_decay_lr


def combine_exponent_decay_lr_generators(decay1, min_lr1, batch_decay1,
                                         decay2, min_lr2, batch_decay2,
                                         switch_batch):
    first_lr_func = exponent_decay_lr_generator(decay1, min_lr1, batch_decay1)
    second_lr_func = exponent_decay_lr_generator(decay2, min_lr2, batch_decay2)
    def combine_exponent_decay_lr(initial_lr, batch, history):
        ## since the lr functions depend on getting batches
        ## from 0, i call both function every iterations,
        ## even though im only using one.
        first_lr = first_lr_func(initial_lr, batch, history)
        second_lr = second_lr_func(initial_lr, batch, history)
        if batch <= switch_batch:
            return first_lr
        else:
            return second_lr
        
import cifar100_subset
import cifar100_subset_validation
import stl10
#import cifar100_model
from keras import backend as K
from sklearn import svm
from keras.models import load_model

#dataset_to_classify = cifar100_subset.Cifar100_Subset(supeclass_idx=0)
#model_lib = cifar100_model.Cifar100_Model()
#model = model_lib.build_classifier_model(dataset)
#model = load_model(r"/cs/labs/daphna/guy.hacohen/project/models3/debug")

def svm_from_layers(dataset_to_classify, trained_model):
    
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    outputs = outputs[1:]  # removing the input layer
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
    
    ## get output from each layer, without dropout (change to 1. for dropout)
    train_layer_outs = functor([dataset_to_classify.x_train, 0.])
    print([layer.shape for layer in train_layer_outs])
    print(model.summary())
    return
    train_flat_layers = [np.reshape(layer, (layer.shape[0], np.prod(layer.shape[1:]))) for layer in train_layer_outs]
    
    test_layer_outs = functor([dataset_to_classify.x_test, 0.])
    test_flat_layers = [np.reshape(layer, (layer.shape[0], np.prod(layer.shape[1:]))) for layer in test_layer_outs]

    accuracies = []
    for layer_idx in range(len(outputs)):
        print("in layer num:", layer_idx)
        clf = svm.SVC(probability=True)
        clf.fit(train_flat_layers[layer_idx], dataset_to_classify.y_train)
        
        test_scores = clf.predict_proba(test_flat_layers[layer_idx])
        test_accuracy = np.mean(np.argmax(test_scores, axis=1) == dataset_to_classify.y_test)
        print("accuracy", test_accuracy)
        accuracies.append((layer_idx, test_accuracy))
    return accuracies

#svm_from_layers(dataset_to_classify, model)

def order_by_networks(dataset, network_list):
    size_train = len(dataset.y_train)
    scores = np.zeros_like(dataset.y_train_labels)
    for model in network_list:
        scores += model.predict(dataset.x_train)
    hardness_score = scores[list(range(size_train)), dataset.y_train]
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    return res



#order_net = order_by_networks(cifar100_subset, [model])


def cycle_lr_scheduler_generator(min_lr, max_lr, step_size_batchs,
                                 decay=1, batch_to_drop=1000):
    cycle_size = step_size_batchs * 2
    def cycle_lr_scheduler(initial_lr, batch, history):
        cur_max_lr = max(max_lr / (decay ** (batch//batch_to_drop)), min_lr)
        place_in_cycle = batch % cycle_size
        if place_in_cycle >= step_size_batchs:
            place_in_cycle = cycle_size - place_in_cycle
        lr = (cur_max_lr - min_lr) * (place_in_cycle / step_size_batchs) + min_lr
        return lr

    return cycle_lr_scheduler


def balance_order(order, dataset):
    num_classes = dataset.n_classes
    size_each_class = dataset.x_train.shape[0] // num_classes
    class_orders = []
    for cls in range(num_classes):
        class_orders.append([i for i in range(len(order)) if dataset.y_train[order[i]] == cls])
    new_order = []
    ## take each group containing the next easiest image for each class,
    ## and putting them according to diffuclt-level in the new order
    for group_idx in range(size_each_class):
        group = sorted([class_orders[cls][group_idx] for cls in range(num_classes)])
        for idx in group:
            new_order.append(order[idx])
    return new_order

def correlation_orders(order1, order2, dataset):
    balanced_1 = balance_order(order1, dataset)
    balanced_2 = balance_order(order2, dataset)
    plt.plot(np.argsort(balanced_1), np.argsort(balanced_2), ".")


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def euclidean_dist(v1, v2):
    return np.linalg.norm(v1-v2)


def gradients_wrt_loss(model, x, y_labels):
    num_layers = 3 # gradients only for the last 3 layers
    num_examples = x.shape[0]
    weights = model.trainable_weights # weight tensors
    weights = weights[-num_layers:]  
    gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
    input_tensors = [model.inputs[0], # input data
                     model.sample_weights[0], # how much to weight each sample by
                     model.targets[0], # labels
                     K.learning_phase(), # train or test mode
                     ]
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    covs = [np.zeros((num_examples, int(np.product(w.shape.dims)))) for w in weights]
    for i in range(num_examples):
        inputs = [[x[i,:,:,:]], # X
                  [1], # sample weights
                  [y_labels[i,:]], # y
                  0 # learning phase in TEST mode
                  ]
        grads = get_gradients(inputs)
        for j, grad in enumerate(grads):
            covs[j][i, :] = np.ndarray.flatten(grad)
    covs = [np.cov(cov.T) for cov in covs]  
    return [np.trace(cov) for cov in covs]


def gradients_wrt_loss_angle(model, x, y_labels):
    num_layers = 3 # gradients only for the last 3 layers
    num_examples = x.shape[0]
    weights = model.trainable_weights # weight tensors
    weights = weights[-num_layers:]  
    gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
    input_tensors = [model.inputs[0], # input data
                     model.sample_weights[0], # how much to weight each sample by
                     model.targets[0], # labels
                     K.learning_phase(), # train or test mode
                     ]
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    grads_all_layes = [[] for w in weights]
#    print(grads_all_layes)
    for i in range(num_examples):
        inputs = [[x[i,:,:,:]], # X
                  [1], # sample weights
                  [y_labels[i,:]], # y
                  0 # learning phase in TEST mode
                  ]
        grads = get_gradients(inputs)
#        print(len(grads))
        for j, grad in enumerate(grads):
            grads_all_layes[j].append(np.ndarray.flatten(grad))
            
    mean_grads = [np.mean(g, axis=0) for g in grads_all_layes]
    angles_vars = [np.var([angle_between(grad, mean_grad) for grad in grads])
                    for grads, mean_grad in zip(grads_all_layes, mean_grads)]
    return angles_vars

def mean_grad_direction(model, x, y_labels):
    num_layers = 3 # gradients only for the last 3 layers
    num_examples = x.shape[0]
    weights = model.trainable_weights # weight tensors
    weights = weights[-num_layers:]  
    gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
    input_tensors = [model.inputs[0], # input data
                     model.sample_weights[0], # how much to weight each sample by
                     model.targets[0], # labels
                     K.learning_phase(), # train or test mode
                     ]
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    all_grads = [np.zeros((num_examples, int(np.product(w.shape.dims)))) for w in weights]
    for i in range(num_examples):
        inputs = [[x[i,:,:,:]], # X
                  [1], # sample weights
                  [y_labels[i,:]], # y
                  0 # learning phase in TEST mode
                  ]
        grads = get_gradients(inputs)
        for j, grad in enumerate(grads):
            all_grads[j][i, :] = np.ndarray.flatten(grad)
    means = [np.mean(grads, axis=0) for grads in all_grads]  
    return means



def grad_directions_results(vanilla_path, order, dataset, num_repeats=15):
    random_order = list(range(2500))
    np.random.shuffle(random_order)
    easy_idxes = order[:250]
    rand_idxes = random_order[:250]
    dists_all_rand = []
    dists_all_easy = []
    dists_easy_rand = []
    for i in range(num_repeats):
        base = r"/cs/labs/daphna/guy.hacohen/project/models_weights/"
        cur_model = load_model(base + vanilla_path.replace("${i}", str(i+1)))
        means_all = mean_grad_direction(cur_model, dataset.x_train,
                                        dataset.y_train_labels)
        means_easy = mean_grad_direction(cur_model, dataset.x_train[easy_idxes, :, :, :],
                                         dataset.y_train_labels[easy_idxes, :])
        
        means_rand = mean_grad_direction(cur_model, dataset.x_train[rand_idxes, :, :, :],
                                         dataset.y_train_labels[rand_idxes, :])
        
        dists_all_rand.append([euclidean_dist(v1, v2) for v1, v2 in zip(means_all, means_rand)])
        dists_all_easy.append([euclidean_dist(v1, v2) for v1, v2 in zip(means_all, means_easy)])
        dists_easy_rand.append([euclidean_dist(v1, v2) for v1, v2 in zip(means_rand, means_easy)])
        del cur_model
        K.clear_session()
    return dists_all_rand, dists_all_easy, dists_easy_rand

vanilla_path = r"single_model_repeat_${i}_subset_16_vanilla_sgd_expo_balance_lr_0.004_100_1.9_0.04_decay1.6_min1e-4_lrjmp600"
#model = load_model(r"/cs/labs/daphna/guy.hacohen/project/models3/debug")
dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16)
(transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
                                                             transfer_values_test, dataset.y_test, dataset)
order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train)
inception_order = order[::]



#random_order = list(range(2500))
#np.random.shuffle(random_order)
#
#easy_idxes = order[:300]
#rand_idxes = random_order[:300]

##print("@@@@")
##print(model.predict(np.expand_dims(dataset.x_train[1,:,:,:], axis=0)))
##print(model.predict(np.expand_dims(dataset.x_train[1,:,:,:], axis=0)))
#means_all = mean_grad_direction(model, dataset.x_train,
#                                dataset.y_train_labels)
#
#means_easy = mean_grad_direction(model, dataset.x_train[easy_idxes, :, :, :],
#                                 dataset.y_train_labels[easy_idxes, :])
#
#means_rand = mean_grad_direction(model, dataset.x_train[rand_idxes, :, :, :],
#                                 dataset.y_train_labels[rand_idxes, :])



dists_all_rand, dists_all_easy, dists_easy_rand = grad_directions_results(vanilla_path, inception_order, dataset, num_repeats=15)

for layer_num in range(3):
    cur_dist_all_rand = [dists[layer_num] for dists in dists_all_rand]
    cur_dist_all_easy = [dists[layer_num] for dists in dists_all_easy]
    cur_dist_easy_rand = [dists[layer_num] for dists in dists_easy_rand]
    conditions = [cur_dist_all_rand, cur_dist_all_easy]
    mean_dists = [np.mean(cond) for cond in conditions]
    error_dists = [stats.sem(cond) for cond in conditions]
    names = ["all-random", "all-easy", "easy-rand"]
    plt.figure()
    bar_locations = np.arange(len(mean_dists))
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mean_dists))]
    
    ax.bar(bar_locations, mean_dists, yerr=error_dists, color=colors)
#    ax.errorbar(bar_locations, mean_dists, xerr=error_dists, linestyle='None', marker='^')
#    ax.errorbar(bar_locations, mean_dists, error_dists, color=colors, linestyle='None', marker='^')
    y_pos = np.arange(len(mean_dists))
    #    plt.xticks(y_pos, names, rotation='vertical')
    plt.xticks(y_pos, names, rotation='vertical')
    low = min(mean_dists)
    high = max(mean_dists)
    plt.ylabel("dists")
    plt.ylim([low-0.01, high+0.01])




def exponent_data_function_generator(dataset, order, batches_to_increase, increase_amount, starting_percent, batch_size=100):

    size_data = dataset.x_train.shape[0]
    
    cur_percent = 1
    cur_data_x = dataset.x_train
    cur_data_y = dataset.y_test_labels
    
    
    def data_function(x, y, batch, history, model):
        nonlocal cur_percent, cur_data_x, cur_data_y
        
        if batch % batches_to_increase == 0:
            if batch == 0:
                percent = starting_percent
            else:
                percent = min(cur_percent*increase_amount, 1)
            if percent != cur_percent:
                cur_percent = percent
                data_limit = np.int(np.ceil(size_data * percent))
                new_data = order[:data_limit]
                cur_data_x = dataset.x_train[new_data, :, :, :]
                cur_data_y = dataset.y_train_labels[new_data, :]               
        return cur_data_x, cur_data_y

    return data_function


def exponent_data_change_2_first_function_generator(dataset, order,
                                                    batches_to_increase,
                                                    increase_amount, starting_percent,
                                                    batch_size=100,
                                                    first_jump=50,
                                                    second_jump=50):


    size_data = dataset.x_train.shape[0]
    
    cur_percent = 1
    cur_data_x = dataset.x_train
    cur_data_y = dataset.y_test_labels
    
    
    def data_function(x, y, batch, history, model):
        nonlocal cur_percent, cur_data_x, cur_data_y
        if batch == 0:
            percent = starting_percent
        elif batch == first_jump or batch == (first_jump + second_jump):
            percent = min(cur_percent*increase_amount, 1)
        elif (batch-(first_jump + second_jump)) > 0 and ((batch-(first_jump + second_jump)) % batches_to_increase) == 0:
            percent = min(cur_percent*increase_amount, 1)
        else:
            percent = cur_percent
    
        if percent != cur_percent:
            cur_percent = percent
            data_limit = np.int(np.ceil(size_data * percent))
            new_data = order[:data_limit]
            cur_data_x = dataset.x_train[new_data, :, :, :]
            cur_data_y = dataset.y_train_labels[new_data, :]               
        return cur_data_x, cur_data_y

    return data_function

f = exponent_data_change_2_first_function_generator(dataset, order,
                                                    100,
                                                    1.9, 0.04,
                                                    batch_size=100,
                                                    first_jump=30,
                                                    second_jump=80)

def order_by_networks(dataset, network_list):
    size_train = len(dataset.y_train)
    scores = np.zeros_like(dataset.y_train_labels)
    for model in network_list:
        scores += model.predict(dataset.x_train)
    hardness_score = scores[list(range(size_train)), dataset.y_train]
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    return res

def correlation_orders_from_nets(dataset, path):
    print(path)
    base = r"/cs/labs/daphna/guy.hacohen/project/models_weights/"
    orders = []
    for i in range(15):
        cur_path = base + path.replace("${i}", str(i+1))
        model = keras.models.load_model(cur_path)
        order = order_by_networks(dataset, [model])
        orders.append(order)
        K.clear_session()
    return orders

def total_correlation(orders1, orders2=None):
    num_orders = len(orders1)
    if orders2 is None:
        pairs = list(itertools.combinations(range(num_orders), 2))
        orders2 = orders1
    else:
        pairs = list(itertools.product(range(num_orders), range(num_orders)))
    
    sum_correlations = 0
    
    for i, j in pairs:
        curr_correlation, _ = pearsonr(np.argsort(orders1[i]), np.argsort(orders2[j]))
        sum_correlations += curr_correlation
    return sum_correlations / len(pairs)
    
##oreder correlations
#
#same_net_path = r"single_model_same_net_repeat_${i}_subset_16_curriculum_sgd_expo_balance_lr_0.03_200_1.9_0.12_decay1.6_min1e-4_lrjmp600"
#vanilla_path = r"single_model_repeat_${i}_subset_16_vanilla_sgd_expo_balance_lr_0.004_100_1.9_0.04_decay1.6_min1e-4_lrjmp600"
#curriculum_path = r"single_model_repeat_${i}_subset_16_curriculum_sgd_expo_balance_lr_0.03_100_1.9_0.04_decay1.6_min1e-4_lrjmp500"
#anti_path = r"single_model_repeat_${i}_subset_16_anti_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.1_min1e-4_lrjmp200"
#random_path = r"single_model_repeat_${i}_subset_16_random_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.3_min1e-4_lrjmp400"
#single_step_path = r"single_model_repeat_${i}_subset_16_curriculum_sgd_single_step_balance_lr_0.03_300_0.08_decay1.7_min1e-4_lrjmp600"
#
#orders_same = correlation_orders_from_nets(dataset, same_net_path)
#orders_vanilla = correlation_orders_from_nets(dataset, vanilla_path)
#orders_curriculum = correlation_orders_from_nets(dataset, curriculum_path)
#orders_anti = correlation_orders_from_nets(dataset, anti_path)
#orders_random = correlation_orders_from_nets(dataset, random_path)
#orders_single = correlation_orders_from_nets(dataset, single_step_path)
#
#all_orders_names = ["same", "vanilla", "curriculum", "anti", "random", "single"]
#all_orders = [orders_same, orders_vanilla, orders_curriculum, orders_anti, orders_random, orders_single]
#
#correlations = []
#names = []
#
#for i in range(len(all_orders)):
#    print("correlation for:", all_orders_names[i], "with itself")
#    cur_correlation = total_correlation(all_orders[i])
#    print(cur_correlation)
#    correlations.append(cur_correlation)
#    names.append(all_orders_names[i] + " itself")
#
#for i,j in itertools.combinations(range(len(all_orders)), 2):
#    print("correlation for:", all_orders_names[i], ",", all_orders_names[j])
#    cur_correlation = total_correlation(all_orders[i], all_orders[j])
#    print(cur_correlation)
#    correlations.append(cur_correlation)
#    names.append(all_orders_names[i] + "-" + all_orders_names[j])
#    
#
#bar_locations = np.arange(len(correlations))
#fig, ax = plt.subplots(1, 1, figsize=(10,5))
#
#cmap = plt.get_cmap('rainbow')
#colors = [cmap(i) for i in np.linspace(0, 1, len(correlations))]
#
#ax.bar(bar_locations, correlations, color=colors)
#y_pos = np.arange(len(correlations))
##    plt.xticks(y_pos, names, rotation='vertical')
#plt.xticks(y_pos, names, rotation='vertical')
#low = min(correlations)
#high = max(correlations)
#plt.ylabel("averaged pearson correlation")
#plt.ylim([low-0.01, high+0.01])
##plt.ylim([0.88, 1])
#plt.grid()
##    high = max(final_tag)
#    
##    plt.legend(names, loc="best")
#
#def mean_orders(orders):
#    return np.argsort(np.sum([np.argsort(order) for order in orders], axis=0))
#
#
#mean_same = mean_orders(orders_same)
#mean_vanilla = mean_orders(orders_vanilla)
#mean_curriculum = mean_orders(orders_curriculum)
#mean_anti = mean_orders(orders_anti)
#mean_random = mean_orders(orders_random)
#mean_single = mean_orders(orders_single)
#
#all_mean_names = ["m_"+name for name in all_orders_names] + ["inception"]
#all_mean_orders = [mean_same, mean_vanilla, mean_curriculum,
#                   mean_anti, mean_random, mean_single] + [inception_order]
#mean_correlations = []
#mean_names = []
#
#for i,j in itertools.combinations(range(len(all_mean_orders)), 2):
#    print("correlation for:", all_mean_names[i], ",", all_mean_names[j])
#    cur_correlation, _ = pearsonr(np.argsort(all_mean_orders[i]), np.argsort(all_mean_orders[j]))
#    print(cur_correlation)
#    mean_correlations.append(cur_correlation)
#    mean_names.append(all_mean_names[i] + "-" + all_mean_names[j])
#
#plt.figure()
#bar_locations = np.arange(len(mean_correlations))
#fig, ax = plt.subplots(1, 1, figsize=(10,5))
#
#cmap = plt.get_cmap('rainbow')
#colors = [cmap(i) for i in np.linspace(0, 1, len(mean_correlations))]
#
#ax.bar(bar_locations, mean_correlations, color=colors)
#y_pos = np.arange(len(mean_correlations))
##    plt.xticks(y_pos, names, rotation='vertical')
#plt.xticks(y_pos, mean_names, rotation='vertical')
#low = min(mean_correlations)
#high = max(mean_correlations)
#plt.ylabel("averaged pearson correlation")
#plt.ylim([low-0.01, high+0.01])
##plt.ylim([0.88, 1])
#plt.grid()
##       
##angles = gradients_wrt_loss_angle(model, dataset.x_train,
##                                  dataset.y_train_labels)



## @@@


#with open("/cs/labs/daphna/guy.hacohen/project/models4/debug_grads_angles", 'rb') as file_pi:
#    (grads_easy_dict, grads_all_dict) = pickle.load(file_pi)
#
#batchs_easy = sorted(list(grads_easy_dict.keys()))
#grads_easy = [grads_easy_dict[batch] for batch in batchs_easy]
#
#batchs_all = sorted(list(grads_all_dict.keys()))
#grads_all = [grads_all_dict[batch] for batch in batchs_all]
#
#for layer_idx, layer_name in enumerate(["2 before last", "1 before last", "last"]):
#    cur_easy_grads = [g[layer_idx] for g in grads_easy]
#    cur_rand_grads = [g[layer_idx] for g in grads_all]
#    plt.figure()
#    plt.plot(batchs_easy, cur_easy_grads, ".", label="easy")
#    plt.plot(batchs_all, cur_rand_grads, ".", label="random")
#    plt.legend()
#    plt.xlabel("batch")
##    plt.ylabel("trace covariance")
##    plt.title("trace covariance as function of batch, for " + layer_name + " layer")
#    plt.ylabel("varice of dist from mean")
#    plt.title("variace of dists from mean as function of batch, for " + layer_name + " layer")
#    plt.figure()
#    plt.plot(batchs_easy, [g_rand/g_easy for g_easy, g_rand in zip(cur_easy_grads, cur_rand_grads)], ".", label="easy_rand_ratio")
#    plt.xlabel("batch")
##    plt.ylabel("trace covariance ratio")
##    plt.title("trace covariance ratio as function of batch, for " + layer_name + " layer")
#    plt.ylabel("varice of dist from mean ratio")
#    plt.title("variace of dists from mean ratio as function of batch, for " + layer_name + " layer")
#    print()


def exponent_change_jumps_data_function_generator(dataset, order, batches_to_increase_list, increase_amount, starting_percent, batch_size=100):

    size_data = dataset.x_train.shape[0]
    
    cur_percent = 1
    cur_data_x = dataset.x_train
    cur_data_y = dataset.y_test_labels
    
    inc_locations = np.cumsum(batches_to_increase_list)
    def data_function(x, y, batch, history, model):
        nonlocal cur_percent, cur_data_x, cur_data_y
        if batch == 0:
            percent = starting_percent
        elif batch in inc_locations:
            percent = min(cur_percent*increase_amount, 1)
        else:
            percent = cur_percent
            
        if percent != cur_percent:
            cur_percent = percent
            data_limit = np.int(np.ceil(size_data * percent))
            new_data = order[:data_limit]
            cur_data_x = dataset.x_train[new_data, :, :, :]
            cur_data_y = dataset.y_train_labels[new_data, :]               
        return cur_data_x, cur_data_y

    return data_function


#print(grads[-1])
#print(grads([dataset.x_train, dataset.y_train_labels]))

#weights = model.trainable_weights # weight tensors
##weights = [weight for weight in weights if model.get_layer(weight.name[:-9]).trainable] # filter down weights tensors to only ones which are trainable
#gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
#
#input_tensors = [model.inputs[0], # input data
#                 model.sample_weights[0], # how much to weight each sample by
#                 model.targets[0], # labels
#                 K.learning_phase(), # train or test mode
#]
#get_gradients = K.function(inputs=input_tensors, outputs=gradients)
#
#inputs = [[dataset.x_train[0,:,:,:]], # X
#          [1], # sample weights
#          [dataset.y_train_labels[0,:]], # y
#          0 # learning phase in TEST mode
#]
#
#print(get_gradients(inputs))

#
#orders = []
#names = ["vgg16", "vgg19", "inception", "xception", "resnet"]
#for network_name in names:
#    if network_name == "inception":
#        (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
#
#    else:
#        (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_classic_networks(dataset,
#                                                                                                               network_name)
#
#    train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
#                                                                 transfer_values_test, dataset.y_test, dataset,
#                                                                 network_name=network_name)
#    print(train_scores.shape)
#    
#    order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train)
#    orders.append(order)


def combine_orders_with_func(orders, order_func):
    return np.argsort(order_func([np.argsort(order) for order in orders]))

def combine_orders_max(*args):
    return combine_orders_with_func(args, lambda x: np.max(x, axis=0))

def combine_orders_min(*args):
    return combine_orders_with_func(args, lambda x: np.min(x, axis=0))

def combine_orders_sum(*args):
    return combine_orders_with_func(args, lambda x: np.sum(x, axis=0))

def combine_imagenet_networks(network_names, dataset):
    orders = []
    for network_name in network_names:
        if network_name == "inception":
            (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
    
        else:
            (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_classic_networks(dataset,
                                                                                                                   network_name)
    
        train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
                                                                     transfer_values_test, dataset.y_test, dataset,
                                                                     network_name=network_name)        
        order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train)
        orders.append(order)
    return combine_orders_sum(*orders)


def evaluate_model(model, x, y):
    predicted_y = np.argmax(model.predict(x), axis=1)
    print(predicted_y.shape)
    print(y.shape)
    return np.mean(predicted_y == y)

def evaluate_test_set_cross_val(model_path, validation_dataset):
    model = keras.models.load_model(model_path)
    x_test, y_test, _ = validation_dataset.get_original_test_data()
#    return evaluate_model(model, validation_dataset.x_test, validation_dataset.y_test)
    return evaluate_model(model, x_test, y_test)
    

#validation_dataset = cifar100_subset_validation.Cifar100_Subset_Validation(supeclass_idx=16)
##base_4 = r"/cs/labs/daphna/guy.hacohen/project/models4/"
##curriculum_model_path = base_4 + r"validation_16_curriculum_sgd_expo_balance_lr_0.04_100_1.9_0.05_decay2_min1e-4_lrjmp600"
##vanilla_model_path = base_4 + r"validation_16_vanilla_sgd_expo_balance_lr_0.07_100_1.9_0.05_decay1.3_min1e-4_lrjmp400"
#
#base_validation = r"/cs/labs/daphna/guy.hacohen/project/models_validation/"
#curriculum_model_path = base_validation + r"debug_curriculum"
#vanilla_model_path = base_validation + r"debug_vanilla"
#
#
#
#vanilla_test_score = evaluate_test_set_cross_val(vanilla_model_path, validation_dataset)
#curriculum_test_score = evaluate_test_set_cross_val(curriculum_model_path, validation_dataset)
#
#print("vanilla score:", vanilla_test_score)
#print("curriculum score:", curriculum_test_score)
#
#fig, ax = plt.subplots()
#cmap = plt.get_cmap('rainbow')
#colors = [cmap(i) for i in np.linspace(0, 1, 2)]
#final_tag = [vanilla_test_score, curriculum_test_score]
#bar_locations = np.arange(2)
#ax.bar(bar_locations, final_tag, color=colors)
#plt.ylim([0.45, 0.65])
#names = ["vanilla", "curriculum"]
#y_pos = np.arange(len(names))
#plt.xticks(y_pos, names, rotation='vertical')
#plt.title("cross validation - curriculum vs vanilla")
#combined_order_no_vgg = combine_imagenet_networks(names[2:], dataset)
#combined_order_all = combine_imagenet_networks(names, dataset)
#orders = [combined_order_all, combined_order_no_vgg] + orders
#names = ["combined_all", "combined_no_vgg"] + names
#
#for i in range(len(orders)):
#    for j in range(i+1, len(orders)):
#        correlation_orders(orders[i], orders[j], dataset)
#        plt.title(names[i] + " vs " + names[j])
##        plt.savefig("/cs/labs/daphna/guy.hacohen/project/graphs/to_daphna/24.7.18/" + names[i] + "_vs_" + names[j] + ".png")
#        plt.show()



    # print("dataset: " + dataset.name)
    # print("network: " + args.order)
    # print("svm train score:")
    # print(np.mean(np.argmax(train_scores, axis=1) == dataset.y_train))
    # print("svm test score:")
    # print(np.mean(np.argmax(test_scores, axis=1) == dataset.y_test))


#    
#    if len(test_x) != 0:
#        print("evaluating svm")
#        test_scores = clf.predict_proba(test_x)
#        print('accuracy for svm = ', str(np.mean(np.argmax(test_scores, axis=1) == test_y)))
#        print(np.mean(np.argmax(test_scores, axis=1)))
#        print(test_y)
#        print("evaluating answer")
#    else:
#        test_scores = []
#    train_scores = clf.predict_proba(train_x)
#    return train_scores, test_scores


#def exponent_change_lr_generator(decay_rate, minimum_lr, batch_to_decay):
#    cur_lr = None
#    def exponent_decay_lr(initial_lr, batch, history):
#        nonlocal cur_lr
#        if batch == 0:
#            cur_lr = initial_lr
#        if (batch % batch_to_decay) == 0 and batch !=0:
#            new_lr = cur_lr / decay_rate
#            cur_lr = max(new_lr, minimum_lr)
#        return cur_lr
#    return exponent_decay_lrin_labels[new_data, :]               
#        return cur_data_x, cur_data_y



def basic_lr_scheduler(initial_lr, batch, history):
    return initial_lr

def basic_change_lr(orig_lr, full_data_size, cur_data_size):
    return orig_lr * (cur_data_size / full_data_size)

def exponent_change_lr_generator(exponent):
    
    def exponent_change_lr(orig_lr, full_data_size, cur_data_size):
        return orig_lr * (cur_data_size**exponent / full_data_size**exponent)
    
    return exponent_change_lr

def change_lr_schedule_according_to_curriculum(lr_scheduler, dataset, change_lr_func):
    real_data_size = dataset.y_train.size
    
    def new_schedule_function(initial_lr, batch, history):
        original_lr = lr_scheduler(initial_lr, batch, history)
        new_lr = change_lr_func(original_lr, real_data_size, history["data_size"][-1])
        return new_lr
    
    return new_schedule_function
    
#changed_lr_schdule = change_lr_schedule_according_to_curriculum(basic_lr_scheduler, dataset, exponent_change_lr_generator(0.2))
    

def single_step_data_function_generator(dataset, order, step_batch, starting_percent):

    size_data = dataset.x_train.shape[0]
    
    cur_percent = None
    cur_data_x = dataset.x_train
    cur_data_y = dataset.y_test_labels
    
    def data_function(x, y, batch, history, model):
        nonlocal cur_percent, cur_data_x, cur_data_y
        if batch == step_batch or  batch == 0:
            if batch == 0:
                percent = starting_percent
            else:
                percent = 1
            if percent != cur_percent:
                cur_percent = percent
                data_limit = np.int(np.ceil(size_data * percent))
                new_data = order[:data_limit]
                cur_data_x = dataset.x_train[new_data, :, :, :]
                cur_data_y = dataset.y_train_labels[new_data, :]             
        return cur_data_x, cur_data_y

    return data_function



#
#
#import train_keras_model
#import transfer_learning
#import cifar100_model
#import cifar100_subset
#
#def balance_order(order, dataset):
#    num_classes = dataset.n_classes
#    size_each_class = dataset.x_train.shape[0] // num_classes
#    class_orders = []
#    for cls in range(num_classes):
#        class_orders.append([i for i in range(len(order)) if dataset.y_train[order[i]] == cls])
#    new_order = []
#    ## take each group containing the next easiest image for each class,
#    ## and putting them according to diffuclt-level in the new order
#    for group_idx in range(size_each_class):
#        group = sorted([class_orders[cls][group_idx] for cls in range(num_classes)])
#        for idx in group:
#            new_order.append(order[idx])
#    return new_order
#
#def gad_order(reverse=False, random=False):
#    with open(r"/cs/labs/daphna/guy.hacohen/project/data/cifar_100_superclass_16/gad_indexes", "rb+") as pick_file:
#        order = pickle.load(pick_file)
#    res = np.asarray(order)
#    if reverse:
#        res = np.flip(res, 0)
#    if random:
#        np.random.shuffle(res)
#    return res
#
#
#def order_according_to_model(dataset, model_lib, net_type, reverse=False, random=False):
#    model = model_lib.build_classifier_model(dataset, model_type=net_type)
#    train_keras_model.compile_model(model)
#    predicted = model.predict(dataset.x_train)
#    order = transfer_learning.rank_data_according_to_score(predicted, dataset.y_train, reverse=reverse,
#                                                           random=random)
#    return order
#
#def order_according_to_trained_model(model, dataset, reverse=False, random=False):
#    predicted = model.predict(dataset.x_train)
#    order = transfer_learning.rank_data_according_to_score(predicted, dataset.y_train, reverse=reverse,
#                                                           random=random)
#    return order
#
#
#def order_by_freq(dataset):
#    """
#    returns training order of the given dataset by freqency
#    low freq images will be first, high freq images will be last
#    """
#    images = dataset.x_train
#    num_images, h, w, c = images.shape
#
#    ## fourier transform for getting the freq map
#    images_last = images.transpose(0,3,1,2)
#    freq_map = np.abs(np.fft.fft2(images_last))
#    scores = np.zeros(num_images)
#    for img_idx in range(num_images):
#        image_score = 0
#        for c_idx in range(c):
#            for freq_x in range(h):
#                for freq_y in range(w):
#                    ## the freq at 0,0 is simply the images mean, which is usally normalized anyway.
#                    ## this if makes the score invariant to it.
#                    if freq_x == 0 and freq_y == 0:
#                        continue
#                    image_score += freq_map[img_idx, c_idx, freq_x, freq_y] / (freq_x+freq_y)
#        scores[img_idx] = image_score
#    ## takes the scores of every image, and produces an ordering.
#    ## res[0] is the index of the "easiest" image by the scoring, res[1] is the index of a bit harder image, etc...
#    res = np.asarray(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
#    return res
#
#
#def order_by_prototype(dataset, reverse=False, random=False):
#    num_images, h, w, c = dataset.x_train.shape
#    prototypes = np.zeros((dataset.n_classes, h, w, c))
#    for class_idx in range(dataset.n_classes):
#        class_indexes = [i for i in range(num_images) if dataset.y_train[i] == class_idx]
#        prototypes[class_idx, :, :, :] = np.mean(dataset.x_train[class_indexes, :, :, :], axis=0)
#
#    scores = np.zeros(num_images)
#
#    for img_idx in range(num_images):
#        cur_img = dataset.x_train[img_idx, :, :, :]
#        cur_proto = prototypes[dataset.y_train[img_idx], :, :, :]
#        score = np.sum(np.abs(cur_img - cur_proto))
#        scores[img_idx] = score
#
#    res = np.asarray(sorted(range(len(scores)), key=lambda k: scores[k]))
#
#    if reverse:
#        res = np.flip(res, 0)
#    if random:
#        np.random.shuffle(res)
#
#    return res
#
#def order_by_small_network(dataset, model_lib=cifar100_model.Cifar100_Model()):
#    file_path_cache = os.path.join(dataset.data_dir, 'small_network_order_' + dataset.name + '.pkl')
#
#    if os.path.exists(file_path_cache):
#        with open(file_path_cache, "rb") as pick_file:
#            res = pickle.load(pick_file)
#    else:
#        epochs = 100
#        size_train = dataset.x_train.shape[0]
#        batch_size = 100
#        num_batchs = (epochs * size_train) // batch_size
#        dropout1 = 0.25
#        dropout2 = 0.5
#        lr = 1e-3
#        reg_factor = 50e-4
#        bias_reg_factor = None
#        optimizer = "sgd"
#        model = model_lib.build_classifier_model(dataset, model_type="small",
#                                                 dropout_1_rate=dropout1, dropout_2_rate=dropout2,
#                                                 reg_factor=reg_factor,
#                                                 bias_reg_factor=bias_reg_factor)
#
#        train_keras_model.compile_model(model, initial_lr=lr,
#                                        loss='categorical_crossentropy',
#                                        optimizer=optimizer, metrics=['accuracy'])
#
#        history = train_keras_model.train_model_batches(model, dataset.x_train, dataset.y_train_labels, dataset.x_test,
#                                                        dataset.y_test_labels, num_batchs, verbose=False,
#                                                        batch_size=batch_size,
#                                                        initial_lr=lr,
#                                                        loss='categorical_crossentropy',
#                                                        optimizer=optimizer, Compile=False,
#                                                        model_output_path=None, metrics=['accuracy'],
#                                                        reduce_history=True)
#        train_keras_model.evaluate_model(model, dataset.x_train, dataset.y_train_labels, dataset.x_test, dataset.y_test_labels)
#        train_scores = model.predict(dataset.x_train)
#        hardness_score = train_scores[list(range(size_train)), dataset.y_train]
#        res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
#        with open(file_path_cache, "wb") as pick_file:
#            pickle.dump(res, pick_file)
#    return res
#
#
#def order_by_same_network(dataset, model_lib, model_type, dropout_1_rate, dropout_2_rate,
#                          reg_factor, bias_reg_factor, initial_lr, num_batchs,
#                          batch_size, optimizer, lr_scheduler):
#    model = model_lib.build_classifier_model(dataset, model_type=model_type,
#                                             dropout_1_rate=dropout_1_rate, dropout_2_rate=dropout_2_rate,
#                                             reg_factor=reg_factor,
#                                             bias_reg_factor=bias_reg_factor)
#    train_keras_model.compile_model(model, initial_lr=initial_lr,
#                                    loss='categorical_crossentropy',
#                                    optimizer=optimizer, metrics=['accuracy'])
#    history = train_keras_model.train_model_batches(model, dataset.x_train, dataset.y_train_labels, dataset.x_test,
#                                                    dataset.y_test_labels, num_batchs, verbose=False,
#                                                    batch_size=batch_size,
#                                                    initial_lr=initial_lr,
#                                                    lr_scheduler=lr_scheduler, loss='categorical_crossentropy',
#                                                    optimizer=optimizer, Compile=False,
#                                                    model_output_path=None, metrics=['accuracy'])
#    train_keras_model.evaluate_model(model, dataset.x_train, dataset.y_train_labels, dataset.x_test, dataset.y_test_labels)
#    train_scores = model.predict(dataset.x_train)
#    hardness_score = train_scores[list(range(size_train)), dataset.y_train]
#    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
#    return res
#
#
#def expo_lr_scheduler_generator(batches_to_increase, amount, starting_percent, to_increase=False):
#    lr_multipliers = [1]
#    cur_percent = starting_percent
#    cur_batch = 0
#    while cur_percent < 1:
#        cur_percent *= amount
#        cur_batch += batches_to_increase
#        if to_increase:
#            lr_multipliers.append(min(lr_multipliers[-1]*amount, 1/starting_percent))
#
#        else:
#            lr_multipliers.append(max(lr_multipliers[-1]/amount, starting_percent))
#
#    def decrease_lr_scheduler(initial_lr, batch, history):
#
#        increase_idx = batch // batches_to_increase
#        if increase_idx >= len(lr_multipliers):
#            return initial_lr * lr_multipliers[-1]
#        else:
#            return initial_lr * lr_multipliers[increase_idx]
#
#    return decrease_lr_scheduler
#
#lr_scheduler = expo_lr_scheduler_generator(500,
#                                           2,
#                                           600)
#
#dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=14)
#model_lib = cifar100_model.Cifar100_Model()
#
#
#orders = []
## "vgg16", "vgg19", "xception", "resnet"
#for net in ["inception"]:
#    network_name = net
#    if network_name == "inception":
#        (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
#
#    else:
#        (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_classic_networks(dataset,
#                                                                                                               network_name)
#
#    train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
#                                                                 transfer_values_test, dataset.y_test, dataset,
#                                                                 network_name=network_name)
#    order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train)
#    orders.append(order)
#
#gad_order = gad_order()
#freq_order = order_by_freq(dataset)
#proto_order = order_by_prototype(dataset)
##small_order = order_by_small_network(dataset, model_lib)
##same_order = order_by_same_network(dataset, model_lib, "large", 0.25, 0.5,
##                                   200e-4, None, 0.035, 100,
##                                   100, "sgd", lr_scheduler)
###
#
#orders.append(gad_order)
#orders.append(freq_order)
#orders.append(proto_order)
##orders.append(same_order)
## "vgg16", "vgg19", "xception", "resnet",
#names = ["inception", "gad", "freq", "proto"]
#def correlation_orders(order1, order2, dataset):
#    balanced_1 = balance_order(order1, dataset)
#    balanced_2 = balance_order(order2, dataset)
#    plt.plot(np.argsort(balanced_1), np.argsort(balanced_2), ".")
#    
#for i in range(len(names)):
#    for j in range(i + 1, len(names)):
#        plt.figure()
#        correlation_orders(orders[i], orders[j], dataset)
#        plt.xlabel(names[i])
#        plt.ylabel(names[j])
#        plt.title("orders: " + names[i] + ", " + names[j])