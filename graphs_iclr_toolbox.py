#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:37:12 2018

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
from keras import backend as K
import gc
from scipy.stats.stats import pearsonr
import itertools
from scipy import stats
import itertools
from collections import defaultdict


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
#    return [np.linalg.det(cov) for cov in covs]
    return [np.trace(cov) for cov in covs]


def gradients_wrt_loss_angle(model, x, y_labels):
    num_layers = 1 # gradients only for the last 3 layers
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
    num_layers = 1 # gradients only for the last 3 layers
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



def grad_directions_results(vanilla_path, order, dataset, num_repeats=15, order1=None, order2=None):
    random_order = list(range(2500))
    np.random.shuffle(random_order)
    
    names = ["all", "random", "inception", "easy vgg", "resnet"]
    idxes = [list(range(len(dataset.y_train))),
             random_order[:250],
             order[:250],
             order1[:250],
             order2[:250]]
    
    dists = defaultdict(lambda: [])
    variances = defaultdict(lambda: [])
        
    for i in range(num_repeats):
        base = r"/cs/labs/daphna/guy.hacohen/project/models_weights/"
        cur_model = keras.models.load_model(base + vanilla_path.replace("${i}", str(i+1)))
        
        for idx_first, idx_second in itertools.combinations(range(len(names)), 2):
            
            idxs1 = idxes[idx_first]
            idxs2 = idxes[idx_second]
            name1 = names[idx_first]
            name2 = names[idx_second]
            
            
            means1 = mean_grad_direction(cur_model,
                                         dataset.x_train[idxs1, :, :, :],
                                         dataset.y_train_labels[idxs1, :])
            
            means2 = mean_grad_direction(cur_model,
                                         dataset.x_train[idxs2, :, :, :],
                                         dataset.y_train_labels[idxs2, :])
            

            dists[name1 + "/" + name2].append([euclidean_dist(v1, v2) for v1, v2 in zip(means1, means2)])
        
        #    stds = gradients_wrt_loss(model, x, y_labels)
        
        for idx in range(len(names)):
            idxs = idxes[idx]
            stds = gradients_wrt_loss(cur_model,
                                      dataset.x_train[idxs, :, :, :],
                                      dataset.y_train_labels[idxs, :])
            variances[names[idx]].append(stds)
                        
        del cur_model
        K.clear_session()
    
    return dists, variances


def get_classic_order(order_name, dataset):
    ## posiible orderes: ["vgg16", "resnet", "inception"]
    network_name = order_name
    if order_name == "inception":
        (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)

    else:
        (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_classic_networks(dataset,
                                                                                                               network_name)

    train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
                                                                 transfer_values_test, dataset.y_test, dataset,
                                                                 network_name=network_name)
    order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train)
    
    return order