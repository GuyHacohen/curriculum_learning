# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

# Functions and classes for loading and using the Inception model.
import models.inception
#import stl10
from models.inception import transfer_values_cache

from sklearn import svm
import numpy as np
import pickle
import classic_nets_imagenet

# download the models / datasets
def get_transfer_values_inception(dataset):
    data_dir = r'./data/'
    models.inception.data_dir = os.path.join(data_dir, 'inception/')
    dataset.data_dir = os.path.join(data_dir, dataset.name + r'/')
    if not os.path.exists(dataset.data_dir):
        os.mkdir(dataset.data_dir)
    models.inception.maybe_download()
#    dataset.maybe_download()
    
    #load the inception model
    model = models.inception.Inception()
    
    #load the dataset data
#    images_train, cls_train, labels_train = dataset.load_training_data()
#    images_test, cls_test, labels_test = dataset.load_test_data()
    
    images_train = dataset.x_train
#    cls_train = dataset.y_train
#    labels_train = dataset.y_train_labels

    images_test = dataset.x_test
#    cls_test = dataset.y_test
#    labels_test = dataset.y_test_labels
    
    # path to save the cache values
    file_path_cache_train = os.path.join(dataset.data_dir, 'inception_' + dataset.name + '_train.pkl')
    file_path_cache_test = os.path.join(dataset.data_dir, 'inception_' + dataset.name + '_test.pkl')
    
    # stl10 and inception both need pixels between 0 to 255.
    # however, when using other datasets, preprocessing might 
    # be required.

    # images_scaled = images_train * 255.0

    print("Transfering training set")
    
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                                  images=images_train,
                                                  model=model)
    
    print("Transfering test set")
    
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                                  images=images_test,
                                                  model=model)
    return transfer_values_train, transfer_values_test


def get_transfer_values_classic_networks(dataset, network_name):

    # path to save the cache values
    file_path_cache_train = os.path.join(dataset.data_dir, network_name + '_' + dataset.name + '_train.pkl')
    file_path_cache_test = os.path.join(dataset.data_dir, network_name + '_' + dataset.name + '_test.pkl')


    #
    # if output_path is not None:
    #     history_output = output_path + "_nets" + str(args.num_models) + "_history"
    #     print('saving trained model to:', history_output)
    #     with open(history_output, 'wb') as file_pi:
    #         pickle.dump(history, file_pi)

    print("Transfering training set")

    if os.path.exists(file_path_cache_train):
        print("training set already exist on disk")
        with open(file_path_cache_train, "rb") as pick_file:
            transfer_values_train = pickle.load(pick_file)
    else:
        transfer_values_train = classic_nets_imagenet.classify_img(dataset.x_train, network_name)
        with open(file_path_cache_train, "wb") as pick_file:
            pickle.dump(transfer_values_train, pick_file)

    print("Transfering test set")

    if os.path.exists(file_path_cache_test):
        print("test set already exist on disk")
        with open(file_path_cache_test, "rb") as pick_file:
            transfer_values_test = pickle.load(pick_file)
    else:
        transfer_values_test = classic_nets_imagenet.classify_img(dataset.x_test, network_name)
        with open(file_path_cache_test, "wb") as pick_file:
            pickle.dump(transfer_values_test, pick_file)

    return transfer_values_train, transfer_values_test


def transfer_values_svm_scores(train_x, train_y, test_x, test_y):
    clf = svm.SVC(probability=True)
    print("fitting svm")
    clf.fit(train_x, train_y)
    if len(test_x) != 0:
        print("evaluating svm")
        test_scores = clf.predict_proba(test_x)
        print('accuracy for svm = ', str(np.mean(np.argmax(test_scores, axis=1) == test_y)))
    else:
        test_scores = []
    train_scores = clf.predict_proba(train_x)
    return train_scores, test_scores

def svm_scores_exists(dataset, network_name="inception",
                      alternative_data_dir="."):
    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir
    
    svm_train_path = os.path.join(data_dir, network_name + 'svm_train_values.pkl')
    svm_test_path = os.path.join(data_dir, network_name + 'svm_test_values.pkl')
    return os.path.exists(svm_train_path) and os.path.exists(svm_test_path)

def get_svm_scores(transfer_values_train, y_train, transfer_values_test,
                   y_test, dataset, network_name="inception",
                   alternative_data_dir="."):
    
    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir
    
    svm_train_path = os.path.join(data_dir, network_name + 'svm_train_values.pkl')
    svm_test_path = os.path.join(data_dir, network_name + 'svm_test_values.pkl')
    if not os.path.exists(svm_train_path) or not os.path.exists(svm_test_path):
        train_scores, test_scores = transfer_values_svm_scores(transfer_values_train, y_train, transfer_values_test, y_test)
        with open(svm_train_path, 'wb') as file_pi:
            pickle.dump(train_scores, file_pi)

        with open(svm_test_path, 'wb') as file_pi:
            pickle.dump(test_scores, file_pi)
    else:
        with open(svm_train_path, 'rb') as file_pi:
            train_scores = pickle.load(file_pi)

        with open(svm_test_path, 'rb') as file_pi:
            test_scores = pickle.load(file_pi)
    return train_scores, test_scores


def rank_data_according_to_score(train_scores, y_train, reverse=False, random=False):
    train_size, _ = train_scores.shape
    hardness_score = train_scores[list(range(train_size)), y_train]
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    if reverse:
        res = np.flip(res, 0)
    if random:
        np.random.shuffle(res)
    return res
