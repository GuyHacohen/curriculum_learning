#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:03:40 2018

@author: stenly
"""
import os
import download
import sys
from six.moves import cPickle
from keras import backend as K
from dataset import one_hot_encoded
import numpy as np
import Dataset

# Various directories and file-names.
def array_to_str(arr):
    res = ""
    for i in arr:
        res += str(i) + "_"
    if arr:
        res = res[:-1]
    return res
    
class Cifar100_Custom_Subset(Dataset.Dataset):

    def __init__(self, smaller_data_size=None, normalize=True,
                 subsets_idxes=[10,20,30,40,50]):
        if subsets_idxes == "all":
            self.subsets_idxes = list(range(100))
            self.name = "cifar100_all"
        else:
            self.subsets_idxes = sorted(subsets_idxes)
            self.name = 'cifar_100_custom_subset_' + array_to_str(subsets_idxes)

            
        # Internet URL for the tar-file with the Inception model.
        # Note that this might change in the future and will need to be updated.
        self.data_url = r"https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        # Directory to store the downloaded data.
        self.data_dir = "../data/cifar100/"

        self.height, self.width, self.depth = 32, 32, 3
        self.n_classes = len(self.subsets_idxes)
        self.img_size_flat = self.height * self.width * self.depth


        self.smaller_data_set = False
        if smaller_data_size is not None:
            self.smaller_data_set = True
            self.data_size = smaller_data_size
        super(Cifar100_Custom_Subset, self).__init__(smaller_data_size=smaller_data_size,
                                                     normalize=normalize)

    def _load_batch(self, fpath, label_key='labels'):
        """Internal utility for parsing CIFAR data.

        # Arguments
            fpath: path the file to parse.
            label_key: key for label data in the retrieve
                dictionary.

        # Returns
            A tuple `(data, labels)`.
        """
        f = open(fpath, 'rb')
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
        f.close()
        data = d['data']
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    def maybe_download(self):
        """
        Download and extract the CIFAR-100 data-set if it doesn't already exist
        in data_path (set this variable first to the desired path).
        """

        download.maybe_download_and_extract(url=self.data_url, download_dir=self.data_dir)

    def load_training_data(self):
        dirname = 'cifar-100-python'
        path = os.path.join(self.data_dir, dirname)
        fpath = os.path.join(path, 'train')
        x_train, y_train_fine = self._load_batch(fpath, 'fine_labels')
        data_size = len(y_train_fine)

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            
        relevant_idxes = [i for i in range(data_size) if y_train_fine[i] in self.subsets_idxes]
        x_train = x_train[relevant_idxes, :, :, :]
        y_train = np.asarray(y_train_fine)[relevant_idxes]
        y_train_values = sorted(list(set(y_train)))
        assert(len(y_train_values) == self.n_classes)
        map_dict = {val: i for i, val in enumerate(y_train_values)}
        for i, y in enumerate(y_train):
            y_train[i] = map_dict[y]

        y_train_labels = one_hot_encoded(y_train, num_classes=self.n_classes)
        return x_train, y_train, y_train_labels

    def load_test_data(self):
        dirname = 'cifar-100-python'
        path = os.path.join(self.data_dir, dirname)
        fpath = os.path.join(path, 'test')
        x_test, y_test_fine = self._load_batch(fpath, 'fine_labels')
        data_size = len(y_test_fine)

        if K.image_data_format() == 'channels_last':
            x_test = x_test.transpose(0, 2, 3, 1)
            
        relevant_idxes = [i for i in range(data_size) if y_test_fine[i] in self.subsets_idxes]
        x_test = x_test[relevant_idxes, :, :, :]
        y_test = np.asarray(y_test_fine)[relevant_idxes]
        y_test_values = sorted(list(set(y_test)))
        assert(len(y_test_values) == self.n_classes)
        map_dict = {val: i for i, val in enumerate(y_test_values)}
        for i, y in enumerate(y_test):
            y_test[i] = map_dict[y]

        y_test_labels = one_hot_encoded(y_test, num_classes=self.n_classes)
        return x_test, y_test, y_test_labels

    def normalize_dataset(self):
        self.x_train = self.x_train / 256.
        # x_test = x_test.astype('float32')
        self.x_test = self.x_test / 256.

        
    def normalize_vgg(self):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(self.x_train, axis=(0,1,2,3))
        std = np.std(self.x_train, axis=(0, 1, 2, 3))
        self.x_train = (self.x_train-mean)/(std+1e-7)
        self.x_test = (self.x_test-mean)/(std+1e-7)