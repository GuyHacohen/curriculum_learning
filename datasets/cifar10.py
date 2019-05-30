#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:58:27 2018

@author: guy.hacohen
"""

import os
import download
import sys
from six.moves import cPickle
from keras import backend as K
import numpy as np
from datasets.Dataset import one_hot_encoded
import datasets.Dataset

class Cifar10(datasets.Dataset.Dataset):

    def __init__(self, smaller_data_size=None, normalize=True):
        self.name = 'cifar10'
            
        # Internet URL for the tar-file with the Inception model.
        # Note that this might change in the future and will need to be updated.
        self.data_url = r"https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        # Directory to store the downloaded data.
        self.data_dir = "../data/cifar10/"

        self.height, self.width, self.depth = 32, 32, 3
        self.n_classes = 10
        self.img_size_flat = self.height * self.width * self.depth


        self.smaller_data_set = False
        if smaller_data_size is not None:
            self.smaller_data_set = True
            self.data_size = smaller_data_size
        super(Cifar10, self).__init__(smaller_data_size=smaller_data_size,
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

        data = data.reshape(data.shape[0], self.depth, self.width, self.height)
        return data, labels

    def maybe_download(self):
        """
        Download and extract the CIFAR-100 data-set if it doesn't already exist
        in data_path (set this variable first to the desired path).
        """

        download.maybe_download_and_extract(url=self.data_url, download_dir=self.data_dir)

    def load_training_data(self):
        dirname = 'cifar-10-batches-py'
        path = os.path.join(self.data_dir, dirname)
        n_train_batchs = 5
        x_train = np.zeros((0, self.depth, self.width, self.height))
        y_train = []
        for batch in range(n_train_batchs):
            fpath = os.path.join(path, 'data_batch_' + str(batch + 1))
            cur_data, cur_labels = self._load_batch(fpath)
            x_train = np.concatenate((cur_data, x_train), axis=0)
            y_train = cur_labels + y_train
        x_train = x_train.astype(np.uint8)
        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
        y_train_labels = one_hot_encoded(y_train, num_classes=self.n_classes)
        return x_train, np.array(y_train), y_train_labels
    
    
    def load_test_data(self):
        dirname = 'cifar-10-batches-py'
        path = os.path.join(self.data_dir, dirname)
        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = self._load_batch(fpath)

        if K.image_data_format() == 'channels_last':
            x_test = x_test.transpose(0, 2, 3, 1)

        y_test_labels = one_hot_encoded(y_test, num_classes=self.n_classes)
        return x_test, np.array(y_test), y_test_labels

    def normalize_dataset(self):

        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        if not self.normalized:
            mean = np.mean(self.x_train, axis=(0,1,2,3))
            std = np.std(self.x_train, axis=(0, 1, 2, 3))
            self.x_train = (self.x_train-mean)/(std+1e-7)
            self.x_test = (self.x_test-mean)/(std+1e-7)
        self.normalized = True