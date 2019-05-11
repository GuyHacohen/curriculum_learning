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
import sklearn

# Various directories and file-names.

class Cifar100_Subset_Validation(Dataset.Dataset):

    def __init__(self, supeclass_idx=0, normalize=True):
        self.superclass_idx = supeclass_idx
        self.name = 'cifar_100_validation_superclass_' + str(self.superclass_idx)

        # Internet URL for the tar-file with the Inception model.
        # Note that this might change in the future and will need to be updated.
        self.data_url = r"https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        # Directory to store the downloaded data.
        self.data_dir = "../data/cifar100/"

        self.height, self.width, self.depth = 32, 32, 3
        self.n_classes = 5
        self.n_super_classes = 1
        self.img_size_flat = self.height * self.width * self.depth


        ########################################################################
        # Various constants used to allocate arrays of the correct size.

        # Number of files for the training-set.
        self._num_files_train = 1

        # Number of images for each batch-file in the training-set.
        self._images_per_file = 2500

        self.total_train_size = 2500
        
        # list of n_classes lists, each list contains the random order
        # of indexes for corresponds class
        self.new_pre_class_order = []
        for i in range(self.n_classes):
            cur_idxes = list(range(self.total_train_size // self.n_classes))
            cur_idxes = sklearn.utils.shuffle(cur_idxes, random_state=self.superclass_idx + i + 31)
            self.new_pre_class_order.append(cur_idxes)
            
        self.new_train_idxes = np.array(sklearn.utils.shuffle(list(range(2000)), random_state=self.superclass_idx + 11))
        self.new_val_idxes = np.array(sklearn.utils.shuffle(list(range(500)), random_state=self.superclass_idx + 13))
        # Total number of images in the training-set.
        # This is used to pre-allocate arrays for efficiency.
        self._num_images_train = self._num_files_train * self._images_per_file
        
        super(Cifar100_Subset_Validation, self).__init__(normalize=normalize)

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

    def load_orig_training_data(self):
        dirname = 'cifar-100-python'
        path = os.path.join(self.data_dir, dirname)
        fpath = os.path.join(path, 'train')
        x_train, y_train_fine = self._load_batch(fpath, 'fine_labels')
        _, y_train_coarse = self._load_batch(fpath, 'coarse_labels')

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)

        curr_superclass_idxes = [i for i in range(len(y_train_fine)) if y_train_coarse[i] == self.superclass_idx]
        x_train = x_train[curr_superclass_idxes]
        y_train = np.asarray(y_train_fine)[curr_superclass_idxes]
        y_train_values = sorted(list(set(y_train)))
        assert(len(y_train_values) == self.n_classes)
        map_dict = {val: i for i, val in enumerate(y_train_values)}
        for i, y in enumerate(y_train):
            y_train[i] = map_dict[y]

        y_train_labels = one_hot_encoded(y_train, num_classes=self.n_classes)
        return x_train, y_train, y_train_labels


    def load_training_data(self):
        orig_x_train, orig_y_train, orig_y_labels = self.load_orig_training_data()
        
        shuffled_train_idxes = []
        for i in range(self.n_classes):
            class_indexes = np.array([idx for idx, clss in enumerate(orig_y_train) if clss == i])
            shuffled_train_idxes += list(class_indexes[self.new_pre_class_order[i]][:400])
        
        
        return (orig_x_train[shuffled_train_idxes][self.new_train_idxes],
                orig_y_train[shuffled_train_idxes][self.new_train_idxes],
                orig_y_labels[shuffled_train_idxes][self.new_train_idxes])

    def load_test_data(self):
        orig_x_train, orig_y_train, orig_y_labels = self.load_orig_training_data()
        
        shuffled_train_idxes = []
        for i in range(self.n_classes):
            class_indexes = np.array([idx for idx, clss in enumerate(orig_y_train) if clss == i])
            shuffled_train_idxes += list(class_indexes[self.new_pre_class_order[i]][400:])
            
        return (orig_x_train[shuffled_train_idxes][self.new_val_idxes],
                orig_y_train[shuffled_train_idxes][self.new_val_idxes],
                orig_y_labels[shuffled_train_idxes][self.new_val_idxes])
    
    def get_original_test_data(self):
        dirname = 'cifar-100-python'
        path = os.path.join(self.data_dir, dirname)
        fpath = os.path.join(path, 'test')
        x_test, y_test_fine = self._load_batch(fpath, 'fine_labels')
        _, y_test_coarse = self._load_batch(fpath, 'coarse_labels')

        if K.image_data_format() == 'channels_last':
            x_test = x_test.transpose(0, 2, 3, 1)

        curr_superclass_idxes = [i for i in range(len(y_test_fine)) if y_test_coarse[i] == self.superclass_idx]
        x_test = x_test[curr_superclass_idxes]
        y_test = np.asarray(y_test_fine)[curr_superclass_idxes]
        y_test_values = sorted(list(set(y_test)))
        assert(len(y_test_values) == self.n_classes)
        map_dict = {val: i for i, val in enumerate(y_test_values)}
        for i, y in enumerate(y_test):
            y_test[i] = map_dict[y]

        y_test_labels = one_hot_encoded(y_test, num_classes=self.n_classes)
        
        #normalization
        x_test = (x_test - 128.) / 128
        
        return x_test, y_test, y_test_labels
    
    def set_superclass_idx(self, new_idx):
        self.superclass_idx = new_idx
        self.name = 'cifar_100_superclass_' + str(new_idx)
        smaller_data_set = None
        if self.smaller_data_set:
            smaller_data_set = self.data_size
        super(Cifar100_Subset_Validation, self).update_data_set(smaller_data_set)

    def normalize_dataset(self):
        self.x_train = (self.x_train - 128.) / 128
        # x_test = x_test.astype('float32')
        self.x_test = (self.x_test - 128.) / 128

    def normalize_vgg(self):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(self.x_train, axis=(0,1,2,3))
        std = np.std(self.x_train, axis=(0, 1, 2, 3))
        self.x_train = (self.x_train-mean)/(std+1e-7)
        self.x_test = (self.x_test-mean)/(std+1e-7)