#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import download
import sys
from six.moves import cPickle
from keras import backend as K
import numpy as np
from datasets.Dataset import one_hot_encoded
import datasets.Dataset

# Various directories and file-names.
                 
class Cifar100_Subset(datasets.Dataset.Dataset):

    def __init__(self, smaller_data_size=None, supeclass_idx=0, normalize=True,
                 cross_val=False, order=None, order_name=""):
        self.superclass_idx = supeclass_idx
        self.name = 'cifar_100_superclass_' + str(self.superclass_idx)

        # Internet URL for the tar-file with the Inception model.
        # Note that this might change in the future and will need to be updated.
        self.data_url = r"https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        # Directory to store the downloaded data.
        self.data_dir = "./data/cifar100/"

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

        # Total number of images in the training-set.
        # This is used to pre-allocate arrays for efficiency.
        self._num_images_train = self._num_files_train * self._images_per_file
        self.smaller_data_set = False
        if smaller_data_size is not None:
            self.smaller_data_set = True
            self.data_size = smaller_data_size
        super(Cifar100_Subset, self).__init__(normalize=normalize)

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

    def load_test_data(self):
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
        return x_test, y_test, y_test_labels

    def set_superclass_idx(self, new_idx):
        self.superclass_idx = new_idx
        self.name = 'cifar_100_superclass_' + str(new_idx)
        smaller_data_set = None
        if self.smaller_data_set:
            smaller_data_set = self.data_size
        super(Cifar100_Subset, self).update_data_set(smaller_data_set)

    def normalize_dataset(self):
        if not self.normalized:
            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_test.astype('float32')
            mean_r = np.mean(self.x_train[:,:,:,0])
            mean_g = np.mean(self.x_train[:,:,:,1])
            mean_b = np.mean(self.x_train[:,:,:,2])
            
            std_r = np.std(self.x_train[:,:,:,0])
            std_g = np.std(self.x_train[:,:,:,1])
            std_b = np.std(self.x_train[:,:,:,2])
            
            
            self.x_train[:,:,:,0] = (self.x_train[:,:,:,0] - mean_r) / std_r
            self.x_train[:,:,:,1] = (self.x_train[:,:,:,1] - mean_g) / std_g
            self.x_train[:,:,:,2] = (self.x_train[:,:,:,2] - mean_b) / std_b
            
            self.x_test[:,:,:,0] = (self.x_test[:,:,:,0] - mean_r) / std_r
            self.x_test[:,:,:,1] = (self.x_test[:,:,:,1] - mean_g) / std_g
            self.x_test[:,:,:,2] = (self.x_test[:,:,:,2] - mean_b) / std_b
        self.normalized = True