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

# Various directories and file-names.

name = 'cifar100'

# Internet URL for the tar-file with the Inception model.
# Note that this might change in the future and will need to be updated.
data_url = r"https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

# Directory to store the downloaded data.
data_dir = "data/cifar100/"

height, width, depth = 32, 32, 3
n_classes = 100
n_super_classes = 20
img_size_flat = height * width * depth

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 1

# Number of images for each batch-file in the training-set.
_images_per_file = 50000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file


def _load_batch(fpath, label_key='labels'):
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

def maybe_download():
    """
    Download and extract the CIFAR-100 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """

    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


def load_training_data(label_mode='fine'):
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    path = os.path.join(data_dir, dirname)
    fpath = os.path.join(path, 'train')
    x_train, y_train = _load_batch(fpath, label_key=label_mode + '_labels')

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)

    y_train_labels = one_hot_encoded(y_train, num_classes=n_classes)

    return x_train, np.asarray(y_train), y_train_labels


def load_test_data(label_mode='fine'):
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    path = os.path.join(data_dir, dirname)
    fpath = os.path.join(path, 'test')
    x_test, y_test = _load_batch(fpath, label_key=label_mode + '_labels')

    if K.image_data_format() == 'channels_last':
        x_test = x_test.transpose(0, 2, 3, 1)

    y_test_labels = one_hot_encoded(y_test, num_classes=n_classes)

    return x_test, np.asarray(y_test), y_test_labels
