#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:03:40 2018

@author: guy.hacohen
"""

import numpy as np

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.

    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]

    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.

    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.

    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

class Dataset():

    def __init__(self, normalize=True):
        self.maybe_download()
        self.update_data_set()
        self.normalized = False
        if normalize:
            self.normalize_dataset()
            self.normalized = True
            
    def update_train_test_cross_validate(self, train_idx, val_idx):
        
        self.x_test = self.x_train[val_idx, :, :, :]
        self.x_train = self.x_train[train_idx, :, :, :]
        self.y_test_labels = self.y_train_labels[val_idx, :]
        self.y_train_labels = self.y_train_labels[train_idx, :]
        self.y_test = self.y_train[val_idx]
        self.y_train = self.y_train[train_idx]
        
        self.test_size = self.y_test.size
        self.train_size = self.y_train.size        
                    
    def update_data_set(self):
        
        self.x_train, self.y_train, self.y_train_labels = self.load_training_data()
        self.x_test, self.y_test, self.y_test_labels = self.load_test_data()

        self.test_size = self.y_test.size
        self.train_size = self.y_train.size
        
        
    def normalize_dataset(self):
        raise NotImplementedError

    def maybe_download(self):
        raise NotImplementedError

    def load_training_data(self):
        raise NotImplementedError

    def load_test_data(self):
        raise NotImplementedError
