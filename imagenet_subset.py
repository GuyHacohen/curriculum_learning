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
import imagenet_database_utils
from keras.preprocessing import image
import pickle

# Various directories and file-names.
def array_to_str(arr):
    res = ""
    for i in arr:
        res += str(i) + "_"
    if arr:
        res = res[:-1]
    return res



class Imagenet_Subset(Dataset.Dataset):

    def __init__(self, normalize=True,
                 subsets_idxes=[10,20,30,40,50], subset_name="NoName"):
        if subsets_idxes == "all":
            self.subsets_idxes = list(range(1000))
            self.name = "imagenet"
        else:
            self.subsets_idxes = sorted(subsets_idxes)
            self.name = 'imagenet_' + subset_name

            
#        # Internet URL for the tar-file with the Inception model.
#        # Note that this might change in the future and will need to be updated.
#        self.data_url = r"https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        # Directory to store the downloaded data.
        self.train_data_dir = "/mnt/local/guy.hacohen/ILSVRC2012_img_train_preprocessed"
        self.val_data_dir = "/mnt/local/guy.hacohen/ILSVRC2012_img_val_preprocessed"
        self.metadata_path = "/cs/labs/daphna/guy.hacohen/imagenet/ILSVRC2012_devkit_t12/data/meta.mat"
        self.val_metadata_path = "/cs/labs/daphna/guy.hacohen/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        all_metadata = imagenet_database_utils.parse_metadata(self.metadata_path)
        self.metadata = [data_dict for data_dict in all_metadata if data_dict["id"] in subsets_idxes]
        self.height, self.width, self.depth = 224, 224, 3
        self.n_classes = len(self.subsets_idxes)
        self.img_size_flat = self.height * self.width * self.depth
        self.n_train_imgs = sum([clss["num_images"] for clss in self.metadata])
        super(Imagenet_Subset, self).__init__(normalize=normalize)
        
    def maybe_download(self):
        """
        Download and extract the CIFAR-100 data-set if it doesn't already exist
        in data_path (set this variable first to the desired path).
        """
        return True

    def load_training_data(self):
        x_train = np.zeros((0, self.height, self.width, self.depth), dtype=np.uint8)
        y_train = np.zeros((0), dtype=np.uint16)
        for data_dict in self.metadata:
            wnid = data_dict["wnid"]
            cur_x_train = np.zeros((data_dict["num_images"], self.height,
                                    self.width, self.depth), dtype=np.uint8)
            path = os.path.join(self.train_data_dir, wnid)
            x_images = os.listdir(path)
            assert(len(x_images) == data_dict["num_images"])
            for idx, x_image in enumerate(x_images):
                image_path = os.path.join(path, x_image)
                img = image.load_img(image_path, target_size=(self.height, self.width))
                cur_x_train[idx, :, :, :] = img
            cur_y_train = [data_dict["id"]] * data_dict["num_images"]
            x_train = np.concatenate((x_train, cur_x_train), axis=0)
            y_train = np.concatenate((y_train, cur_y_train), axis=0)
        
        y_train_values = sorted(list(set(y_train)))
        assert(len(y_train_values) == self.n_classes)
        map_dict = {val: i for i, val in enumerate(y_train_values)}
        for i, y in enumerate(y_train):
            y_train[i] = map_dict[y]
#        print(y_train.shape)
        
        order_randomizer_path = os.path.join(self.train_data_dir, "train_order_randomizer_" + self.name)
        if os.path.exists(order_randomizer_path):
            with open(order_randomizer_path, "rb+") as out_file:
                train_order = pickle.load(out_file)
        else:
            train_order = np.arange(len(y_train))
            np.random.shuffle(train_order)
            with open(order_randomizer_path, "wb+") as out_file:
                 pickle.dump(train_order, out_file)
        
        x_train = x_train[train_order, :, :, :]
        y_train = y_train[train_order]
        y_train_labels = one_hot_encoded(y_train, num_classes=self.n_classes)
        return x_train, y_train, y_train_labels

    def load_test_data(self):
        y_test = imagenet_database_utils.parse_validation_gt(self.val_metadata_path)
        relevant_images_names = ["ILSVRC2012_val_000" + imagenet_database_utils.pad_int_into_str(i+1) + ".JPEG"
                                 for i, label in enumerate(y_test)
                                 if label in self.subsets_idxes]
        assert(len(relevant_images_names) == 50 * len(self.subsets_idxes))
        x_test = np.zeros((len(relevant_images_names), self.height, self.width,
                           self.depth), dtype=np.uint8)
        y_test = np.array([label for label in y_test if label in self.subsets_idxes])
        y_test_values = sorted(list(set(y_test)))
        assert(len(y_test_values) == self.n_classes)
        map_dict = {val: i for i, val in enumerate(y_test_values)}
        for i, y in enumerate(y_test):
            y_test[i] = map_dict[y]
        for idx, image_name in enumerate(relevant_images_names):
            image_path = os.path.join(self.val_data_dir, image_name)
            img = image.load_img(image_path, target_size=(self.height, self.width))
            x_test[idx, :, :, :] = img
        y_test_labels = one_hot_encoded(y_test, num_classes=self.n_classes)
        return x_test, y_test, y_test_labels
#        dirname = 'cifar-100-python'
#        path = os.path.join(self.data_dir, dirname)
#        fpath = os.path.join(path, 'test')
#        x_test, y_test_fine = self._load_batch(fpath, 'fine_labels')
#        data_size = len(y_test_fine)
#
#        if K.image_data_format() == 'channels_last':
#            x_test = x_test.transpose(0, 2, 3, 1)
#            
#        relevant_idxes = [i for i in range(data_size) if y_test_fine[i] in self.subsets_idxes]
#        x_test = x_test[relevant_idxes, :, :, :]
#        y_test = np.asarray(y_test_fine)[relevant_idxes]
#        y_test_values = sorted(list(set(y_test)))
#        assert(len(y_test_values) == self.n_classes)
#        map_dict = {val: i for i, val in enumerate(y_test_values)}
#        for i, y in enumerate(y_test):
#            y_test[i] = map_dict[y]
#
#        y_test_labels = one_hot_encoded(y_test, num_classes=self.n_classes)
#        return x_test, y_test, y_test_labels

    def normalize_dataset(self):
        self.x_train = (self.x_train - 128.) / 128
        # x_test = x_test.astype('float32')
        self.x_test = (self.x_test - 128.) / 128


