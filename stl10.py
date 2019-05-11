#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:03:40 2018

@author: stenly
"""
import Dataset
import os
import urllib.request as urllib
import tarfile
import sys
import numpy as np
from dataset import one_hot_encoded


class Stl10(Dataset.Dataset):
    # Various directories and file-names.
    def __init__(self, smaller_data_size=None, normalize=True):
        self.name = 'stl10'
        self.data_url = r'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

        # Directory to store the downloaded data.
        self.data_dir = "../data/stl10/"

        # path to the binary train file with image data
        self.train_data_path = os.path.join(self.data_dir, 'stl10_binary/train_X.bin')

        # path to the binary test file with image data
        self.test_data_path = os.path.join(self.data_dir, 'stl10_binary/test_X.bin')

        # path to the binary train file with labels
        self.train_labels_path = os.path.join(self.data_dir, 'stl10_binary/train_y.bin')

        # path to the binary test file with labels
        self.test_labels_path = os.path.join(self.data_dir, 'stl10_binary/test_y.bin')

        # File containing the mappings between uid and string.
        self.path_uid_to_name = os.path.join(self.data_dir, 'stl10_binary/class_names.txt')

        self.height, self.width, self.depth = 96, 96, 3
        self.n_classes = 10
        super(Stl10, self).__init__(smaller_data_size, normalize)

    def read_labels(self, path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(self, path_to_data):
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)

            # We force the data into 3x96x96 chunks, since the
            # images are stored in "column-major order", meaning
            # that "the first 96*96 values are the red channel,
            # the next 96*96 are green, and the last are blue."
            # The -1 is since the size of the pictures depends
            # on the input file, and this way numpy determines
            # the size on its own.

            images = np.reshape(everything, (-1, self.depth, self.width, self.height))

            # Now transpose the images into a standard image format
            # readable by, for example, matplotlib.imshow
            # You might want to comment this line or reverse the shuffle
            # if you will use a learning algorithm like CNN, since they like
            # their channels separated.
            images = np.transpose(images, (0, 3, 2, 1))
            return images

    def maybe_download(self):
        """
        Download the stl10 dataset from the internet if it does not already
        exist in the data_dir.
        """

        # if the dataset already exists locally, no need to download it again.
        if all((
            os.path.exists(self.train_data_path),
            os.path.exists(self.test_data_path),
            os.path.exists(self.train_labels_path),
            os.path.exists(self.test_labels_path))):
            return

        dest_directory = self.data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)

        filename = self.data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                              float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.urlretrieve(self.data_url, filepath, reporthook=_progress)
            print('Downloaded', filename)
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def load_data(self, data_type='train'):
        # download the extract the dataset.
        self.maybe_download()
        if data_type == 'train':
            data_path = self.train_data_path
            labels_path = self.train_labels_path
        elif data_type == 'test':
            data_path = self.test_data_path
            labels_path = self.test_labels_path
        else:
            raise ValueError("only supports train or test data types")

        # load the train and test data and labels.
        x = self.read_all_images(data_path)
        y = self.read_labels(labels_path)

        # convert all images to floats in the range [0, 1]
        #    x_train = x_train.astype('float32')
        #    x_train = (x_train - 127.5) / 127.5

        # convert the labels to be zero based.
        y -= 1

        # convert labels to hot-one vectors.

        y_labels = one_hot_encoded(y, num_classes=self.n_classes)

        return x, y, y_labels

    def load_training_data(self):
        return self.load_data('train')

    def load_test_data(self):
        return self.load_data('test')

    def normalize_dataset(self):
        return
