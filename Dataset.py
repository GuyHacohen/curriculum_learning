import numpy as np
import os
import pickle

class Dataset():

    def __init__(self, smaller_data_size=None, normalize=True, cross_val=False,
                 val_size=500, order=None, order_name=""):
        self.maybe_download()
        self.update_data_set(smaller_data_size, order, order_name)
        if cross_val:
            self.name = self.name + "_cross_val"
            self.val_size = val_size
            self.height, self.width, self.depth = 32, 32, 3
            self.n_classes = 5
            self.n_super_classes = 1
            self._num_images_train = self._num_files_train * self._images_per_file - val_size
            self.train_idx, self.val_idx = self.cross_validation_idxes()
            self.update_train_test_cross_validate(self.train_idx, self.val_idx)
        if normalize:
            self.normalize_dataset()
            
    def update_train_test_cross_validate(self, train_idx, val_idx):
        
        self.x_test = self.x_train[val_idx, :, :, :]
        self.x_train = self.x_train[train_idx, :, :, :]
        self.y_test_labels = self.y_train_labels[val_idx, :]
        self.y_train_labels = self.y_train_labels[train_idx, :]
        self.y_test = self.y_train[val_idx]
        self.y_train = self.y_train[train_idx]
        
        self.test_size = self.y_test.size
        self.train_size = self.y_train.size        
        
        
    def cross_validation_idxes(self):
        train_order_file = os.path.join(self.data_dir,
                                        self.name + "_shuffled_train_order")
        if not os.path.exists(train_order_file):
            train_idx = []
            val_idx = []
            val_class_size = self.val_size // self.n_classes
            for cls in range(self.n_classes):
                class_idxes = [int(i) for i in range(self.train_size) if self.y_train[i] == cls]
                idxes_for_val = np.random.choice(class_idxes, size=val_class_size, replace=False)
                idxes_for_train = [int(i) for i in class_idxes if i not in idxes_for_val]
                val_idx = np.concatenate((val_idx, idxes_for_val))
                train_idx = np.concatenate((train_idx, idxes_for_train))
            train_idx = [int(i) for i in train_idx]
            val_idx = [int(i) for i in val_idx]
            np.random.shuffle(train_idx)
            np.random.shuffle(val_idx)
            
            assert(len(train_idx) == (self.train_size - self.val_size))
            assert(len(val_idx) == self.val_size)
            assert(np.all([i in np.concatenate((train_idx, val_idx)) for i in range(self.train_size)]))
            
            with open(train_order_file, 'wb+') as file_pi:
                pickle.dump((train_idx, val_idx), file_pi)
        else:
            with open(train_order_file, 'rb+') as file_pi:
                (train_idx, val_idx) = pickle.load(file_pi)
        return train_idx, val_idx
            
    def update_data_set(self, smaller_data_size=None, order=None, order_name=""):
        
        self.x_train, self.y_train, self.y_train_labels = self.load_training_data()
        self.x_test, self.y_test, self.y_test_labels = self.load_test_data()

        if smaller_data_size is not None:
            self.smaller_train_size = smaller_data_size;
            self.smaller_test_size = smaller_data_size;
            self.x_train = self.x_train[:self.smaller_train_size]
            self.y_train = self.y_train[:self.smaller_train_size]
            self.y_train_labels = self.y_train_labels[:self.smaller_train_size]
            self.x_test = self.x_test[:self.smaller_train_size]
            self.y_test = self.y_test[:self.smaller_train_size]
            self.y_test_labels = self.y_test_labels[:self.smaller_train_size]

        self.test_size = self.y_test.size
        self.train_size = self.y_train.size
        
        if order is not None:
            self.name += "_order_" + order_name
            all_x = np.concatenate((self.x_train, self.x_test), axis=0)
            all_y = np.concatenate((self.y_train, self.y_test), axis=0)
            all_y_labels = np.concatenate((self.y_train_labels, self.y_test_labels), axis=0)
            self.x_train = all_x[:self.train_size, :, :, :]
            self.x_test = all_x[-self.test_size:, :, :, :]
            self.y_train = all_y[:self.train_size]
            self.y_test = all_y[-self.test_size:]
            self.y_train_labels = all_y_labels[:self.train_size, :]
            self.y_test_labels = all_y_labels[-self.test_size:, :]
        
        
        
    def normalize_dataset(self):
        raise NotImplementedError

    def maybe_download(self):
        raise NotImplementedError

    def load_training_data(self):
        raise NotImplementedError

    def load_test_data(self):
        raise NotImplementedError
        
    def split_data_cross_validation(self, test_indexes, name_postfix="",
                                    just_train=False):
        """
        splitting data for cross validation:
        changing the database such that the given indexes will be its
        test set, and the rest the train set.
        indexes are assumed to be in range of (len(x_train) + len(y_train))
        and the resulting test size will be in size of len(indexes)
        """
        
        if not just_train:
            all_x = np.concatenate((self.x_train, self.x_test), axis=0)
            all_y = np.concatenate((self.y_train, self.y_test), axis=0)
            all_y_label = np.concatenate((self.y_train_labels, self.y_test_labels), axis=0)
        else:
            all_x = self.x_train
            all_y = self.y_train
            all_y_label = self.y_train_labels
        size_data = all_x.shape[0]
        train_indexes = [i for i in range(size_data) if i not in test_indexes]
        
        self.x_train = all_x[train_indexes, :, :, :]
        self.x_test = all_x[test_indexes, :, :, :]
        
        self.y_train = all_y[train_indexes]
        self.y_test = all_y[test_indexes]
        
        self.y_train_labels = all_y_label[train_indexes, :]
        self.y_test_labels = all_y_label[test_indexes, :]
        if name_postfix:
            self.name += name_postfix