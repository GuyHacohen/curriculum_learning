
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:19:32 2018

@author: stenly
"""


from keras import backend as K, regularizers
from keras.engine.training import Model
from keras.layers import Add, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Input
import transfer_learning
import numpy as np
import ModelLib


class Stl10_Model(ModelLib.ModelLib):

    def build_classifier_model(self, dataset, **kwargs):
        n_conv_blocks = 5  # number of convolution blocks to have in our model.
        n_filters = 32  # number of filters to use in the first convolution block.
        l2_reg = regularizers.l2(2e-4)  # weight to use for L2 weight decay.
        activation = 'elu'  # the activation function to use after each linear operation.

        if K.image_data_format() == 'channels_first':
            input_shape = (dataset.depth, dataset.height, dataset.width)
        else:
            input_shape = (dataset.height, dataset.width, dataset.depth)

        x = input_1 = Input(shape=input_shape)

        # each convolution block consists of two sub-blocks of Conv->Batch-Normalization->Activation,
        # followed by a Max-Pooling and a Dropout layer.
        for i in range(n_conv_blocks):
            shortcut = Conv2D(filters=n_filters, kernel_size=(1, 1), padding='same', kernel_regularizer=l2_reg)(x)
            x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
            x = BatchNormalization()(x)
            x = Activation(activation=activation)(x)

            x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
            x = Add()([shortcut, x])
            x = BatchNormalization()(x)
            x = Activation(activation=activation)(x)

            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(rate=0.25)(x)

            n_filters *= 2

        # finally, we flatten the output of the last convolution block, and add two Fully-Connected layers.
        x = Flatten()(x)
        x = Dense(units=512, kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)

        x = Dropout(rate=0.5)(x)
        x = Dense(units=dataset.n_classes, kernel_regularizer=l2_reg)(x)
        output = Activation(activation='softmax')(x)

        return Model(inputs=[input_1], outputs=[output])

    def lr_scheduler_generator(self, initial_lr):
        ## returns learning rate scheduler, which can be used as input to the fit
        ## function of keras models.
        ## the scheduler simply reduce the learning rate every 20 epochs.
        def lr_scheduler(epoch):
            if epoch < 20:
                return initial_lr
            elif epoch < 40:
                return initial_lr / 2
            elif epoch < 50:
                return initial_lr / 4
            elif epoch < 60:
                return initial_lr / 8
            elif epoch < 70:
                return initial_lr / 16
            elif epoch < 80:
                return initial_lr / 32
            elif epoch < 90:
                return initial_lr / 64
            else:
                return initial_lr / 128
        return lr_scheduler

    def lr_scheduler_phases(self, initial_lr, num_batch, batch_size, train_size):
        batchs_in_epoch = train_size // batch_size
        cur_epoch = (num_batch // batchs_in_epoch) + 1
        if cur_epoch < 20:
            return initial_lr
        elif cur_epoch < 40:
            return initial_lr / 2
        elif cur_epoch < 50:
            return initial_lr / 4
        elif cur_epoch < 60:
            return initial_lr / 8
        elif cur_epoch < 70:
            return initial_lr / 16
        elif cur_epoch < 80:
            return initial_lr / 32
        elif cur_epoch < 90:
            return initial_lr / 64
        else:
            return initial_lr / 128

    def phases_epochs(self):
        return [20, 20, 10, 10, 10, 10, 10, 10]

    # def corriculum_svm_based_training_data(self, x_train, y_train, x_test, y_test, dataset, anti_corriculum=False, random=False):
    #     (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
    #     train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, y_train,
    #                                                                  transfer_values_test, y_test, dataset)
    #     order = transfer_learning.rank_data_according_to_score(train_scores, y_train, reverse=anti_corriculum, random=random)
    #     size_data = x_train.shape[0]
    #     epochs_each_data = 10
    #     jumps = 0.1
    #     data_sizes = list(int(size_data * frac) for frac in (np.arange(0, 1, jumps) + jumps))
    #     epochs = [epochs_each_data] * len(data_sizes)
    #     total_batchs = sum(epoch*data_size for epoch, data_size in zip(epochs, data_sizes))
    #     total_batchs_original = 100 * size_data
    #     epochs[-1] += (total_batchs_original - total_batchs) // size_data
    #
    #     def data_function(x, y, cur_phase, num_phases):
    #         data_limit = data_sizes[cur_phase]
    #         new_data = order[:data_limit]
    #         return x[new_data, :, :, :], y[new_data, :]
    #
    #     return epochs, data_function

    def creatue_svm_data_scheduler(self):
        train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, y_train, transfer_values_test, y_test)
        order = transfer_learning.rank_data_according_to_score(train_scores, y_train)

        def data_scheduler(x, y, epoch):
            if epoch < 2:
                data_limit = np.int(np.ceil(len(x) * 0.1))  # 500
            elif epoch < 4:
                data_limit = np.int(np.ceil(len(x) * 0.2))  # 1000
            elif epoch < 6:
                data_limit = np.int(np.ceil(len(x) * 0.3))  # 1500
            elif epoch < 8:
                data_limit = np.int(np.ceil(len(x) * 0.4))  # 2000
            elif epoch < 10:
                data_limit = np.int(np.ceil(len(x) * 0.5))  # 2500
            elif epoch < 12:
                data_limit = np.int(np.ceil(len(x) * 0.6))  # 3000
            elif epoch < 14:
                data_limit = np.int(np.ceil(len(x) * 0.7))  # 3500
            elif epoch < 16:
                data_limit = np.int(np.ceil(len(x) * 0.8))  # 4000
            elif epoch < 18:
                data_limit = np.int(np.ceil(len(x) * 0.9))  # 4500
            elif epoch < 20:
                data_limit = np.int(np.ceil(len(x) * 0.95))  # 4750

            else:
                data_limit = np.int(np.ceil(len(x) * 1))  # 5000
            new_data = order[:data_limit]
            new_data = np.concatenate((new_data, np.random.choice(new_data, len(y_train) - len(new_data))))
            return x[new_data, :, :, :], y[new_data, :]
        return data_scheduler