#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:03:40 2018

@author: guy.hacohen
"""

from keras import backend as K, regularizers
from keras.engine.training import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    BatchNormalization, Activation, Input
import ModelLib


class Cifar100_Model(ModelLib.ModelLib):
    def build_classifier_model(self, dataset, n_classes=5,
                               activation='elu', dropout_1_rate=0.25,
                               dropout_2_rate=0.5,
                               reg_factor=50e-4, bias_reg_factor=None, batch_norm=False):
        
        n_classes = dataset.n_classes
        
        l2_reg = regularizers.l2(reg_factor) #K.variable(K.cast_to_floatx(reg_factor))
        l2_bias_reg = None
        if bias_reg_factor:
            l2_bias_reg = regularizers.l2(bias_reg_factor) #K.variable(K.cast_to_floatx(bias_reg_factor))

        # input image dimensions
        h, w, d = 32, 32, 3

        if K.image_data_format() == 'channels_first':
            input_shape = (d, h, w)
        else:
            input_shape = (h, w, d)

        # input image dimensions
        x = input_1 = Input(shape=input_shape)

        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Flatten()(x)
        x = Dense(units=512, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)


        x = Dropout(rate=dropout_2_rate)(x)
        x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation='softmax')(x)

        model = Model(inputs=[input_1], outputs=[x])
        return model
