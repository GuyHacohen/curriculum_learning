from keras import backend as K, regularizers
from keras.engine.training import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, \
    Input
import cifar100_subset
import transfer_learning
import numpy as np
import ModelLib


class Cifar100_Model(ModelLib.ModelLib):
    def build_classifier_model(self, dataset, model_type='large', n_classes=5, activation='elu', dropout_1_rate=0.25, dropout_2_rate=0.5,
                               reg_factor=200e-4, bias_reg_factor=None, batch_norm=False):
        
        n_classes = dataset.n_classes
        
        if model_type == 'large':
            return self._build_model_large(n_classes=n_classes, activation=activation, dropout_1_rate=dropout_1_rate,
                                           dropout_2_rate=dropout_2_rate, reg_factor=reg_factor,
                                           bias_reg_factor=bias_reg_factor, batch_norm=batch_norm)

        elif model_type == 'medium':
            return self._build_model_medium(n_classes=n_classes, activation=activation, dropout_1_rate=dropout_1_rate,
                                            dropout_2_rate=dropout_2_rate, reg_factor=reg_factor,
                                            bias_reg_factor=bias_reg_factor, batch_norm=batch_norm)

        if model_type == 'small':
            return self._build_model_small(n_classes=n_classes, activation=activation, dropout_1_rate=dropout_1_rate,
                                           dropout_2_rate=dropout_2_rate, reg_factor=reg_factor,
                                           bias_reg_factor=bias_reg_factor, batch_norm=batch_norm)
        else:
            return None

    def _build_model_large(self, n_classes=5, activation='elu', dropout_1_rate=0.25, dropout_2_rate = 0.5,
                           reg_factor=200e-4, bias_reg_factor=None, batch_norm=False):

        l2_reg = regularizers.l2(reg_factor) #K.variable(K.cast_to_floatx(reg_factor))
        l2_bias_reg = None
        if bias_reg_factor:
            l2_bias_reg = regularizers.l2(bias_reg_factor) #K.variable(K.cast_to_floatx(bias_reg_factor))

        # input image dimensions
        h, w, d = 32, 32, 3

        if K.image_data_format() == 'channels_first':
            input_shape = (3, h, w)
        else:
            input_shape = (h, w, 3)

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
        # model.l2_reg = l2_reg
        # model.l2_bias_reg = l2_bias_reg
        return model

    def _build_model_medium(self, n_classes=5,activation = 'elu',dropout_1_rate=0.25,dropout_2_rate = 0.5,
                               reg_factor = 200e-4, bias_reg_factor=None,batch_norm=False ):
        l2_reg = regularizers.l2(reg_factor)
        l2_bias_reg = None
        if bias_reg_factor:
            regularizers.l2(bias_reg_factor)

        # input image dimensions
        h, w, d = 32, 32, 3

        if K.image_data_format() == 'channels_first':
            input_shape = (3, h, w)
        else:
            input_shape = (h, w, 3)

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

        return Model(inputs=[input_1], outputs=[x])



    def _build_model_small(self, n_classes=5,activation = 'elu',dropout_1_rate=0.25,dropout_2_rate = 0.5,
                               reg_factor = 200e-4, bias_reg_factor=None,batch_norm=False ):
        l2_reg = regularizers.l2(reg_factor)
        l2_bias_reg = None
        if bias_reg_factor:
            regularizers.l2(bias_reg_factor)

        # input image dimensions
        h, w, d = 32, 32, 3

        if K.image_data_format() == 'channels_first':
            input_shape = (3, h, w)
        else:
            input_shape = (h, w, 3)

        # input image dimensions
        x = input_1 = Input(shape=input_shape)

        x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

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

        # x = Conv2D(filters=64, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        # if batch_norm:
        #     x = BatchNormalization()(x)
        # x = Activation(activation=activation)(x)
        # x = Conv2D(filters=64, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        # if batch_norm:
        #     x = BatchNormalization()(x)
        # x = Activation(activation=activation)(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(rate=dropout_1_rate)(x)

        x = Flatten()(x)
        x = Dense(units=64, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)


        x = Dropout(rate=dropout_2_rate)(x)
        x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation='softmax')(x)

        return Model(inputs=[input_1], outputs=[x])

    def phases_epochs(self):
        return [20, 20, 10, 10, 10, 10, 10, 10]

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


    # def corriculum_svm_based_training_data(x_train, y_train, x_test, y_test, dataset, anti_corriculum=False, random=False):
    #
    #     (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
    #     train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, y_train,
    #                                                                  transfer_values_test, y_test, dataset)
    #     order = transfer_learning.rank_data_according_to_score(train_scores, y_train, reverse=anti_corriculum, random=random)
    #     size_data = x_train.shape[0]
    #     epochs_each_data = 10
    #     jumps = 0.1
    #     data_sizes = list(int(size_data * frac) for frac in (np.arange(0, 1, jumps) + jumps))
    #     # epochs = [epochs_each_data] * len(data_sizes)
    #     epochs = [int(np.floor(epochs_each_data * size_data/data_size)) for data_size in data_sizes]
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
    
    