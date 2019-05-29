#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:33:53 2018

@author: stenly
"""
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import time
import os
import pickle

def train_model(model, x_train, y_train, x_test, y_test, model_output_path=None, batch_size=128,
                epochs=100, initial_lr=1e-3, lr_scheduler=None,
                loss='categorical_crossentropy', optimizer='adam', compile=True):
    ## compile the model
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(initial_lr)
    if compile:
        model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
        )
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(x_test, y_test),
        callbacks=[LearningRateScheduler(lr_scheduler)]
    )
    
    if model_output_path is not None:
        print('saving trained model to:', model_output_path)
        model.save(model_output_path)
    
    return history


def train_batchs(model, batch_generator, num_batchs):
    loss = []
    accuracy = []
    for i in range(num_batchs):
        batch_x, batch_y = next(batch_generator)
        cur_loss, cur_accuracy = model.train_on_batch(batch_x, batch_y)
        loss.append(cur_loss)
        accuracy.append(cur_accuracy)
    return loss, accuracy


def basic_data_function(x, y, cur_phase, num_phases, model):
    return x, y


def basic_lr_scheduler(initial_lr, batch_num, batch_size, size_train):
    return initial_lr


def basic_batch_generator(x, y, batch_size):
    size_data = x.shape[0]
    while True:
        cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
        yield x[cur_batch_idxs, :, :, :], y[cur_batch_idxs,:]


def compile_model(model, initial_lr=1e-3, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], momentum=0.0):
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                                  amsgrad=False)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(initial_lr, momentum=momentum)
    elif optimizer == 'modified_sgd':
        optimizer = ModifiedSGD(lr=initial_lr, momentum=momentum)
    else:
        print("optimizer not supported")
        raise ValueError
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)


def train_model_batch(model, x_train, y_train, x_test, y_test, num_training_phases=10, epoch_per_phase=[10]*10, batch_size=128,
                      batch_generator=basic_batch_generator, initial_lr=1e-3,
                      lr_scheduler=basic_lr_scheduler, loss='categorical_crossentropy', optimizer='adam', Compile=False,
                      model_save_scheduler=None, model_output_path=None, metrics=['accuracy'], data_function=basic_data_function,
                      test_each_epoch=False, verbose=False):

    if Compile:
        compile_model(model, initial_lr=initial_lr, loss=loss, optimizer=optimizer, metrics=metrics)


    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "batch_num": []}
    batch_num = 0
    for phase in range(num_training_phases):
        cur_x, cur_y = data_function(x_train, y_train, phase, num_training_phases, model)
        size_train = cur_x.shape[0]
        batches_per_epoch = size_train // batch_size
        batch_gen = batch_generator(cur_x, cur_y, batch_size)
        for epoch in range(epoch_per_phase[phase]):
            cur_lr = lr_scheduler(initial_lr, batch_num, batch_size, x_train.shape[0])
            K.set_value(model.optimizer.lr, cur_lr)
            start_time = time.time()
            if verbose:
                print("epoch: " + str(epoch + sum(epoch_per_phase[:phase])))
            cur_loss, curr_acc = train_batchs(model, batch_gen, batches_per_epoch)
            batch_num += batches_per_epoch
            if verbose:
                print("total number of batches done: " + str(batch_num))
                print("last lr used: " + str(cur_lr))
                print("epoch: " + str(epoch) + " phase: " + str(phase))
            history["loss"].append(np.mean(cur_loss))
            history["acc"].append(np.mean(curr_acc))
            history["batch_num"].append(batch_num)
            if test_each_epoch:
                cur_val_loss, cur_val_acc = model.evaluate(x_test, y_test, verbose=0)
                history["val_loss"].append(cur_val_loss)
                history["val_acc"].append(cur_val_acc)
            if model_save_scheduler is not None:
                to_save, cur_output_path = model_save_scheduler(epoch, epoch_per_phase[phase],
                                                                phase, num_training_phases)
                if to_save:
                    if verbose:
                        print('saving trained model to:', cur_output_path)
                    model.save(cur_output_path)
                    with open(cur_output_path + "_history", 'wb') as file_pi:
                        pickle.dump(history, file_pi)
            if verbose:
                print("--- %s seconds ---" % (time.time() - start_time))

    if model_output_path is not None:
        if verbose:
            print('saving trained model to:', model_output_path)
        model.save(model_output_path)
        with open(model_output_path + "_history", 'wb') as file_pi:
            pickle.dump(history, file_pi)

    return history


def no_curriculum_data_function(x_train, y_train, batch, history, model):
    return x_train, y_train

def naive_curriculum(x_train, y_train, batch, history):

    return x_train, y_train

def basic_lr_scheduler(initial_lr, batch, history):
    return initial_lr

def generate_random_batch(x, y, batch_size):
    size_data = x.shape[0]
    cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
    return x[cur_batch_idxs, :, :, :], y[cur_batch_idxs,:]


def train_model_batches(model, x_train, y_train, x_test, y_test, num_batches, batch_size=100,
                        test_each=50, batch_generator=generate_random_batch, initial_lr=1e-3,
                        lr_scheduler=basic_lr_scheduler, loss='categorical_crossentropy', optimizer='adam', Compile=False,
                        model_output_path=None, metrics=['accuracy'], data_function=no_curriculum_data_function,
                        verbose=False, reduce_history=True, save_each=None, save_results=True, net_num=0):
    

    
    if Compile:
        compile_model(model, initial_lr=initial_lr, loss=loss, optimizer=optimizer, metrics=metrics)
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "batch_num": [], "data_size": []}
    start_time = time.time()
    for batch in range(num_batches):
        cur_x, cur_y = data_function(x_train, y_train, batch, history, model)
        cur_lr = lr_scheduler(initial_lr, batch, history)
        K.set_value(model.optimizer.lr, cur_lr)
        batch_x, batch_y = batch_generator(cur_x, cur_y, batch_size)
        cur_loss, cur_accuracy = model.train_on_batch(batch_x, batch_y)
        history["loss"].append(cur_loss)
        history["acc"].append(cur_accuracy)
        history["data_size"].append(cur_x.shape[0])
        if test_each is not None and (batch+1) % test_each == 0:
            cur_val_loss, cur_val_acc = model.evaluate(x_test, y_test, verbose=0)
            history["val_loss"].append(cur_val_loss)
            history["val_acc"].append(cur_val_acc)
            history["batch_num"].append(batch)
            if verbose:
                print("val accuracy:", cur_val_acc)
        if verbose and (batch+1) % 5 == 0:
            print("batch: " + str(batch+1) + r"/" + str(num_batches))
            print("last lr used: " + str(cur_lr))
            print("data_size: " + str(cur_x.shape[0]))
            print("loss: " + str(cur_loss))
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
        if save_each is not None and model_output_path is not None:
            if batch % save_each == 0:
                if verbose:
                    print("saving model to: " + model_output_path + "_iter" + str(batch))
                if not save_results:
                    model.save(model_output_path + "_iter" + str(batch))
                else:
                    cur_model_path = model_output_path + "_net" + str(net_num) + "_iter" + str(batch)
                    res_test = model.predict(x_test)
                    res_train = model.predict(x_train)
                    with open(cur_model_path + "_res_test", 'wb') as file_pi:
                        pickle.dump(res_test, file_pi)
                    with open(cur_model_path + "_res_train", 'wb') as file_pi:
                        pickle.dump(res_train, file_pi)

    if test_each is not None and reduce_history:
        batchs_spaces = [0] + history["batch_num"]
        loss = [np.mean(history["loss"][batchs_spaces[i-1]:batchs_spaces[i]]) for i in
                range(1, len(batchs_spaces))]
        acc = [np.mean(history["acc"][batchs_spaces[i - 1]:batchs_spaces[i]]) for i in
               range(1, len(batchs_spaces))]
        history["loss"] = loss
        history["acc"] = acc

    if model_output_path is not None:
        print('saving trained model to:', model_output_path)
        model.save(model_output_path)
        with open(model_output_path + "_history", 'wb') as file_pi:
            pickle.dump(history, file_pi)

    return history



def early_stopping_save_scheduler(base_path, name, epochs_to_save):
    model_base_name = os.path.join(base_path, name + "_")

    def save_scheduler(epoch, max_epoch, phase, max_phase):
        if epoch % epochs_to_save == 0:
            return True, model_base_name + str(epoch) + "_" + str(phase)
        else:
            return False, None
    return save_scheduler

def plot_training_history(history):
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs = len(loss)
    
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].plot(np.arange(1, epochs + 1), loss, label='train')
    axs[0].plot(np.arange(1, epochs + 1), val_loss, label='test')
    axs[0].set_xlabel('Epoch number')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc="best")

    acc, val_acc = history.history['acc'], history.history['val_acc']
    axs[1].plot(np.arange(1, epochs + 1), acc, label='train')
    axs[1].plot(np.arange(1, epochs + 1), val_acc, label='test')
    axs[1].set_xlabel('Epoch number')
    axs[1].set_ylabel('Top-1 Accuracy')
    axs[1].legend(loc="best")
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, x_train, y_train, x_test, y_test):
    print('Train-set performance --> loss: {:.4f}, accuracy: {:.4f}'.format(*model.evaluate(x_train, y_train, verbose=0)))
    print('Test-set performance --> loss: {:.4f}, accuracy: {:.4f}'.format(*model.evaluate(x_test, y_test, verbose=0)))