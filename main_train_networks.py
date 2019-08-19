#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:53:31 2018

@author: guy.hacohen
"""
import numpy as np

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import datasets.cifar100_subset
import datasets.cifar10
import datasets.cifar100
import models.cifar100_model
import train_keras_model
import transfer_learning
import pickle
import argparse
import time
import scipy
import matplotlib.pyplot as plt

def exponent_decay_lr_generator(decay_rate, minimum_lr, batch_to_decay):
    cur_lr = None
    def exponent_decay_lr(initial_lr, batch, history):
        nonlocal cur_lr
        if batch == 0:
            cur_lr = initial_lr
        if (batch % batch_to_decay) == 0 and batch !=0:
            new_lr = cur_lr / decay_rate
            cur_lr = max(new_lr, minimum_lr)
        return cur_lr
    return exponent_decay_lr

def exponent_data_function_generator(dataset, order, batches_to_increase,
                                     increase_amount, starting_percent,
                                     batch_size=100):

    size_data = dataset.x_train.shape[0]
    
    cur_percent = 1
    cur_data_x = dataset.x_train
    cur_data_y = dataset.y_test_labels
    
    
    def data_function(x, y, batch, history, model):
        nonlocal cur_percent, cur_data_x, cur_data_y
        
        if batch % batches_to_increase == 0:
            if batch == 0:
                percent = starting_percent
            else:
                percent = min(cur_percent*increase_amount, 1)
            if percent != cur_percent:
                cur_percent = percent
                data_limit = np.int(np.ceil(size_data * percent))
                new_data = order[:data_limit]
                cur_data_x = dataset.x_train[new_data, :, :, :]
                cur_data_y = dataset.y_train_labels[new_data, :]               
        return cur_data_x, cur_data_y

    return data_function

def order_by_loss(dataset, model):
    size_train = len(dataset.y_train)
    scores = model.predict(dataset.x_train)
    hardness_score = scores[list(range(size_train)), dataset.y_train]
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    return res

def balance_order(order, dataset):
    num_classes = dataset.n_classes
    size_each_class = dataset.x_train.shape[0] // num_classes
    class_orders = []
    for cls in range(num_classes):
        class_orders.append([i for i in range(len(order)) if dataset.y_train[order[i]] == cls])
    new_order = []
    ## take each group containing the next easiest image for each class,
    ## and putting them according to diffuclt-level in the new order
    for group_idx in range(size_each_class):
        group = sorted([class_orders[cls][group_idx] for cls in range(num_classes)])
        for idx in group:
            new_order.append(order[idx])
    return new_order


def data_function_from_input(curriculum, batch_size,
                             dataset, order, batch_increase,
                             increase_amount, starting_percent):
    if curriculum == "random":
        np.random.shuffle(order)
        
    if curriculum == "None" or curriculum == "vanilla":
        data_function = train_keras_model.basic_data_function
    elif curriculum in ["curriculum", "vanilla", "anti", "random"]:
        data_function = exponent_data_function_generator(dataset, order, batch_increase, increase_amount,
                                                         starting_percent, batch_size=batch_size)
        
    else:
        print("unsupprted condition (not vanilla/curriculum/random/anti)")
        print("got the value:", curriculum)
        raise ValueError
    return data_function


def load_dataset(dataset_name):
    
    if dataset_name.startswith('cifar100_subset'):
        superclass_idx = int(dataset_name[len("cifar100_subset_"):])
        dataset = datasets.cifar100_subset.Cifar100_Subset(supeclass_idx=superclass_idx,
                                                  normalize=False)
    
    elif dataset_name == "cifar10":
        dataset = datasets.cifar10.Cifar10(normalize=False)

    elif dataset_name == "cifar100":
        dataset = datasets.cifar100.Cifar100(normalize=False)
        
    else:
        print("do not support datset: %s" % dataset_name)
        raise ValueError

    return dataset


def load_model():
    return models.cifar100_model.Cifar100_Model()


def load_order(order_name, dataset):
    classic_networks = ["vgg16", "vgg19", "inception", "xception", "resnet"]
    if order_name in classic_networks:
        network_name = order_name
        if not transfer_learning.svm_scores_exists(dataset,
                                                   network_name=network_name):
            if order_name == "inception":
                (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
    
            else:
                (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_classic_networks(dataset,
                                                                                                                       network_name)
        else:
            (transfer_values_train, transfer_values_test) = (None, None)

        train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
                                                                     transfer_values_test, dataset.y_test, dataset,
                                                                     network_name=network_name)
        order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train)
        
    else:
        print("do not support order: %s" % args.order)
        raise ValueError
    
    return order


def combine_histories(history_list):
    
    num_repeats = len(history_list)
    
    combined_history = history_list[0].copy()
    for key in ["loss", "acc", "val_loss", "val_acc"]:
        size_key = len(history_list[0][key])
        results = np.zeros((num_repeats, size_key))
        for i in range(num_repeats):
            results[i, :] = history_list[i][key]
        combined_history[key] = np.mean(results, axis=0)
        if key == "acc":
            if num_repeats >1:
                combined_history["std_acc"] = scipy.stats.sem(results, axis=0)
            else:
                combined_history["std_acc"] = None
        if key == "val_acc":
            if num_repeats >1:
                combined_history["std_val_acc"] = scipy.stats.sem(results, axis=0)
            else:
                combined_history["std_val_acc"] = None
    
    return combined_history



def graph_from_history(history, plot_train=False, plot_test=True):
    
    fig, axs = plt.subplots(figsize=(10,5))
    
    if plot_train:
        x = np.array(history['batch_num'])
        y = history['acc'][x]
        error = history['std_acc'][x]
        
        if error is not None:
            axs.errorbar(x, y, error, marker='^', label="train")
        else:
            plt.plot(x, y, label="train")
    
    if plot_test:
        x = np.array(history['batch_num'])
        y = history['val_acc']
        error = history['std_val_acc']
        
        if error is not None:
            axs.errorbar(x, y, error, marker='^', label="test")
        else:
            plt.plot(x, y, label="test")
        
        
    
    axs.set_xlabel("batch number")
    axs.set_ylabel("accuracy")
    plt.legend()
#    axs.legend(loc="best")

def run_expriment(args):
    dataset = load_dataset(args.dataset)
    model_lib = load_model()

    size_train = dataset.x_train.shape[0]
    num_batches = (args.num_epochs * size_train) // args.batch_size

    lr_scheduler = exponent_decay_lr_generator(args.lr_decay_rate,
                                               args.minimal_lr,
                                               args.lr_batch_size)
    order = load_order(args.order, dataset)

    order = balance_order(order, dataset)    
    
    if args.curriculum == "anti":
        order = np.flip(order, 0)
    elif args.curriculum == "random":
        np.random.shuffle(order)
        
    elif (args.curriculum not in ["None", "curriculum", "vanilla"]):
        print("--curriculum value of %s is not supported!" % args.curriculum)
        raise ValueError
        
    dataset.normalize_dataset()
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = None
    
    ## start expriment
    start_time_all = time.time()
    histories =[]
    for repeat in range(args.repeats):
        
        data_function = data_function_from_input(args.curriculum,
                                                 args.batch_size,
                                                 dataset,
                                                 order,
                                                 args.batch_increase,
                                                 args.increase_amount,
                                                 args.starting_percent)
        
        print("starting repeat number: " + str(repeat + 1))
        model = model_lib.build_classifier_model(dataset)
        
        train_keras_model.compile_model(model,
                                        initial_lr=args.learning_rate,
                                        loss='categorical_crossentropy',
                                        optimizer="sgd")
        
        
        
        history = train_keras_model.train_model_batches(model,
                                                        dataset,
                                                        num_batches,
                                                        verbose=args.verbose,
                                                        batch_size=args.batch_size,
                                                        initial_lr=args.learning_rate,
                                                        lr_scheduler=lr_scheduler,
                                                        data_function=data_function)

        histories.append(history)
        
        
    print("time all: --- %s seconds ---" % (time.time() - start_time_all))
    
    
    combined_history = combine_histories(histories)
    
    if output_path:
        with open(output_path + "_history", 'wb') as file_pi:
            pickle.dump(combined_history, file_pi)
        
    print("training acc:", combined_history['acc'][-1])
    print("test acc:", combined_history['val_acc'][-1])
    
    graph_from_history(combined_history, plot_train=False, plot_test=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--dataset", default="cifar100_subset_16", help="dataset to use")
    parser.add_argument("--output_path", default=r'', help="where to save the model")
    parser.add_argument("--verbose", default=True, type=bool, help="print more stuff")
    
    parser.add_argument("--curriculum", "-cl", default="curriculum", help="which test case to use. supports: vanilla, curriculum, anti and random")
    parser.add_argument("--batch_size", default=100, type=int, help="determine batch size")
    parser.add_argument("--num_epochs", default=140, type=int, help="number of epochs to train on")
#    parser.add_argument("--num_epochs", default=10, type=int)

#     lr params    
    parser.add_argument("--learning_rate", "-lr", default=0.035, type=float)
    parser.add_argument("--lr_decay_rate", default=1.5, type=float)
    parser.add_argument("--minimal_lr", default=1e-4, type=float)
    parser.add_argument("--lr_batch_size", default=300, type=int)    
    
#    parser.add_argument("--learning_rate", "-lr", default=0.05, type=float, help="initial learning")
#    parser.add_argument("--lr_decay_rate", default=1.8, type=float, help="factor by which we drop learning rate exponentially")
#    parser.add_argument("--minimal_lr", default=1e-4, type=float, help="min learning rate we allow")
#    parser.add_argument("--lr_batch_size", default=500, type=int, help="interval of batches in which we drop the learning rate")    
    
    # curriculum params
    parser.add_argument("--batch_increase", default=100, type=int, help="interval of batches to increase the amount of data we sample from")
    parser.add_argument("--increase_amount", default=1.9, type=float, help="factor by which we increase the amount of data we sample from")
    parser.add_argument("--starting_percent", default=100/2500, type=float, help="percent of data to sample from in the inital batch")
    parser.add_argument("--order", default="inception", help="determine the order of the examples, supports transfer learning. options: inception, vgg16, vgg19, xception, resnet")
    
    parser.add_argument("--repeats", default=1, type=int, help="number of times to repeat the experiment")
    
    args = parser.parse_args()
    
    run_expriment(args)