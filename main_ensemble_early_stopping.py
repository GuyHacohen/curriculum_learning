#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:53:31 2018

@author: stenly
"""
import numpy as np

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#import stl10
#import cifar100
import datasets.cifar10
import datasets.cifar100_subset
import datasets.cifar100_custom_subset
import models.cifar100_model
import train_keras_model
import transfer_learning
#from keras.models import load_model
#import gc
import pickle
import argparse
import time
#import itertools
import scipy
#from sklearn import svm


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


def self_pace_exponent_data_function_generator(dataset, batches_to_increase,
                                               increase_amount,
                                               starting_percent, batch_size=100,
                                               anti=False):
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
                loss_order = balance_order(order_by_loss(dataset, model), dataset)
                if anti:
                    loss_order = loss_order[::-1]
                new_data = loss_order[:data_limit]
                cur_data_x = dataset.x_train[new_data, :, :, :]
                cur_data_y = dataset.y_train_labels[new_data, :]               
        return cur_data_x, cur_data_y

    return data_function


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


def order_by_same_network(dataset, model_lib, dropout_1_rate, dropout_2_rate,
                          reg_factor, bias_reg_factor, initial_lr, num_batchs,
                          batch_size, optimizer, lr_scheduler):
    model = model_lib.build_classifier_model(dataset,
                                             dropout_1_rate=dropout_1_rate, dropout_2_rate=dropout_2_rate,
                                             reg_factor=reg_factor,
                                             bias_reg_factor=bias_reg_factor)
    train_keras_model.compile_model(model, initial_lr=initial_lr,
                                    loss='categorical_crossentropy',
                                    optimizer=optimizer, metrics=['accuracy'])
    
    
    train_keras_model.train_model_batches(model, dataset.x_train, dataset.y_train_labels, dataset.x_test,
                                          dataset.y_test_labels, num_batchs, verbose=True,
                                          batch_size=batch_size,
                                          initial_lr=initial_lr,
                                          lr_scheduler=lr_scheduler, loss='categorical_crossentropy',
                                          optimizer=optimizer, Compile=False,
                                          model_output_path=None, metrics=['accuracy'])
    
    train_scores = model.predict(dataset.x_train)
    hardness_score = train_scores[list(range(size_train)), dataset.y_train]
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    return res


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
        data_function = train_keras_model.no_curriculum_data_function
    elif curriculum in ["curriculum", "vanilla", "anti", "random"]:
        data_function = exponent_data_function_generator(dataset, order, batch_increase, increase_amount,
                                                         starting_percent, batch_size=batch_size)

    elif curriculum == "self_pace" or curriculum == "anti_self_pace":
        anti_pace = False
        if curriculum == "anti_self_pace":
            anti_pace = True
        data_function = self_pace_exponent_data_function_generator(dataset, batch_increase, increase_amount,
                                                                   starting_percent, batch_size=batch_size,
                                                                   anti=anti_pace)
        
    else:
        print("unsupprted condition (not vanilla/curriculum/random/anti)")
        print("got the value:", curriculum)
        raise ValueError
    return data_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument("--generate_weights", default="False", help="dataset to use")
    parser.add_argument("--dataset", default="cifar100_subset_16", help="dataset to use")
#    parser.add_argument("--net_type", default="large", help="network size ..")
    parser.add_argument("--model", default="stVGG", help="which model to train to the dataset on")
    parser.add_argument("--model_dir", default=r'../models/', help="where to save the model")
    parser.add_argument("--output_name", default="", help="name of output file - will be added to model_dir")
    parser.add_argument("--verbose", default=True, type=bool, help="print more stuff")
    
    parser.add_argument("--optimizer", default="sgd", help="")
    parser.add_argument("--l2_reg", default=0, type=float)
    parser.add_argument("--bias_l2_reg", default=None, type=float)
    parser.add_argument("--dropout1", default=0.25, type=float)
    parser.add_argument("--dropout2", default=0.5, type=float)
    parser.add_argument("--curriculum", "-cl", default="curriculum")
    parser.add_argument("--curriculum_scheduler", default="adaptive")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_epochs", default=140, type=int)

    # lr params    
    parser.add_argument("--learning_rate", "-lr", default=0.03, type=float)
    parser.add_argument("--lr_decay_rate", default=1.6, type=float)
    parser.add_argument("--minimal_lr", default=1e-4, type=float)
    parser.add_argument("--lr_batch_size", default=500, type=int)    
    
    # curriculum params
    parser.add_argument("--batch_increase", default=100, type=int)
    parser.add_argument("--increase_amount", default=1.9, type=float)
    parser.add_argument("--starting_percent", default=100/2500, type=float)
    parser.add_argument("--order", default="inception", help="determine the order of the examples")
    
    parser.add_argument("--test_each", default=50, type=int, help="num batchs to run test-set after")
    parser.add_argument("--repeats", default=1, type=int, help="number of times to repeat the experiment")
    parser.add_argument("--save_each", default=None, type=int, help="numbers of batches before saving the model")
    parser.add_argument("--momentum", default=0.0, type=float)
    parser.add_argument("--balance", default=True, help="balance the ordering of the curriculum")
    

        
    parser.add_argument("--save_model", default=False)
    
    args = parser.parse_args()
            
        
    if args.momentum != 0.0 and not args.optimizer.endswith("sgd"):
        print(args.optimizer)
        print("do not support momentum for non-SGD optimizer")
        raise ValueError

    if args.dataset.startswith('cifar100_subset'):
        superclass_idx = int(args.dataset[len("cifar100_subset_"):])
        dataset = datasets.cifar100_subset.Cifar100_Subset(supeclass_idx=superclass_idx,
                                                  normalize=False)
        model_lib = models.cifar100_model.Cifar100_Model()
        
    elif args.dataset == "cifar100":
        subsets_idxes = "all"
        dataset = datasets.cifar100_custom_subset.Cifar100_Custom_Subset(normalize=False,
                                                                subsets_idxes=subsets_idxes)

    elif args.dataset == "cifar10":
        dataset = datasets.cifar10.Cifar10(normalize=False)
        model_lib = models.cifar100_model.Cifar100_Model()


    model_dir = args.model_dir
    size_train = dataset.x_train.shape[0]
    num_batchs = (args.num_epochs * size_train) // args.batch_size

    lr_scheduler = exponent_decay_lr_generator(args.lr_decay_rate,
                                               args.minimal_lr,
                                               args.lr_batch_size)

    normalized_flag = False

    classic_networks = ["vgg16", "vgg19", "inception", "xception", "resnet"]
    enhance_classic_networks = ["vgg16", "resnet", "inception"]
    if args.order in classic_networks:
        network_name = args.order
        if args.order == "inception":
            (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)

        else:
            (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_classic_networks(dataset,
                                                                                                                   network_name)

        train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
                                                                     transfer_values_test, dataset.y_test, dataset,
                                                                     network_name=network_name)
        order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train)

    elif args.order == "same_network":
        if not normalized_flag:
            dataset.normalize_dataset()
            normalized_flag = True
        order = order_by_same_network(dataset, model_lib, args.dropout1, args.dropout2,
                                      args.l2_reg, args.bias_l2_reg, args.learning_rate, num_batchs,
                                      args.batch_size, args.optimizer, lr_scheduler)
    else:
        print(args.order)
        print("wrong order input")
        raise ValueError
    
    
    if args.curriculum == "anti":
        order = np.flip(order, 0)
    elif args.curriculum == "random":
        np.random.shuffle(order)
    elif (args.curriculum not in ["None", "curriculum", "vanilla", "self_pace",
                                  "anti_self_pace"]):
        print("--curriculum value of %s is not supported!" % args.curriculum)
        raise ValueError
        
    if args.balance:
        order = balance_order(order, dataset)
        
    if not normalized_flag:
        dataset.normalize_dataset()
        normalized_flag = True

    if args.output_name:
        output_path = os.path.join(args.model_dir, args.output_name)
    else:
        output_path = None

    data_function = data_function_from_input(args.curriculum, args.batch_size,
                                             dataset, order, args.batch_increase,
                                             args.increase_amount, args.starting_percent)
    
    ## start expriment
    start_time_all = time.time()
    histories =[]
    for repeat in range(args.repeats):
        
        data_function = data_function_from_input(args.curriculum, args.batch_size,
                                         dataset, order, args.batch_increase,
                                         args.increase_amount, args.starting_percent)
        
        print("starting repeat number: " + str(repeat))
        model = model_lib.build_classifier_model(dataset,
                                                 dropout_1_rate=args.dropout1, dropout_2_rate=args.dropout2,
                                                 reg_factor=args.l2_reg,
                                                 bias_reg_factor=args.bias_l2_reg)
        train_keras_model.compile_model(model, initial_lr=args.learning_rate,
                                        loss='categorical_crossentropy',
                                        optimizer=args.optimizer, metrics=['accuracy'],
                                        momentum=args.momentum)
        history = train_keras_model.train_model_batches(model, dataset.x_train, dataset.y_train_labels, dataset.x_test,
                                                        dataset.y_test_labels, num_batchs, verbose=args.verbose,
                                                        batch_size=args.batch_size,
                                                        test_each=args.test_each,
                                                        initial_lr=args.learning_rate,
                                                        lr_scheduler=lr_scheduler, loss='categorical_crossentropy',
                                                        optimizer=args.optimizer, Compile=False,
                                                        metrics=['accuracy'],
                                                        data_function=data_function,
                                                        reduce_history=False,
                                                        save_each=args.save_each,
                                                        save_results=False,
                                                        net_num=repeat)

        histories.append(history)
        if output_path is not None:
            if args.save_model:
                model.save(output_path + "_model_num"+str(repeat))

    print("time all: --- %s seconds ---" % (time.time() - start_time_all))
    

    if output_path is not None:
        print('saving trained model to:', output_path)
        combined_history = histories[0].copy()
        for key in ["loss", "acc", "val_loss", "val_acc"]:
            results = np.zeros((args.repeats, len(histories[0][key])))
            for i in range(args.repeats):
                results[i, :] = histories[i][key]
            combined_history[key] = np.mean(results, axis=0)
            if key == "acc":
                combined_history["std_acc"] = scipy.stats.sem(results, axis=0)
            if key == "val_acc":
                combined_history["std_val_acc"] = scipy.stats.sem(results, axis=0)
        with open(output_path + "_history", 'wb') as file_pi:
            pickle.dump(combined_history, file_pi)
#            if args.save_model:
#                model.save(output_path)
        print(combined_history["loss"])

    train_predictions = model.predict(dataset.x_train)
    test_predictions = model.predict(dataset.x_test)
    print("training acc:", np.mean(np.argmax(train_predictions, axis=1) == dataset.y_train))
    print("test acc:", np.mean(np.argmax(test_predictions, axis=1) == dataset.y_test))
    