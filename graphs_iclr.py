#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:41:08 2018

@author: guy.hacohen
"""

import numpy as np
#import cifar100_subset
import draw_results
import pickle
import itertools
import matplotlib.pyplot as plt
#import transfer_learning
import time
#import cifar100_model
#import cifar100_subset_validation
#import train_keras_model
import os
import pandas as pd
#import keras.backend as K
import itertools
#import tensorflow as tf
#import scipy
#import keras
import re
import itertools
from scipy import stats
#from graphs_iclr_toolbox import grad_directions_results, get_classic_order
import matplotlib
import seaborn as sns
sns.set()
sns.set_style("whitegrid")


def plot_history_list(history_list, names, title, bold_first=False, tag="acc",
                      colors=None, bars=True, xlim=None, ylim=None, figsize=(10,5),
                      tiny_text=12, small_text=17, mid_text=22, big_text=27,
                      visible_x=True, visible_y=True, legend=True,
                      x_title=True, y_title=True, no_background=False,
                      barsize=(3,2), legend_loc="upper left", bar_x=False,
                      ylim_bar=None):
    
    TINY_SIZE = tiny_text
    SMALL_SIZE = small_text
    MEDIUM_SIZE = mid_text
    BIGGER_SIZE = big_text
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
#    plt.figure()
    histories = []
    for history_path in history_list:
        with open(history_path, "rb+") as history_file:
            history = pickle.load(history_file)
        histories.append(history)
    draw_results.plot_keras_history_2(histories[0], histories[1:],
                                  train=False, name1=names[0],
                                  names2=names[1:],
                                  bold=bold_first,
                                  y_tag=tag,
                                  colors=colors,
                                  error=True,
                                  figsize=figsize,
                                  visible_x=visible_x,
                                  visible_y=visible_y,
                                  legend=legend,
                                  legend_loc=legend_loc,
                                  x_title=x_title,
                                  y_title=y_title,
                                  no_background=no_background)
    # draw_results.plot_keras_history_2(history_curriculum2)
#    plt.ylim([0.50, 0.57])
#    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title)
    
    plt.show()
    
    
    if bars:
        if tag == "loss":
            std = False
        else:
            std = True
#        plt.figure()
        bar_plot_histories(histories, names, "val_" + tag, std=std, bold=bold_first,
                           colors=colors, figsize=barsize, bar_x=bar_x)
        plt.title(title)
        plt.tight_layout()
        if ylim_bar is not None:
            plt.ylim(ylim_bar)
        plt.show()

def bar_plot_histories(histories, names, tag, std=False, base_line_history=None, base_line_name="baseline",
                       colors=None, bold=False, figsize=(3,2), bar_x=False):
    
    SMALL_SIZE = 14
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 27
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    scale_down = 1
    if base_line_history is not None:
        histories += [base_line_history]
        names += [base_line_name]
    final_tag = [history[tag][-1] for history in histories]
    if std:
        stds = [history["std_" + tag][-1] for history in histories]
    bar_locations = np.arange(len(histories)) / scale_down
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.xaxis.grid()
#    fig, ax = plt.subplots(1, 1, figsize=(5,5))
#    fig, ax = plt.subplots(1, 1)
    if colors is None:
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(histories))]
    else:
        if bold:
            colors = ["grey"] + colors
    if std:
        ax.bar(bar_locations, final_tag, color=colors, yerr=stds,
               capsize=3)
    else:
        ax.bar(bar_locations, final_tag, color=colors)
    y_pos = np.arange(len(names)) / (scale_down)
    plt.xticks(y_pos, names, rotation='vertical')
    
    if not bar_x:
        ax.xaxis.set_visible(False)
#    plt.xticks(y_pos, names)
    low = min(final_tag)
    high = max(final_tag)
    plt.ylim([low-0.01, high+0.01])
    plt.tight_layout()
    fig.patch.set_facecolor('None')
    fig.patch.set_alpha(0.0)
#    fig.savefig('/cs/labs/daphna/guy.hacohen/project/graphs/to_daphna/iclr2019/problem1_curriculum_vanilla_anti_random_bars.png')
#    plt.xlim([-0.1, (len(histories)-1)/scale_down + 0.1])
#    plt.grid()
#    high = max(final_tag)
    
#    plt.legend(names, loc="best")




def choose_lr_parameters(base_output, params):
    regex = "\$\{(.+?)\}+?"   # match stuff in ${something} format
    params_in_base_output = re.findall(regex, base_output)
    param_order = [param for param in list(params.keys()) if param in params_in_base_output]
    results = np.zeros([len(params[param]) for param in param_order])
    models_checked = 0
    memoize = {}
    for idx_tuple, param_tuple in zip(itertools.product(*[range(len(params[key])) for key in param_order]),
                                      itertools.product(*[params[key] for key in param_order])):
        try:
            
            output_path = base_output
            for param_idx, param_name in enumerate(param_order):
                output_path = output_path.replace(r"${" + param_name + "}",
                                                  str(param_tuple[param_idx]))
            if output_path not in memoize:
                memoize[output_path] = 0
            else:
                continue
#            print(output_path)
            with open(output_path, "rb+") as history_file:
                history = pickle.load(history_file)
            results[idx_tuple] = np.mean(history["val_acc"][-5:])
#            results[idx_tuple] = np.trapz(history["val_acc"], dx=50)
#            results[idx_tuple] = scipy.integrate.simps(history["val_acc"], dx=50)
            models_checked += 1
        except:
            results[idx_tuple] = -float("inf")
            

#    print(sorted(results.flatten()))
    best_idxes_list = results.flatten().argsort()[-3:][::-1]
    best_idxes_list = [np.unravel_index(i, results.shape) for i in best_idxes_list]
    for idx in best_idxes_list:
        print(param_order)
        print([params[param_order[i]][idx[i]] for i in range(len(param_order))])
    best_idx = np.unravel_index(results.argmax(), results.shape)
    if results[best_idx] == -float("inf"):
        return None
    best_params = [params[param_order[i]][best_idxes_list[0][i]] for i in range(len(param_order))]
    best_history = base_output
    
    best_params_to_print = {}
    for param_idx, param_name in enumerate(param_order):
        old_history = best_history
        best_history = best_history.replace(r"${" + param_name + "}",
                                          str(best_params[param_idx]))
        if best_history != old_history:
            best_params_to_print[param_name] = best_params[param_idx]
    print(best_params_to_print)
    print("models checked: ", models_checked)
    return best_history

base_3 = r'C:\Users\stenl\Documents\GitHub\curriculum_project\models\models3\\'[:-1]
base_4 = r'C:\Users\stenl\Documents\GitHub\curriculum_project\models\models4\\'[:-1]

##### SUMMARY
# original curriculum
example_output_path_curriculum = base_4 + r"adjust_lr_long_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_random = base_4 + r"adjust_lr_long_16_random_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_anti = base_4 + r"adjust_lr_long_16_anti_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# original vanila
example_output_path_vanilla = base_4 + r"adjust_lr_long_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# same network
example_output_path_same_network_curriculum = base_4 + r"adjust_lr_long_norm_same_network_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_same_network_anti = base_4 + r"adjust_lr_long_norm_same_network_16_anti_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# single step
example_output_path_single_step_curriculum = base_4 + r"adjust_lr_long_16_curriculum_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_single_step_anti = base_4 + r"adjust_lr_long_16_anti_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_single_step_random = base_4 + r"adjust_lr_long_16_random_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"

# self pace
example_output_path_self_pace = base_4 + r"adjust_lr_long_16_self_pace_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_anti_self_pace = base_4 + r"adjust_lr_long_16_anti_self_pace_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# jmp change
example_output_path_change_jumps = base_4 + r"adjust_lr_long_16_curriculum_sgd_expo_change_jumps_${cng_jmp1}_${cng_jmp2}_${cng_jmp3}_${cng_jmp4}_${cng_jmp5}_balance_lr_0.05_100_2_0.04_decay1.7_min1e-4_lrjmp600_history"
example_output_path_change_jumps_random = base_4 + r"adjust_lr_long_16_random_sgd_expo_change_jumps_${cng_jmp1}_${cng_jmp2}_${cng_jmp3}_${cng_jmp4}_${cng_jmp5}_balance_lr_0.05_100_2_0.04_decay1.7_min1e-4_lrjmp600_history" 

#2 jumps
example_output_two_jumps = base_4 + r"adjust_lr_longer_16_curriculum_sgd_expo_2_first_jumps_${chng1}_${chng2}_balance_lr_${lrate}_100_1.9_${strt}_decay1.6_min1e-4_lrjmp600_history"

exmaple_output_cifar_10_curriculum = base_3 + r"adjust_lr_cifar10_36epochs_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
exmaple_output_cifar_10_vanilla = base_3 + r"adjust_lr_cifar10_36epochs_vanilla_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"

exmaple_output_cifar_100_curriculum = base_3 + r"adjust_lr_cifar100_48epochs_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
exmaple_output_cifar_100_vanilla = base_3 + r"adjust_lr_cifar100_48epochs_vanilla_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"

example_output_vgg_cifar100_curriculum = base_4 + r"cifar100_vgg_long_curriculum_single_step_no_augmentation_lr_${lrate}_jmp${jmp}_strt${strt}_history"
example_output_vgg_cifar100_vanilla = base_3 + r"cifar100_vgg_vanilla_no_augmentation_history"
example_output_vgg_cifar100_2_jumps = base_4 + r"cifar100_vgg_long_curriculum_single_step_no_augmentation_lr_${lrate}_jmp${jmp}_strt${strt}_inc1.9_chng1_${chng1}_chng2_${chng2}_history"

example_output_vgg_cifar10_curriculum = base_4 + r"cifar10_vgg_long_curriculum_single_step_no_augmentation_lr_${lrate}_jmp${jmp}_strt${strt}_history"
example_output_vgg_cifar10_vanilla = base_3 + r"cifar10_vgg_vanilla_no_augmentation_history"
example_output_vgg_cifar10_2_jumps = base_4 + r"cifar10_vgg_long_curriculum_single_step_no_augmentation_lr_${lrate}_jmp${jmp}_strt${strt}_inc1.9_chng1_${chng1}_chng2_${chng2}_history"
example_output_vgg_cifar10_same_net = base_4 + r"cifar10_vgg_long_same_network_curriculum_single_step_no_augmentation_lr_${lrate}_jmp${jmp}_strt${strt}_inc1.9_history"


#stl10
example_stl10_curriculum = base_3 + r"adjust_lr_stl10_curriculum_sgd_expo_balance_lr_${lrate}_100_1.9_${strt}_decay${decay}_min1e-4_lrjmp${lrjump}_history"
example_stl10_vanilla = base_3 + r"adjust_lr_stl10_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_${strt}_decay${decay}_min1e-4_lrjmp${lrjump}_history"


#validation
example_validation_curriculum = base_4 + r"validation_16_curriculum_sgd_expo_balance_lr_${lrate}_100_1.9_${strt}_decay${decay}_min1e-4_lrjmp{lrjump}_history"
example_validation_same = base_4 + r"validation_16_same_network_curriculum_sgd_expo_balance_lr_${lrate}_100_1.9_${strt}_decay${decay}_min1e-4_lrjmp{lrjump}_history"
example_validation_vanilla = base_4 + r"validation_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.05_decay${decay}_min1e-4_lrjmp{lrjump}_history"
example_validation_anti_single_step = base_4 + r"validation_long_norm_inception_16_anti_sgd_single_step_balance_lr_${lrate}_100_0.05_decay${decay}_min1e-4_lrjmp{lrjump}_history"
example_validation_curriculum_single_step = base_4 + r"validation_long_norm_inception_16_curriculum_sgd_single_step_balance_lr_${lrate}_100_0.05_decay${decay}_min1e-4_lrjmp{lrjump}_history"
example_validation_random_single_step = base_4 + r"validation_long_norm_inception_16_random_sgd_single_step_balance_lr_${lrate}_100_0.05_decay${decay}_min1e-4_lrjmp{lrjump}_history"

# github reproduce
base_git = r"/cs/labs/daphna/guy.hacohen/project/ICML2019/github/models/"
example_github_curriculum = base_git + r"grid_channel_wise_norm_subset_16_curriculum_lr_${lrate}_jmp_${jmp}_inc_1.9_start_0.04_decay_${decay}_min_1e-4_lrjmp_${lrjump}_history"
example_github_vanilla = base_git + r"grid_channel_wise_norm_subset_16_vanilla_lr_${lrate}_jmp_100_inc_1.9_start_0.04_decay_${decay}_min_1e-4_lrjmp_${lrjump}_history"

COLORS = ["blue", "red", "green", "yellow", "purple", "orange", "pink", "brown"]


#base_weights = r"/cs/labs/daphna/guy.hacohen/project/models_weights/"
vanilla_models_path = r"single_model_repeat_${i}_subset_16_vanilla_sgd_expo_balance_lr_0.004_100_1.9_0.04_decay1.6_min1e-4_lrjmp600"

def problem1_curriculum_vanilla_anti_random_graph(curriculum_base, random_base,
                                                  anti_base, vanilla_base):

    num_non_vanilla = 3
    
    in_graph_params = {
            "lrate": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045,
                      0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01],
            "decay": [2, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1],
            "lrjump": [200 ,300, 400, 500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [100],
            "strt": [0.04],
            }
    

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    curriculum_grid_search = choose_lr_parameters(curriculum_base, in_graph_params)
    anti_grid_search = choose_lr_parameters(anti_base, in_graph_params)
    random_grid_search = choose_lr_parameters(random_base, in_graph_params)
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)
    plot_history_list([vanilla_grid_search,
                       curriculum_grid_search,
                       anti_grid_search,
                       random_grid_search],
                       ["Vanilla",
                        "Curriculum",
                        "Anti Curriculum",
                        "Random"],
                        "",
                        bold_first=True,
                        colors=colors,
                        figsize=(10,4))
    

def problem1_github_graph(curriculum_base, vanilla_base):
#                          anti_base, random_base):

    num_non_vanilla = 3

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    in_graph_params = {
        "lrate": [0.025, 0.03, 0.035, 0.04, 0.045],
        "decay": [1.1, 1.3, 1.5, 1.7, 1.9],
        "lrjump": [100, 300, 500, 700, 900],
        "minimal": ["1e-4"],
        "inc": [1.9],
        "jmp": [50, 75, 100, 200, 300, 400, 500],
        "strt": [0.04],
        }

    curriculum_grid_search = choose_lr_parameters(curriculum_base, in_graph_params)
#    anti_grid_search = choose_lr_parameters(anti_base, in_graph_params)
#    random_grid_search = choose_lr_parameters(random_base, in_graph_params)
    
    
    in_graph_params = {
        "lrate": [0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
                  0.05, 0.055, 0.06, 0.065, 0.07],
        "decay": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        "lrjump": [100, 200, 300, 400, 500,
                   600, 700, 800, 900],
        "minimal": ["1e-4"],
        "inc": [1.9],
        "jmp": [100],
        "strt": [0.04],
        }
        
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)
    plot_history_list([vanilla_grid_search,
                       curriculum_grid_search],
#                       anti_grid_search,
#                       random_grid_search],
                       ["Vanilla",
                        "Curriculum"],
#                        "Anti Curriculum",
#                        "Random"],
                        "",
                        bold_first=True,
                        colors=colors)


def problem23_curriculum_vanilla_graph(curriculum_base, vanilla_base,
                                       cifar_name):
    num_non_vanilla = 1
    
    in_graph_params = {
            "lrate": [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
            "decay": [2, 1.7, 1.6, 1.5, 1.3, 1.1],
            "lrjump": [200 ,300, 400, 500, 600, 700],
            "minimal": ["1e-3"],
            "inc": [1.1, 1.5, 1.9, 2.5],
            "jmp": [100, 300, 600, 1000, 1500, 2000],
            "strt": [0.04, 0.06, 0.08, 0.1, 0.12],
            }
    

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    curriculum_grid_search = choose_lr_parameters(curriculum_base, in_graph_params)
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)

    plot_history_list([vanilla_grid_search,
                       curriculum_grid_search],
                       ["Vanilla",
                        "Curriculum"],
                        "",
                        bold_first=True,
                        colors=colors,
                        bars=False,
                        figsize=(7.5,4))
    
    if cifar_name == "Cifar-10":
        ylim=[0.74, 0.77]
        xlim=[8000, 18200]
    elif cifar_name == "Cifar-100":
        ylim=[0.37, 0.41]
        xlim=[16000, 24200]
        
    plot_history_list([vanilla_grid_search,
                   curriculum_grid_search],
                   ["Vanilla",
                    "Curriculum"],
                    "",
                    bold_first=True,
                    colors=colors,
                    bars=False,
                    ylim=ylim,
                    xlim=xlim,
                    figsize=(3,2),
                    tiny_text=6, small_text=15, mid_text=18, big_text=21,
                    legend=False,
                    x_title=False,
                    y_title=False,
                    no_background=True
                    )


def problem45_vgg_curriculum_vanilla_graph(curriculum_base, vanilla_base,
                                           two_jump_base, cifar_name):
    
    num_non_vanilla = 2

    in_graph_params = {
            "lrate": [0.12, 0.1, 0.08, 0.06, 0.04],
            "jmp": [250, 500, 1000, 1500, 2500, 5000],
            "strt": [0.02, 0.04, 0.08, 0.1, 0.2],
            }
    

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    curriculum_grid_search = choose_lr_parameters(curriculum_base, in_graph_params)
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)

    in_graph_params = {
            "lrate": [0.1, 0.09],
            "strt": [0.04, 0.08],
            "jmp": [350, 500, 750],
            "chng1": [75, 100, 125, 150, 200],
            "chng2": [100, 200],
            }
    two_jump_grid_search = choose_lr_parameters(two_jump_base, in_graph_params)
    plot_history_list([vanilla_grid_search,
#                       curriculum_grid_search,
                       two_jump_grid_search],
                       ["Vanilla",
#                        "curriculum",
                        "Curriculum"],
                        "",
                        bold_first=True,
                        colors=colors,
                        figsize=(7.5,4),
                        barsize=(2,2))


def problem6_stl10_curriculum_vanilla_graph(curriculum_base, vanilla_base):
    
    num_non_vanilla = 1

    in_graph_params = {
            "lrate": [0.2, 0.15, 0.14, 0.13, 0.12, 0.1, 0.08, 0.06, 0.04],
            "decay": [2, 1.6, 1.2],
            "lrjump": [400, 600, 800, 1000, 1400, 1800],
            "strt": [0.02, 0.04, 0.08, 0.12],
            }
    

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    curriculum_grid_search = choose_lr_parameters(curriculum_base, in_graph_params)
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)
    
    plot_history_list([vanilla_grid_search,
                       curriculum_grid_search],
                       ["Vanilla",
                        "Curriculum"],
                        "",
                        bold_first=True,
                        colors=colors)

def problem1_single_step_curriculum_vanilla_graph(curriculum_base, vanilla_base,
                                                  anti_base, random_base):
    num_non_vanilla = 3
    
    in_graph_params = {
            "lrate": [0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035,
                      0.03, 0.025, 0.02],
            "decay": [1.7, 1.6],
            "lrjump": [500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [50, 70, 100, 200, 300],
            "strt": [0.04, 0.06, 0.08, 0.1, 0.12],
            }
    

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    curriculum_grid_search = choose_lr_parameters(curriculum_base, in_graph_params)
    random_grid_search = choose_lr_parameters(random_base, in_graph_params)
    anti_grid_search = choose_lr_parameters(anti_base, in_graph_params)

    in_graph_params = {
        "lrate": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045,
                  0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01],
        "decay": [2, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1],
        "lrjump": [200 ,300, 400, 500, 600],
        "minimal": ["1e-4"],
        "inc": [1.9],
        "jmp": [100],
        "strt": [0.04],
        }
    
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)
    
    plot_history_list([vanilla_grid_search,
                       curriculum_grid_search,
                       anti_grid_search,
                       random_grid_search],
                       ["Vanilla",
                        "Curriculum",
                        "Anti",
                        "Random"],
                        "",
                        bold_first=True,
                        colors=colors,
                        figsize=(7.5,4),
                        xlim=(-490,3900),
                        ylim=(0.31, 0.59),
                        tiny_text=11)


def problem1_same_network_self_pace_curriculum_vanilla_graph(same_network_base, vanilla_base,
                                                             anti_base, random_base,
                                                             self_pace_base, anti_self_pace_base):
    num_non_vanilla = 5
        
    in_graph_params = {
            "lrate": [0.045, 0.04, 0.035, 0.03, 0.025, 0.02],
            "decay": [1.7, 1.6],
            "lrjump": [500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [50, 70, 100, 200, 300],
            "strt": [0.04, 0.06, 0.08, 0.1, 0.12],
            }
    

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    same_network_grid_search = choose_lr_parameters(same_network_base, in_graph_params)
    anti_same_network_grid_search = choose_lr_parameters(anti_base, in_graph_params)

    in_graph_params = {
            "lrate": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045,
                      0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01],
            "decay": [2, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1],
            "lrjump": [200, 300, 400, 500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [100],
            "strt": [0.04],
            }
    
    self_pace_grid_search = choose_lr_parameters(self_pace_base, in_graph_params)
    anti_self_pace_grid_search = choose_lr_parameters(anti_self_pace_base, in_graph_params)

    in_graph_params = {
        "lrate": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045,
                  0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01],
        "decay": [2, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1],
        "lrjump": [200 ,300, 400, 500, 600],
        "minimal": ["1e-4"],
        "inc": [1.9],
        "jmp": [100],
        "strt": [0.04],
        }
    
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)
    random_grid_search = choose_lr_parameters(random_base, in_graph_params)
    
    
    plot_history_list([vanilla_grid_search,
                       same_network_grid_search,
#                       anti_same_network_grid_search,
#                       random_grid_search,
                       self_pace_grid_search],
#                       anti_self_pace_grid_search],
                       ["Vanilla",
                        "Self-Taught",
#                        "anti self-taught",
#                        "random",
                        "Self-Paced"],
#                        "anti self-paced"],
                        "",
                        bold_first=True,
                        colors=colors,
                        figsize=(7.5,4),
                        xlim=(-490, 3900))


def problem1_summary_all_results_graph(curriculum_base, random_base,
                                       anti_base, vanilla_base,
                                       same_net_base, self_pace_base,
                                       single_shot_base, two_jumps_base):

    num_non_vanilla = 7
    
    in_graph_params = {
            "lrate": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045,
                      0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01],
            "decay": [2, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1],
            "lrjump": [200 ,300, 400, 500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [100],
            "strt": [0.04],
            }
    

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    print()
    print("curriculum")
    curriculum_grid_search = choose_lr_parameters(curriculum_base, in_graph_params)
    print()
    print("anti")
    anti_grid_search = choose_lr_parameters(anti_base, in_graph_params)
    print()
    print("random")
    random_grid_search = choose_lr_parameters(random_base, in_graph_params)
    print()
    print("vanilla")
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)
    
    in_graph_params = {
            "lrate": [0.045, 0.04, 0.035, 0.03, 0.025, 0.02],
            "decay": [1.7, 1.6],
            "lrjump": [500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [50, 70, 100, 200, 300],
            "strt": [0.04, 0.06, 0.08, 0.1, 0.12],
            }
    
    print()
    print("same net")
    same_network_grid_search = choose_lr_parameters(same_net_base, in_graph_params)

    in_graph_params = {
            "lrate": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045,
                      0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01],
            "decay": [2, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1],
            "lrjump": [200, 300, 400, 500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [100],
            "strt": [0.04],
            }

    print()
    print("self pace")    
    self_pace_grid_search = choose_lr_parameters(self_pace_base, in_graph_params)


    in_graph_params = {
            "lrate": [0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035,
                      0.03, 0.025, 0.02],
            "decay": [1.7, 1.6],
            "lrjump": [500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [50, 70, 100, 200, 300],
            "strt": [0.04, 0.06, 0.08, 0.1, 0.12],
            }
    
    print()
    print("single step")  
    single_step_grid_search = choose_lr_parameters(single_shot_base, in_graph_params)

    in_graph_params = {
            "lrate": [0.035],
            "strt": [0.04, 0.06, 0.08, 0.1, 0.12],
            "chng1": [20, 30, 40, 50, 60, 70, 80, 90],
            "chng2": [20, 30, 40, 50, 60, 70, 80, 90]
            }

    print()
    print("2 jumps")     
    two_jumps_grid_search = choose_lr_parameters(two_jumps_base, in_graph_params)

   
    plot_history_list([vanilla_grid_search,
                       curriculum_grid_search,
                       anti_grid_search,
                       random_grid_search,
                       same_network_grid_search,
                       self_pace_grid_search,
                       single_step_grid_search,
                       two_jumps_grid_search],
                       ["Vanilla",
                        "Curriculum",
                        "Anti Curriculum",
                        "Random",
                        "Self-Taught",
                        "Self-Paced",
                        "Single Step",
                        "Varied"],
                        "",
                        bold_first=True,
                        colors=colors,
                        figsize=(7.5, 5),
                        legend_loc="lower right")
    
    plot_history_list([vanilla_grid_search,
                       curriculum_grid_search,
                       anti_grid_search,
                       random_grid_search,
                       same_network_grid_search,
                       self_pace_grid_search,
                       single_step_grid_search,
                       two_jumps_grid_search],
                       ["Vanilla",
                        "Curriculum",
                        "Anti Curriculum",
                        "Random",
                        "Self-Taught",
                        "Self-Paced",
                        "Single Step",
                        "Varied"],
                        "",
                        bold_first=True,
                        colors=colors,
                        barsize=(5, 5),
                        legend_loc="lower right",
                        bar_x=True)


def problem1_fixed_vs_varied(curriculum_fixed_base, vanilla_base,
                             curriculum_varied_base):
    num_non_vanilla = 2
    
    in_graph_params = {
            "lrate": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045,
                      0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01],
            "decay": [2, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1],
            "lrjump": [200 ,300, 400, 500, 600],
            "minimal": ["1e-4"],
            "inc": [1.9],
            "jmp": [100],
            "strt": [0.04],
            }
    

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    curriculum_fixed_grid_search = choose_lr_parameters(curriculum_fixed_base, in_graph_params)
    vanilla_grid_search = choose_lr_parameters(vanilla_base, in_graph_params)
    
    in_graph_params = {
                        "lrate": [0.035],
                        "strt": [0.04, 0.06, 0.08, 0.1, 0.12],
                        "chng1": [20, 30, 40, 50, 60, 70, 80, 90],
                        "chng2": [20, 30, 40, 50, 60, 70, 80, 90]
                        }
    
    curriculum_varied_grid_search = choose_lr_parameters(curriculum_varied_base, in_graph_params) 
    
    
    plot_history_list([vanilla_grid_search,
                       curriculum_fixed_grid_search,
                       curriculum_varied_grid_search],
                       ["Vanilla",
                        "Fixed",
                        "Varied"],
                        "",
                        bold_first=True,
                        colors=colors,
                        figsize=(7.5,4),
                        xlim=(-400, 3900))


def gradient_distance_graph(vanilla_path):

    num_non_vanilla = 3
    num_repeats = 15
    num_last_layers = 1
    colors = [COLORS[i] for i in range(num_non_vanilla)]
    
    
    dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16)
    
    inception_order = get_classic_order("inception", dataset)[::]
    vgg_order = get_classic_order("vgg16", dataset)[::]
    resnet_order = get_classic_order("resnet", dataset)[::]
#    (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
#    train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
#                                                                 transfer_values_test, dataset.y_test, dataset)
#    order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train)
#    inception_order = order[::]
    
    
    all_dists, all_variances = grad_directions_results(vanilla_path, inception_order, dataset,
                                                       num_repeats=num_repeats, order1=vgg_order,
                                                       order2=resnet_order)

    for layer_num in range(num_last_layers):
        conditions = []
        names = []
        for name, dists in all_dists.items():
            conditions.append([d[layer_num] for d in dists])
            names.append(name)
                   
        mean_dists = [np.mean(cond) for cond in conditions]
        error_dists = [stats.sem(cond) for cond in conditions]
        plt.figure()
        bar_locations = np.arange(len(mean_dists))
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
#        fig, ax = plt.subplots(1, 1)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(mean_dists))]
        
        ax.bar(bar_locations, mean_dists, yerr=error_dists, color=colors, capsize=3)
    #    ax.errorbar(bar_locations, mean_dists, xerr=error_dists, linestyle='None', marker='^')
    #    ax.errorbar(bar_locations, mean_dists, error_dists, color=colors, linestyle='None', marker='^')
        y_pos = np.arange(len(mean_dists))
        plt.xticks(y_pos, names, rotation='vertical')
#        plt.xticks(y_pos, names)
        low = min(mean_dists)
        high = max(mean_dists)
        plt.ylabel("dist")
        plt.ylim([low-0.01, high+0.01])
        plt.tight_layout()
    
        std_conditions = []
        names = []
        for name, stds in all_variances.items():
            print(name)
            print(dists)
            std_conditions.append([d[layer_num] for d in stds])
            names.append(name)
        
        mean_stds = [np.mean(cond) for cond in std_conditions]
        error_stds = [stats.sem(cond) for cond in std_conditions]
        
        plt.figure()
        bar_locations = np.arange(len(mean_stds))
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(mean_stds))]
        
        ax.bar(bar_locations, mean_stds, yerr=error_stds, color=colors, capsize=3)
        y_pos = np.arange(len(mean_stds))
        plt.xticks(y_pos, names, rotation='vertical')
#        plt.xticks(y_pos, names)
        low = min(mean_stds)
        high = max(mean_stds)
        plt.ylabel("std")
        plt.ylim([low-0.01, high+0.01])
        plt.tight_layout()
    


def evaluate_model(model, x, y):
    predicted_y = np.argmax(model.predict(x), axis=1)
    print(predicted_y.shape)
    print(y.shape)
    return np.mean(predicted_y == y)

def evaluate_test_set_cross_val(model_path, num_models, validation_dataset):

    x_test, y_test, _ = validation_dataset.get_original_test_data()
#    return evaluate_model(model, validation_dataset.x_test, validation_dataset.y_test)
    models = [model_path + "_model_num"+str(repeat) for repeat in range(num_models)]
    scores = []
    for model_path in models:
        model = keras.models.load_model(model_path)
        scores.append(evaluate_model(model, x_test, y_test))
    
    return np.mean(scores), stats.sem(scores)


def cross_validate_models(model_paths, models_params, names,
                          validation_dataset):
    grids = []
    validations = []
    models = []
    for path, params, name in zip(model_paths, models_params, names):
        print(name)
        grid_search = choose_lr_parameters(path, params)
        print(grid_search)
        grids.append(grid_search)
        val_history = grid_search.replace("validation","validation_model").replace("_inception", "")
        validations.append(val_history)
        models.append(val_history[:-len("_history")])
        print("@@@@")
    
    num_non_vanilla = len(validations)

    colors = [COLORS[i] for i in range(num_non_vanilla)]
    
    plot_history_list(grids,
                      names,
                      "",
                      bold_first=False,
                      colors=colors)   
    
    plot_history_list(validations,
                      names,
                      "",
                      bold_first=False,
                      colors=colors)   
    val_dataset = cifar100_subset_validation.Cifar100_Subset_Validation(supeclass_idx=16)
    num_models = 7
    evaluations = [evaluate_test_set_cross_val(model_path, num_models, val_dataset) for model_path in models]
    
    scale_down = 1
    bar_locations = np.arange(len(evaluations)) / scale_down
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
#    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(evaluations))]
    
    ax.bar(bar_locations, [e[0] for e in evaluations], color=colors, yerr=[e[1] for e in evaluations],
           capsize=3)
    
    y_pos = np.arange(len(names)) / scale_down
    plt.xticks(y_pos, names, rotation='vertical')
#    plt.xticks(y_pos, names)
    low = min([e[0] for e in evaluations])
    high = max([e[0] for e in evaluations])
    plt.ylim([low-0.01, high+0.01])
    plt.tight_layout()
#    plt.plot([evaluate_test_set_cross_val(model_path, num_models, val_dataset) for model_path in models], ".")
    



if __name__ == "__main__":
#    font = {'family' : 'normal',
#            'weight' : 'bold',
#            'size'   : 30}
#        
#    matplotlib.rc('font', **font)
#    plt.figure()
#    matplotlib.rcParams.update({'font.size': 80})
    

    
#    print("problem1")
#    problem1_curriculum_vanilla_anti_random_graph(example_output_path_curriculum,
#                                                  example_output_path_random,
#                                                  example_output_path_anti,
#                                                  example_output_path_vanilla)
    
    
    print("problem1 - github")
    problem1_github_graph(example_github_curriculum,
                          example_github_vanilla)
    
    
#    print("problem2 - cifar10")
#    problem23_curriculum_vanilla_graph(exmaple_output_cifar_10_curriculum,
#                                       exmaple_output_cifar_10_vanilla,
#                                       "Cifar-10")
#  
#    print("problem2 - cifar100")
#    problem23_curriculum_vanilla_graph(exmaple_output_cifar_100_curriculum,
#                                       exmaple_output_cifar_100_vanilla,
#                                       "Cifar-100")
##    
#    print("problem4 - cifar100")
#    problem45_vgg_curriculum_vanilla_graph(example_output_vgg_cifar100_curriculum,
#                                           example_output_vgg_cifar100_vanilla,
#                                           example_output_vgg_cifar100_2_jumps,
#                                           "Cifar-100")
##    
##    print("problem4 - cifar10")
##    problem45_vgg_curriculum_vanilla_graph(example_output_vgg_cifar10_curriculum,
##                                           example_output_vgg_cifar10_vanilla,
##                                           example_output_vgg_cifar10_2_jumps,
##                                           "Cifar-10")
##
##
##    
##    print("problem5 - cifar10")
##    problem45_vgg_curriculum_vanilla_graph(example_output_vgg_cifar10_curriculum,
##                                           example_output_vgg_cifar10_vanilla,
##                                           example_output_vgg_cifar10_2_jumps,
##                                           "Cifar-10")
##    
##    print("stl10")
##    problem6_stl10_curriculum_vanilla_graph(example_stl10_curriculum, example_stl10_vanilla)
##    
#    print("problem1 - singler step")
#    problem1_single_step_curriculum_vanilla_graph(example_output_path_single_step_curriculum, example_output_path_vanilla,
#                                                  example_output_path_single_step_anti,
#                                                  example_output_path_single_step_random)
#    
#    print("problem1 - same network")
#    problem1_same_network_self_pace_curriculum_vanilla_graph(example_output_path_same_network_curriculum,
#                                                             example_output_path_vanilla,
#                                                             example_output_path_same_network_anti, ## CHANGE TO ANTI_SAME_NET
#                                                             example_output_path_random,
#                                                             example_output_path_self_pace,
#                                                             example_output_path_anti_self_pace
#                                                             )
# 
#    
#    
#    print("problem1 - summary")
#    problem1_summary_all_results_graph(example_output_path_curriculum,
#                                       example_output_path_random,
#                                       example_output_path_anti,
#                                       example_output_path_vanilla,
#                                       example_output_path_same_network_curriculum,
#                                       example_output_path_self_pace,
#                                       example_output_path_single_step_curriculum,
#                                       example_output_two_jumps)
#    
#    
#    
#    
#    print("problem1 - fixed vs varied")
#    problem1_fixed_vs_varied(example_output_path_curriculum,
#                             example_output_path_vanilla,
#                             example_output_two_jumps)
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    print("gradients")
#    gradient_distance_graph(vanilla_models_path)
#
#    params_same_net = {"lrate": [0.1, 0.09,0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01],
#                       "strt": [0.05, 0.075, 0.01, 0.125, 0.15],
#                       "decay": [2, 1.8, 1.7, 1.5, 1.3, 1.1],
#                       "lrjump": [200, 300, 400, 500, 600],
#                       "jmp": [50, 70, 100, 200, 300]}
#    
#    
#    params_curriculum_vanilla = {"lrate": [0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02],
#                                 "decay": [2, 1.7, 1.5, 1.3, 1.2, 1.1, 1.05],
#                                 "lrjump": [200, 400, 600, 800, 1000, 1200],
#                                 "strt": [0.04, 0.05, 0.075, 0.1, 0.125],
#                                 "jmp": [50, 100, 200, 400]}
#    
#
#    params_single_step = {"lrate": [0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02],
#                          "strt": [0.05, 0.075, 0.01, 0.125, 0.15],
#                          "decay": [1.7, 1.6],
#                          "lrjump": [500, 600],
#                          "jmp": [50, 70, 100, 200, 300]}
#    
#    
#    print("cross validation")
#    example_validation_curriculum = base_4 + r"validation_longer_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_1.9_${strt}_decay${decay}_min1e-4_lrjmp${lrjump}_history"
#    example_validation_same = base_4 + r"validation_16_same_network_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_1.9_${strt}_decay${decay}_min1e-4_lrjmp${lrjump}_history"
#    example_validation_vanilla = base_4 + r"validation_longer_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_${strt}_decay${decay}_min1e-4_lrjmp${lrjump}_history"
#    example_validation_anti_single_step = base_4 + r"validation_long_norm_inception_16_anti_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min1e-4_lrjmp${lrjump}_history"
#    example_validation_curriculum_single_step = base_4 + r"validation_long_norm_inception_16_curriculum_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min1e-4_lrjmp${lrjump}_history"
#    example_validation_random_single_step = base_4 + r"validation_long_norm_inception_16_random_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min1e-4_lrjmp${lrjump}_history"
#    cross_validate_models([example_validation_curriculum,
##                           example_validation_same,
#                           example_validation_vanilla,
##                           example_validation_anti_single_step,
##                           example_validation_curriculum_single_step,
##                           example_validation_random_single_step
#                           ],
#                           [params_curriculum_vanilla,
##                            params_same_net,
#                            params_curriculum_vanilla,
##                            params_single_step,
##                            params_single_step,
##                            params_single_step
#                           ],
#                           ["curriculum",
##                            "self-taught",
#                            "vanilla",
##                            "anti single step",
##                            "single step",
##                            "random single step"
#                           ],
#                            None)

    print("done")    
