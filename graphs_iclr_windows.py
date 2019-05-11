# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:23:54 2018

@author: stenl
"""

import numpy as np
import draw_results
import pickle
import itertools
import matplotlib.pyplot as plt
import time
import os
import itertools
import scipy
import re
import itertools
from scipy import stats
import matplotlib
import seaborn as sns
import math
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
sns.set()
sns.set_style("whitegrid")


num_epochs = 100
data_size = 2500
batch_size = 100

itartions = (num_epochs * data_size) // batch_size

def fixed_exponential_creator(start_percent, inc, jump):
    def fixed_exponential(i):
        return min(1, start_percent*inc**(i//jump))
    return fixed_exponential


def single_step_creator(start_percent, jump):
    def single_step(i):
        if i > jump:
            return 1
        else:
            return start_percent
    return single_step

def varied_exponential_creator(start_percent, inc, jump_rest, jump_lengths):
    
    num_jumps = math.ceil(math.log(1/start_percent, inc))
    jumps = list(np.cumsum(jump_lengths))
    while len(jumps) < num_jumps:
        jumps.append(jumps[-1] + jump_rest)
    
    def varied_exponential(i):
        res = start_percent
        for jump in jumps:
            if jump < i:
                res *= inc
        return min(1, res)
    return varied_exponential

def compare_scheduling_functions(): 
    fixed_expo = fixed_exponential_creator(0.03125, 2, 400)
    single_step = single_step_creator(0.28, 1400)
    varied_expo = varied_exponential_creator(0.0878, 1.5, 250, [100, 400, 30, 80, 300])
    
    
    SMALL_SIZE = 17
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 27
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    fig, axs = plt.subplots(1, 1, figsize=(10,4))
    
    # graph_from_history(history, y_tag="loss", ylabel='Loss', axs=axs[1])
    
    
    iterations = list(range(itartions))
    plt.plot(iterations, [fixed_expo(i) for i in range(len(iterations))], color="red", label="Fixed exponential pacing")
    plt.plot(iterations, [varied_expo(i) for i in range(len(iterations))], "--", color="green", label="Varied exponential pacing")
    plt.plot(iterations, [single_step(i) for i in range(len(iterations))], ":", color="blue", label="Single step")
    
    
    #if histories_2:
    #    for idx, (history_2, name2) in enumerate(zip(histories_2, names2)):
    #        if colors is not None:
    #            color = colors[idx]
    #        else:
    #            color = None
    #        graph_from_history(history_2, x_tag='batch_num', xlabel='batch number',
    #                           axs=axs, train=train, test=not train, label_test=name2,
    #                           label_train=name2, error=error,
    #                           color=color, y_tag=y_tag, ylabel=ylabel)
    
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    #    plt.xlim([0, 5000])
    try:
        plt.tight_layout()
    except:
        pass
    
    
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of Data")



def bar_groups_gradients(): 
    dists_transfer_all_and_rand = [0.08959209556964091, 0.09034959118859846, 0.09093845517460007,  0.10481137593720269, 0.1052892566173875, 0.1042055567941053,]
    dists_transfer_all_and_rand_sem = [0.0012470818377546171, 0.0012631074884274136, 0.0012456773958922104, 0.0015503121225203648, 0.0015485074732366673, 0.0015841784779688799,]
    dists_transfer_all_and_rand_names = ['All-Res', 'All-VGG', 'All-Incep', 'Rand-Res', 'Rand-VGG', 'Rand-Incep',]
    dists_transfer_trasfer = [0.004680990203164276,  0.002968103703689933, 0.0041094308985396575,]
    dists_transfer_trasfer_sem = [0.00017484235766804078, 0.0001385773393494964, 0.00017896290062322442,]
    dists_transfer_trasfer_names = ['Incep-Res', 'VGG-Res', 'Incep-VGG',]
    dists_all_rand = [0.046754558683649305,]
    dists_all_rand_sem = [0.0008058149427894711,]
    dists_all_rand_names = ['All-Rand',]
    
    var_transfer = [0.18439833731183267, 0.18104222150412735, 0.18115361282807652]
    var_transfer_names = ["VGG-16", "Resnet", "Inception"]
    var_transfer_e = [0.002413476360919972, 0.002417395447837624, 0.0025433819474482947]
    var_all_rand = [0.506175023262067, 0.521044167037957]
    var_all_rand_names = ["Random", "All"]
    var_all_rand_e = [0.002994969091636482, 0.0024518587858617805]
    
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
    
    fig, ax = plt.subplots(1, 1, figsize=(7.5,3.5))
    ax.xaxis.grid()
    width = 0.9     
    groupgap=0.9
    y1=dists_transfer_all_and_rand
    y2=dists_transfer_trasfer
    y3=dists_all_rand
    x1 = np.arange(len(y1))
    x2 = np.arange(len(y2))+groupgap+len(y1)
    x3 = np.arange(len(y3)) + groupgap + len(y2) + groupgap + len(y1)
    e1 = dists_transfer_all_and_rand_sem
    e2 = dists_transfer_trasfer_sem
    e3 = dists_all_rand_sem
    ind = np.concatenate((x1,x2,x3))
    rects1 = ax.bar(x1, y1, width, color='r', edgecolor= "black",
                    label="Transfer - All", yerr=e1, capsize=3)
    rects2 = ax.bar(x2, y2, width, color='b', edgecolor= "black",
                    label="Transfer - Transfer", yerr=e2, capsize=3)
    rects3 = ax.bar(x3, y3, width, color='g', edgecolor= "black",
                    label="All - Random", yerr=e3, capsize=3)

    ax.set_ylabel('Distance')
    ax.set_xticks(ind)
    ax.set_xticklabels(dists_transfer_all_and_rand_names + dists_transfer_trasfer_names + dists_all_rand_names, rotation='vertical')


    plt.tight_layout()

    plt.legend(loc="best")


    fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
    ax.xaxis.grid()
    width = 0.9
    groupgap = 0.9
    y1=var_transfer
    y2=var_all_rand
    x1 = np.arange(len(y1))
    x2 = np.arange(len(y2))+groupgap+len(y1)
    e1 = var_transfer_e
    e2 = var_all_rand_e
    ind = np.concatenate((x1,x2))
    rects1 = ax.bar(x1, y1, width, color='r', edgecolor= "black",
                    label="Transfer Scoring", yerr=e1, capsize=3)
    rects2 = ax.bar(x2, y2, width, color='b', edgecolor= "black",
                    label="All - Random", yerr=e2, capsize=3)

    ax.set_ylabel('Total variance')
    ax.set_xticks(ind)
    ax.set_xticklabels(var_transfer_names + var_all_rand_names, rotation='vertical')


    plt.tight_layout()

    plt.legend(loc="best")

if __name__ == "__main__":
#    bar_groups_gradients()
    compare_scheduling_functions()