import numpy as np
import cifar100_subset
import draw_results
import pickle
import itertools
import matplotlib.pyplot as plt
import transfer_learning
import time
import cifar100_model
import train_keras_model
import os
import pandas as pd
import keras.backend as K
import itertools
import tensorflow as tf
import scipy
import keras
import re

#
# with tf.Session() as S:
#     for d in S.list_devices():
#         print (d.name, d.device_type, d.memory_limit_bytes)


def measure_freq(images):
    num_images, h, w, c = images.shape
    images_last = images.transpose(0,3,1,2)
    freq_map = np.abs(np.fft.fft2(images_last))
    scores = np.zeros(num_images)
    for img_idx in range(num_images):
        image_score = 0
        for c_idx in range(c):
            for freq_x in range(h):
                for freq_y in range(w):
                    if freq_x == 0 and freq_y == 0:
                        continue
                    image_score += freq_map[img_idx, c_idx, freq_x, freq_y] / (freq_x+freq_y)
        scores[img_idx] = image_score

    res = np.asarray(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
    return res


def dist_from_prototype(dataset):
    num_images, h, w, c = dataset.x_train.shape
    prototypes = np.zeros((dataset.n_classes, h, w, c))
    for class_idx in range(dataset.n_classes):
        class_indexes = [i for i in range(num_images) if dataset.y_train[i] == class_idx]
        prototypes[class_idx, :, :, :] = np.mean(dataset.x_train[class_indexes, :, :, :], axis=0)

    scores = np.zeros(num_images)

    for img_idx in range(num_images):
        cur_img = dataset.x_train[img_idx, :, :, :]
        cur_proto = prototypes[dataset.y_train[img_idx], :, :, :]
        score = np.sum(np.abs(cur_img - cur_proto))
        scores[img_idx] = score

    res = np.asarray(sorted(range(len(scores)), key=lambda k: scores[k]))
    return res, scores


def stop_criteria_line(acc):
    window_size = 5
    diff = 0.03
    diff_final = 0.005
    final_acc = np.mean(acc[-window_size:])
    res = [0] * window_size
    for i in range(window_size, len(acc)):
        cur_acc =  np.mean(acc[i-window_size:i])
        if np.abs(cur_acc - final_acc) < diff_final:
            untill_end_acc = np.mean(acc[i:])
            if np.abs(cur_acc - untill_end_acc) < diff:
                res.append(0.5)
            else:
                res.append(0)
        else:
            res.append(0)
    return res


def stop_criteria_point(acc):
    line = stop_criteria_line(acc)
    for i in range(len(line)):
        if line[i]:
            return i
    return len(line) - 1


# dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16)

# order, scores = measure_freq(dataset.x_test)

# order, scores = dist_from_prototype(dataset)



# print(scores)
# print(order)
# print(scores[1913])

# draw_results.plot_cifar_100(dataset.x_train[order[:10], :, :, :])
# draw_results.plot_cifar_100(dataset.x_train[order[-10:], :, :, :])

# res = []
# orders = ['gad', 'model', "freq", "prototype", "vgg16", "vgg19", "inception", "xception", "resnet"]
# for combination in itertools.combinations(orders, 3):
#     order_name = ""
#     for name in combination:
#         order_name += name + "_"
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/ensemble_curriculum_adam_" + order_name + "history", "rb+") as history_file:
#         history_ensemble_diff = pickle.load(history_file)
#
#     val_acc = history_ensemble_diff["val_acc"]
#     res.append(np.mean(val_acc[:20]))
#
# print(res)
# print(max(res))
#
# best_idx = np.argmax(res)
# print(list(itertools.combinations(orders, 3))[best_idx])


# with open(r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_history_sgd_curriculum_net_inception_subset0_models5", "rb+") as history_file:
#     history = pickle.load(history_file)


# draw_results.plot_keras_history_2(history, train=True, name1='test')
# axes = plt.gca()
# min_y = 0.35
# max_y = 1
# axes.set_yticks(np.arange(min_y, max_y, 0.025))
# axes.set_ylim([min_y, max_y])
# plt.grid()

# marker_train = stop_criteria_line(history["acc"])
# marker_test = stop_criteria_line(history["val_acc"])
# plt.plot(history["batch_num"], marker_train)
# plt.plot(history["batch_num"], marker_test)

# plt.show()

def get_ensemble_results_from_files(save_each, num_models, model_output_path):
    history = {"acc": [], "val_acc": [], "batch_num": []}
    for iter in range(0, num_batchs, save_each):
        start_time = time.time()
        print("calculating ensemble for batch: " + str(iter) + r"/" + str(num_batchs))
        res_test = np.zeros((dataset.test_size, dataset.n_classes))
        res_train = np.zeros((dataset.train_size, dataset.n_classes))
        for network in range(0, num_models):
            cur_model_path = model_output_path + "_net" + str(network) + "_iter" + str(iter)
            with open(cur_model_path + "_res_test", 'rb') as file_pi:
                results = pickle.load(file_pi)
                res_test += results
            with open(cur_model_path + "_res_train", 'rb') as file_pi:
                res_train += pickle.load(file_pi)
        predicted_train = np.argmax(res_train, axis=1)
        predicted_test = np.argmax(res_test, axis=1)

        history["acc"].append(np.mean(predicted_train == dataset.y_train))
        history["val_acc"].append(np.mean(predicted_test == dataset.y_test))
        history["batch_num"].append(iter)
        print("--- %s seconds ---" % (time.time() - start_time))
    return history


# networks = ["resnet", "inception", "xception", "vgg16", "vgg19", "gad", "prototype", "freq"]
# subsets = [0, 7, 14, 16]
# # networks = ["inception", "freq", "small_network", "same_network"]
# # networks = ["small_network"]
# # subsets = [0, 4, 7, 14, 16]
# num_models_options = [1, 5, 10]

# for num_models in num_models_options:
#     num_models_suffix = ""
#     if num_models != 10:
#         num_models_suffix += "_models" + str(num_models)
#     for net in networks:
#         for subset in subsets:
#             with open(r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_history_vanilla_net_" + net + "_subset" + str(subset) + num_models_suffix, "rb+") as history_file:
#                 history_vanilla = pickle.load(history_file)
#             with open(r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_history_curriculum_net_" + net + "_subset" + str(subset) + num_models_suffix, "rb+") as history_file:
#                 history_curriculum = pickle.load(history_file)
#             with open(r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_history_anti_net_" + net + "_subset" + str(subset) + num_models_suffix, "rb+") as history_file:
#                 history_anti = pickle.load(history_file)
#             with open(r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_history_random_net_" + net + "_subset" + str(subset) + num_models_suffix, "rb+") as history_file:
#                 history_random = pickle.load(history_file)

#             draw_results.plot_keras_history_2(history_vanilla,  [history_curriculum, history_anti, history_random],
#                                               train=False, name1='vanilla', names2=['curriculum', 'anti-curriculum', 'random'], error=True)
#             axes = plt.gca()
#             min_y = 0.25
#             max_y = 0.8
#             axes.set_yticks(np.arange(min_y, max_y, 0.025))
#             axes.set_ylim([min_y, max_y])
            
#             vanilla_converged = stop_criteria_point(history_vanilla["acc"])
#             curriculum_converged = stop_criteria_point(history_curriculum["acc"])
#             anti_converged = stop_criteria_point(history_anti["acc"])
#             random_converged = stop_criteria_point(history_random["acc"])
#             plt.axvline(x=history_vanilla["batch_num"][vanilla_converged], color="blue")
#             plt.axvline(x=history_curriculum["batch_num"][curriculum_converged], color="orange")
#             plt.axvline(x=history_anti["batch_num"][anti_converged], color="green")
#             plt.axvline(x=history_random["batch_num"][random_converged], color="red")
#             plt.grid()

#             if net in ["vgg16", "vgg19", "inception", "xception", "resnet"]:
#                 dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=subset)
#                 if net == "inception":
#                     (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)

#                 else:
#                     (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_classic_networks(dataset,
#                                                                                                                            net)

#                 train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
#                                                                              transfer_values_test, dataset.y_test, dataset,
#                                                                              network_name=net)
#                 svm_score = np.mean(np.argmax(test_scores, axis=1) == dataset.y_test)
#                 plt.title(net + ", subset" + str(subset) + ", repeats: " + str(history_vanilla["num_repeats"]) + ", svm_score: " + str(svm_score) + "num models: " + str(num_models))
#             else:
#                 plt.title(net + ", subset" + str(subset) + ", repeats: " + str(history_vanilla["num_repeats"]) + ", num models: " + str(num_models))


#             plt.savefig(r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_history_net_" + net + "_subset" + str(subset)  + num_models_suffix + ".png")
#             # plt.show()







    # num_images, h, w, c = dataset.x_train.shape
    # prototypes = np.zeros((dataset.n_classes, h, w, c))
    # for class_idx in range(dataset.n_classes):
    #     class_indexes = [i for i in range(num_images) if dataset.y_train[i] == class_idx]
    #     prototypes[class_idx, :, :, :] = np.mean(dataset.x_train[class_indexes, :, :, :], axis=0)

    # scores = np.zeros(num_images)

    # for img_idx in range(num_images):
    #     cur_img = dataset.x_train[img_idx, :, :, :]
    #     cur_proto = prototypes[dataset.y_train[img_idx], :, :, :]
    #     score = np.sum(np.abs(cur_img - cur_proto))
    #     scores[img_idx] = score

    # res = np.asarray(sorted(range(len(scores)), key=lambda k: scores[k]))

    # if reverse:
    #     res = np.flip(res, 0)
    # if random:
    #     np.random.shuffle(res)

    # return res



#def summerize_history(history_path):
#    with open(history_path, "rb+") as history_file:
#        history = pickle.load(history_file)
#    stop_criteria = stop_criteria_point(history["acc"])
#    final_acc = history["val_acc"][-1]
#    final_acc_std = history["std_val_acc"][-1]
#    repeats = history["num_repeats"]
#    return stop_criteria, final_acc, final_acc_std, repeats
#
#conditions = ["vanilla", "curriculum", "anti", "random"]
#networks = ["resnet", "inception", "xception", "vgg16", "vgg19", "gad", "prototype", "freq"]
#subsets = [0, 7, 14, 16]
#
#
## networks = ["inception"]
## subsets = [16]
#
#
## networks = ["inception", "freq", "small_network", "same_network"]
## networks = ["inception", "freq", "same_network"]
## subsets = [0, 4, 7, 14, 16]
## num_models_options = [1, 5, 10]
#num_models_options = [1, 5, 10]
#
#
## network, subset, condition, num_models, (stop criteria, acc, acc std)
#results = np.zeros((len(networks), len(subsets), len(conditions), len(num_models_options), 3))
#results = {"network": [], "subset": [], "condition": [], "num_models": [], "acc": [],  "stop_criteria": [], "std_acc": [], "repeats": []}
#for num_models in num_models_options:
#    num_models_suffix = ""
#    if num_models != 10:
#        num_models_suffix += "_models" + str(num_models)
#    for net in networks:
#        for subset in subsets:
#            for condition in conditions:
#
#                history_path = r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_history_" + condition + "_net_" + net + "_subset" + str(subset) + num_models_suffix
#                stop_criteria, final_acc, final_acc_std, repeats = summerize_history(history_path)
#                results["network"].append(net)
#                results["num_models"].append(num_models)
#                results["condition"].append(condition)
#                results["subset"].append(subset)
#                results["stop_criteria"].append(stop_criteria)
#                results["acc"].append(final_acc)
#                results["std_acc"].append(final_acc_std)
#                results["repeats"].append(repeats)
#
    
#df = pd.DataFrame(results)
#
#
#colors = ['green', 'orange', 'red', 'blue']
#positions = [0, 1, 2, 3]
#plot_stop_criteria = True
#for net in networks:
#    for subset in subsets:
#        fig, ax = plt.subplots()
#        repeats = 0
#        for group, color, pos in zip(df.loc[(df["subset"]==subset) & (df["network"]==net)].groupby('condition'), colors, positions):
#            repeats = list(group[1]["repeats"])[0]
#            key, group = group
#            if not plot_stop_criteria:
#                group.plot("num_models", 'acc', yerr='std_acc', kind='bar', width=0.2, label=key, 
#                           position=pos, color=color, alpha=0.5, ax=ax)
#            else:
#                group.plot("num_models", 'stop_criteria', kind='bar', width=0.2, label=key, 
#                           position=pos, color=color, alpha=0.5, ax=ax)
#
#        plt.legend(loc="best")
#        if plot_stop_criteria:
#            plt.ylabel("stop criteria")
#        else:
#            plt.ylabel("top 1 accuracy")
#        ax.set_xlim(-1, 3.5)
#        # if not plot_stop_criteria:
#        #     ax.set_ylim(0.25, 0.8)
#        # ax.xticks(["subset 1", "subset 5", "subset 10"])
#
#        if net in ["vgg16", "vgg19", "inception", "xception", "resnet"    ]:
#            dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=subset)
#            if net == "inception":
#                (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
#
#            else:
#                (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_classic_networks(dataset,
#                                                                                                                       net)
#
#            train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
#                                                                         transfer_values_test, dataset.y_test, dataset,
#                                                                         network_name=net)
#            svm_score = np.mean(np.argmax(test_scores, axis=1) == dataset.y_test)
#            plt.title(net + ", subset" + str(subset) + ", repeats: " + str(repeats) + ", svm_score: " + str(svm_score))
#        else:
#            plt.title(net + ", subset" + str(subset) + ", repeats: " + str(repeats))
#
#        patches, labels = ax.get_legend_handles_labels()
#        ax.legend(patches, labels, loc='best')
#
#        if not plot_stop_criteria:
#            plt.savefig(r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_bars_adam_net_" + net + "_subset" + str(subset) + ".png")
#        else:
#            plt.savefig(r"/cs/labs/daphna/guy.hacohen/project/models/histories/ensemble_bars_criteria_adam_net_" + net + "_subset" + str(subset) + ".png")
#        # plt.show()



def order_by_freq(dataset):
    """
    returns training order of the given dataset by freqency
    low freq images will be first, high freq images will be last
    """
    images = dataset.x_train
    num_images, h, w, c = images.shape
    
    ## fourier transform for getting the freq map
    images_last = images.transpose(0,3,1,2)
    freq_map = np.abs(np.fft.fft2(images_last))
    scores = np.zeros(num_images)
    for img_idx in range(num_images):
        image_score = 0
        for c_idx in range(c):
            for freq_x in range(h):
                for freq_y in range(w):
                    ## the freq at 0,0 is simply the images mean, which is usally normalized anyway.
                    ## this if makes the score invariant to it.
                    if freq_x == 0 and freq_y == 0:
                        continue
                    image_score += freq_map[img_idx, c_idx, freq_x, freq_y] / (freq_x+freq_y)
        scores[img_idx] = image_score
    ## takes the scores of every image, and produces an ordering.
    ## res[0] is the index of the "easiest" image by the scoring, res[1] is the index of a bit harder image, etc...
    res = np.asarray(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
    return res

def order_by_prototype(dataset, reverse=False, random=False):
    num_images, h, w, c = dataset.x_train.shape
    prototypes = np.zeros((dataset.n_classes, h, w, c))
    for class_idx in range(dataset.n_classes):
        class_indexes = [i for i in range(num_images) if dataset.y_train[i] == class_idx]
        prototypes[class_idx, :, :, :] = np.mean(dataset.x_train[class_indexes, :, :, :], axis=0)

    scores = np.zeros(num_images)

    for img_idx in range(num_images):
        cur_img = dataset.x_train[img_idx, :, :, :]
        cur_proto = prototypes[dataset.y_train[img_idx], :, :, :]
        score = np.sum(np.abs(cur_img - cur_proto))
        scores[img_idx] = score

    res = np.asarray(sorted(range(len(scores)), key=lambda k: scores[k]))

    if reverse:
        res = np.flip(res, 0)
    if random:
        np.random.shuffle(res)

    return res

    
def average_variance(images):
    return np.mean(np.mean(np.mean(np.var(images, axis=0))))

dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16, normalize=False)
(transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)

train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
                                                             transfer_values_test, dataset.y_test, dataset,
                                                             network_name="inception")
order_inception = list(transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train))
order_freq = list(order_by_freq(dataset))
order_proto = list(order_by_prototype(dataset))





def balance_order(order, dataset):
    num_classes = dataset.n_classes
    size_each_class = dataset.x_train.shape[0] // num_classes
    class_orders = []
    for cls in range(num_classes):
        class_orders.append([i for i in range(len(order)) if dataset.y_train[order[i]] == cls])
    print([np.mean(i) for i in class_orders])
    new_order = []
    ## take each group containing the next easiest image for each class,
    ## and putting them according to diffuclt-level in the new order
    for group_idx in range(size_each_class):
        group = sorted([class_orders[cls][group_idx] for cls in range(num_classes)])
        for idx in group:
            new_order.append(order[idx])
    return new_order

#plt.scatter(np.argsort(order_freq), np.argsort(order_proto), marker=".")

#print(balance_order(order_inception, dataset))
#new_order = np.array(balance_order(order_freq, dataset))
#new_new_order = np.array(balance_order(new_order, dataset))
#
##draw_results.plot_cifar_100(dataset.x_train, window_size=7)
#draw_results.plot_cifar_100(dataset.x_train, window_size=4, order=order_inception)
#draw_results.plot_cifar_100(dataset.x_train, window_size=4, order=new_order)
#draw_results.plot_cifar_100(dataset.x_train, window_size=4, order=new_order[::-1])
#
#
#plt.scatter(rank_inception, rank_freq, marker=".")
#plt.show()
#    
    
#import tensorflow as tf
#print("in script")
#with tf.device('/gpu:0'):
#    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#    c = tf.matmul(a, b)
#
#with tf.Session() as sess:
#    print (sess.run(c))

def linear_data_function_generator(dataset, order, batches_to_increase, increase_amount, batch_size=100):

    size_data = dataset.x_train.shape[0]
    
    cur_percent = 0
    cur_data_x = dataset.x_train
    cur_data_y = dataset.y_test_labels
    
    
    def data_function(x, y, batch, history):
        nonlocal cur_percent, cur_data_x, cur_data_y
        
        if batch % batches_to_increase == 0:
            percent = min(cur_percent+increase_amount, 1)
            if percent != cur_percent:
                cur_percent = percent
                data_limit = np.int(np.ceil(size_data * percent))
                new_data = order[:data_limit]
                cur_data_x = dataset.x_train[new_data, :, :, :]
                cur_data_y = dataset.y_train_labels[new_data, :]               
        return cur_data_x, cur_data_y

    return data_function

def exponent_data_function_generator(dataset, order, batches_to_increase, increase_amount, starting_percent, batch_size=100):

    size_data = dataset.x_train.shape[0]
    
    cur_percent = 1
    cur_data_x = dataset.x_train
    cur_data_y = dataset.y_test_labels
    
    
    def data_function(x, y, batch, history):
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

def see_all_data_batch_expo(batches_to_increase, increase_amount, starting_percent):
    cur_percent = starting_percent
    cur_batch = 0
    while cur_percent < 1:
        cur_percent *= increase_amount
        cur_batch += batches_to_increase
    return cur_batch
    

#batches_increase = 3
#increase_amount = 1.1
#starting = 15/2500
#
#dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16)
#order_freq = list(order_by_freq(dataset))
#data_func = exponent_data_function_generator(dataset, order_freq, batches_increase, increase_amount, starting)
#
#for batch in range(170):
#    x,y = data_func(None, None, batch, None)
#    print("batch: " + str(batch))
#    print(x.shape)
#    print(y.shape)
#    print("---")
#    
#
#print(see_all_data_batch_expo(batches_increase, increase_amount, starting))


def bar_plot_histories(histories, names, tag, std=False, base_line_history=None, base_line_name="baseline",
                       to_self_color=True):
    if base_line_history is not None:
        histories += [base_line_history]
        names += [base_line_name]
    final_tag = [history[tag][-1] for history in histories]
    if std:
        stds = [history["std_" + tag][-1] for history in histories]
    bar_locations = np.arange(len(histories))
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(histories))]
    if base_line_history is not None:
        colors[-1] = "grey"
    if not to_self_color:
        colors = None
    if std:
        ax.bar(bar_locations, final_tag, color=colors, yerr=stds)
    else:
        ax.bar(bar_locations, final_tag, color=colors)
    y_pos = np.arange(len(names))
    plt.xticks(y_pos, names, rotation='vertical')
    low = min(final_tag)
    high = max(final_tag)
    plt.ylim([low-0.01, high+0.01])
    plt.grid()
#    high = max(final_tag)
    
#    plt.legend(names, loc="best")
        
def seen_all_data(history, data_size=2500):
    save_each = len(history["data_size"]) / len(history["batch_num"])
#    print(history["batch_num"][int(history["data_size"].index(190)/save_each)])
    return history["batch_num"][int(history["data_size"].index(data_size)/save_each)]

#jumps = [50,100,150,200,250,300,350,400,450,500]
##jumps = [200]
##jumps = [8.4, 8.3, 8.2, 8.1, 8, 7.9, 7.8, 7.7, 7.6, 7.5]
##jumps = [8.3, 8.1, 7.9, 7.7, 7.5]
##jumps = [0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.05, 1.1, 1.2]
##jumps = [300, 500, 700, 900, 1100]
##increases= [1.1, 1.5, 1.9, 2.4, 4]
#increases= [1.9]
#starts = [0.04]
##starts = [0.04, 0.08, 0.1, 0.2, 0.3]
#
##jumps = [50]
##increases= [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4, 3, 4, 6]
##starts = [0.04]
#
##jumps = [200]
##increases= [1.9]
##starts = [0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4]
#
##conditions = ["curriculum", "anti", "random"]
#conditions = ["curriculum"]
#subsets = [16]
#for lr in [0.2, 0.1, 0.05, 0.035, 0.025, 0.015, 0.005]:
#    for subset in subsets:
#        for condition in conditions:
#            for increase in increases:
#                histories = []
#                names = []
#                all_data_batchs = []
#                for start in starts:
#                
#                    for jump in jumps:
#                        path = r"/cs/labs/daphna/guy.hacohen/project/models2/sched_"+str(subset)+"_" + condition + "_sgd_expo_lr"+str(lr)+"_" + str(jump) + "_" + str(increase) + "_" + str(start) + "_history"
#    #                    print(path)
#                        try:
#                            with open(path, "rb+") as history_file:
#                                history = pickle.load(history_file)
#                            history_batch_nums = (np.array(list(range(len(history["val_acc"]))))+1)*50
#                            history["batch_num"] = history_batch_nums
#                            histories.append(history)
#                            names.append(condition + "_" + str(jump) + "_" + str(increase) + "_" + str(start))
#                            all_data_batchs.append(see_all_data_batch_expo(jump, increase, start))
#                        except:
#                            pass
#    
#                with open(r"/cs/labs/daphna/guy.hacohen/project/models2/sched_" + str(subset) + "_vanilla_sgd_expo_lr"+str(lr)+"_450_1.9_0.04_history", "rb+") as history_file:
#        #        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_"+str(subset)+"_vanilla_sgd_expo_tresh_8.3_1.9_0.04_history", "rb+") as history_file:
#        #        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_lin_50_0.05_history", "rb+") as history_file:
#                    vanilla_history = pickle.load(history_file)
#                    history_batch_nums = (np.array(list(range(len(vanilla_history["val_acc"]))))+1)*50
#                    vanilla_history["batch_num"] = history_batch_nums
#                
#                cmap = plt.get_cmap('rainbow')
#                colors = [cmap(i) for i in np.linspace(0, 1, len(histories))]
#        #        colors = [str(np.random.rand()) for history in histories]
#        
#                draw_results.plot_keras_history_2(vanilla_history, histories, 
#                                              train=False, name1="vanilla", names2=names,
#                                              bold=True, colors=colors)
#        #                                      y_tag='loss', ylabel='loss')
#    #            draw_results.plot_keras_history_2(histories[0], histories[1:], 
#    #                                      train=False, name1=names[0], names2=names[1:],
#    #                                      colors=colors[1:])
#    ##                                      y_tag='loss', ylabel='loss')
#                
#    #            for i in range(len(histories)):
#    #    #            plt.axvline(x=all_data_batchs[i], color=colors[i])
#    #    #            print(seen_all_data(histories[i]))
#    #                try:
#    #                    plt.axvline(x=seen_all_data(histories[i]), color=colors[i])
#    #                except:
#    #                    pass
#                plt.xlim([0, histories[0]["batch_num"][-1]+400])
#                title = "lr:" +str(lr)+ " start: " + str(start) + " increase: " + str(increase) + " condition:" + condition + " subset:" + str(subset)
#                plt.title(title)
#        #         draw_results.plot_keras_history_2(history_curriculum2)
#    #            fig_path = r"/cs/labs/daphna/guy.hacohen/project/graphs/sched2/exponent/sched_tresh_increase_" + str(increase) + "_start_" + str(start) + "_" + condition + "_subset_" + str(subset)
#    #            plt.savefig(fig_path + ".png")
#            
#                plt.show()
#                plt.figure()
#        
#                bar_plot_histories(histories, names, "val_acc", std=True, base_line_history=vanilla_history, base_line_name="vanilla")
#                plt.title(title)
#                if subset == 0:
#                    plt.ylim([0.48, 0.54])
#                elif subset == 14:
#                    plt.ylim([0.28, 0.35])
#                    pass
#                elif subset == 16:
#    #                plt.ylim([0.44, 0.50])
#    #                plt.ylim([0.46, 0.52])
#                    plt.ylim([0.47, 0.58])
#                    
#    #            plt.savefig(fig_path + "_bars.png")
#                plt.show()
           

#
#def get_gradients(model):
#    """Return the gradient of every trainable weight in model
#
#    Parameters
#    -----------
#    model : a keras model instance
#
#    First, find all tensors which are trainable in the model. Surprisingly,
#    `model.trainable_weights` will return tensors for which
#    trainable=False has been set on their layer (last time I checked), hence the extra check.
#    Next, get the gradients of the loss with respect to the weights.
#
#    """
#    weights = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
#    optimizer = model.optimizer
#
#    return optimizer.get_gradients(model.total_loss, weights)
#
#
#
#
#model_lib = cifar100_model.Cifar100_Model()
#dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16)
#model = model_lib.build_classifier_model(dataset, "large",
#                                         dropout_1_rate=0.25, dropout_2_rate=0.5,
#                                         reg_factor=50e-4,
#                                         bias_reg_factor=None)
#        
#train_keras_model.compile_model(model, initial_lr=1e-3,
#                                loss='categorical_crossentropy',
#                                optimizer="sgd", metrics=['accuracy'],
#                                momentum=0.0)
#
#
#outputTensor = model.output
##listOfVariableTensors = model.trainable_weights
#gradients = K.gradients(outputTensor, model.input)
##
##np.random.shuffle(order_inception)
##trainingExample = np.expand_dims(dataset.x_train[0,:,:,:],axis=0)
#trainingExample = dataset.x_train[:,:,:,:]
#sess = tf.InteractiveSession()
#sess.run(tf.initialize_all_variables())
#evaluated_gradients = sess.run(gradients, feed_dict={model.input:trainingExample})
#n_imgs, n_rows, n_cols, n_channels = evaluated_gradients[0].shape
#reshaped = np.reshape(evaluated_gradients[0], (n_imgs, n_rows*n_cols*n_channels))
#
#imag_norms = np.linalg.norm(reshaped, axis=1)
#for i in range(n_imgs):
#    if imag_norms[i] != 0:
#        reshaped[i, :] /= imag_norms[i]
#
#print(np.mean(np.var(reshaped, axis=0)))



#print(evaluated_gradients)
#print(average_variance(evaluated_gradients))
#print(sum([np.all(evaluated_gradients[0][i,:,:,:] == 0.0) for i in range(trainingExample.shape[0])]))
#print(evaluated_gradients)


#jumps = [50, 150 ,250 ,350 ,450]
#increases= [1.9]
#starts = [0.04]
#
#conditions = ["random", "curriculum"]
#subsets = [16]
#
#for subset in subsets:
#        for increase in increases:
#            for start in starts:
#                for jump in jumps:
#                    histories = []
#                    names = []
#                    all_data_batchs = []
#                    for condition in conditions:
#                        path = r"/cs/labs/daphna/guy.hacohen/project/models2/sched_"+str(subset)+"_" + condition + "_sgd_expo_" + str(jump) + "_" + str(increase) + "_" + str(start) + "_history"
#    #                    print(path)
#                        try:
#                            with open(path, "rb+") as history_file:
#                                history = pickle.load(history_file)
#                            history_batch_nums = (np.array(list(range(len(history["val_acc"]))))+1)*50
#                            history["batch_num"] = history_batch_nums
#                            histories.append(history)
#                            names.append(condition + "_" + str(jump) + "_" + str(increase) + "_" + str(start))
#                            all_data_batchs.append(see_all_data_batch_expo(jump, increase, start))
#                        except:
#                            pass
#    
#                    with open(r"/cs/labs/daphna/guy.hacohen/project/models2/sched_" + str(subset) + "_vanilla_sgd_expo_decrease_lr_"+str(jump)+"_"+str(increase)+"_"+str(start)+"_history", "rb+") as history_file:
#            #        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_"+str(subset)+"_vanilla_sgd_expo_tresh_8.3_1.9_0.04_history", "rb+") as history_file:
#            #        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_lin_50_0.05_history", "rb+") as history_file:
#                        vanilla_history = pickle.load(history_file)
#                        history_batch_nums = (np.array(list(range(len(vanilla_history["val_acc"]))))+1)*50
#                        vanilla_history["batch_num"] = history_batch_nums
#                    
#                    cmap = plt.get_cmap('rainbow')
#                    colors = [cmap(i) for i in np.linspace(0, 1, len(histories))]
#            #        colors = [str(np.random.rand()) for history in histories]
#            
#                    draw_results.plot_keras_history_2(vanilla_history, histories, 
#                                                  train=False, name1="vanilla", names2=names,
#                                                  bold=True, colors=colors)
#            #                                      y_tag='loss', ylabel='loss')
#        #            draw_results.plot_keras_history_2(histories[0], histories[1:], 
#        #                                      train=False, name1=names[0], names2=names[1:],
#        #                                      colors=colors[1:])
#        ##                                      y_tag='loss', ylabel='loss')
#                    
#        #            for i in range(len(histories)):
#        #    #            plt.axvline(x=all_data_batchs[i], color=colors[i])
#        #    #            print(seen_all_data(histories[i]))
#        #                try:
#        #                    plt.axvline(x=seen_all_data(histories[i]), color=colors[i])
#        #                except:
#        #                    pass
#                    plt.xlim([0, histories[0]["batch_num"][-1]+400])
#                    title = "jump: " + str(jump) + "start: " + str(start) + " increase: " + str(increase) + " condition:" + condition + " subset:" + str(subset)
#                    plt.title(title)
#            #         draw_results.plot_keras_history_2(history_curriculum2)
#        #            fig_path = r"/cs/labs/daphna/guy.hacohen/project/graphs/sched2/exponent/sched_tresh_increase_" + str(increase) + "_start_" + str(start) + "_" + condition + "_subset_" + str(subset)
#        #            plt.savefig(fig_path + ".png")
#                
#                    plt.show()
#                    plt.figure()
#            
#                    bar_plot_histories(histories, names, "val_acc", std=True, base_line_history=vanilla_history, base_line_name="vanilla")
#                    plt.title(title)
#                    if subset == 0:
#                        plt.ylim([0.48, 0.54])
#                    elif subset == 14:
#                        plt.ylim([0.28, 0.35])
#                        pass
#                    elif subset == 16:
#        #                plt.ylim([0.44, 0.50])
#        #                plt.ylim([0.46, 0.52])
#                        plt.ylim([0.47, 0.58])
#                        
#        #            plt.savefig(fig_path + "_bars.png")
#                    plt.show()
    

def params_from_string(string):
    res = []
    record = False
    param = ""
    for idx, c in enumerate(string):
        if c == "}" and record:
            record = False
            res.append(param)
            param = ""
        if record:
            param += c
        if idx >= 1:
            if c == "{" and string[idx - 1] == "$":
                record = True

    return res

def create_graphs(in_graph_params, example_output_path, between_graph_params,
                  exmaple_vanilla_path):
    in_param_names = list(in_graph_params.keys())
    between_param_names = list(between_graph_params.keys())
    vanilla_params = params_from_string(exmaple_vanilla_path)
        
    for in_param_tuple in itertools.product(*[in_graph_params[key] for key in in_param_names]):
        base_output_path = example_output_path
        for param_idx, param_name in enumerate(in_param_names):
            base_output_path = base_output_path.replace(r"${" + param_name + "}",
                                                        str(in_param_tuple[param_idx]))
        histories = []
        names = []
#        all_data_batchs = []
        memoize = {}
        for between_param_tuple in itertools.product(*[between_graph_params[key] for key in between_param_names]):
            output_path = base_output_path
            for param_idx, param_name in enumerate(between_param_names):
                output_path = output_path.replace(r"${" + param_name + "}",
                                                  str(between_param_tuple[param_idx]))
                if output_path not in memoize:
                    memoize[output_path] = 0
                else:
                    continue
            try:
#                print(output_path)
                with open(output_path, "rb+") as history_file:
                    history = pickle.load(history_file)
                history_batch_nums = (np.array(list(range(len(history["val_acc"]))))+1)*50
                history["batch_num"] = history_batch_nums
                histories.append(history)
                name = ""
                for param_idx, param_name in enumerate(between_param_names):
                    name += param_name + "_" + str(between_param_tuple[param_idx]) + "_"
                name = name[:-1]
                
                names.append(name)
    ##                    all_data_batchs.append(see_all_data_batch_expo(jump, increase, start))
            except:
                pass
        vanilla_path = exmaple_vanilla_path
        for param_name in vanilla_params:
            vanilla_path = vanilla_path.replace(r"${" + param_name + "}",
                                                  str(in_param_tuple[in_param_names.index(param_name)]))
        with open(vanilla_path, "rb+") as history_file:
            vanilla_history = pickle.load(history_file)
            history_batch_nums = (np.array(list(range(len(vanilla_history["val_acc"]))))+1)*50
            vanilla_history["batch_num"] = history_batch_nums
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(histories))]
        draw_results.plot_keras_history_2(vanilla_history, histories, 
                                      train=False, name1="vanilla", names2=names,
                                      bold=True, colors=colors)
#                                      y_tag='loss', ylabel='loss')
#            draw_results.plot_keras_history_2(histories[0], histories[1:], 
#                                      train=False, name1=names[0], names2=names[1:],
#                                      colors=colors[1:])
##                                      y_tag='loss', ylabel='loss')
        
#            for i in range(len(histories)):
#    #            plt.axvline(x=all_data_batchs[i], color=colors[i])
#    #            print(seen_all_data(histories[i]))
#                try:
#                    plt.axvline(x=seen_all_data(histories[i]), color=colors[i])
#                except:
#                    pass
        plt.xlim([0, histories[0]["batch_num"][-1]+4000])
        title = ""
        for param_idx, param_name in enumerate(in_param_names):
            title += param_name + "_" + str(in_param_tuple[param_idx]) + "_"
        title = title[:-1]
        plt.title(title)
#         draw_results.plot_keras_history_2(history_curriculum2)
#            fig_path = r"/cs/labs/daphna/guy.hacohen/project/graphs/sched2/exponent/sched_tresh_increase_" + str(increase) + "_start_" + str(start) + "_" + condition + "_subset_" + str(subset)
#            plt.savefig(fig_path + ".png")
    
        plt.show()
        plt.figure()
#        bar_plot_histories(histories, names, "acc", std=True, base_line_history=vanilla_history, base_line_name="vanilla")
        bar_plot_histories(histories, names, "val_acc", std=True, base_line_history=vanilla_history, base_line_name="vanilla")
        plt.title(title)
#        plt.ylim([0.2, 0.4])
        plt.ylim([0.4, 0.6])
#        plt.ylim([0.6, 0.8])
#        plt.ylim([0.8, 1])
#        if subset == 0:
#            plt.ylim([0.48, 0.54])
#        elif subset == 14:
#            plt.ylim([0.28, 0.35])
#            pass
#        elif subset == 16:
##                plt.ylim([0.44, 0.50])
##                plt.ylim([0.46, 0.52])
#            plt.ylim([0.47, 0.58])
            
#            plt.savefig(fig_path + "_bars.png")
        plt.show()

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

def evaluate_model(model, x, y):
    predicted_y = np.argmax(model.predict(x), axis=1)
    print(predicted_y.shape)
    print(y.shape)
    return np.mean(predicted_y == y)

def evaluate_test_set_cross_val(model_path, validation_dataset):
    model = keras.models.load_model(model_path)
    x_test, y_test, _ = validation_dataset.get_original_test_data()
#    return evaluate_model(model, validation_dataset.x_test, validation_dataset.y_test)
    return evaluate_model(model, x_test, y_test)

#jump = [50,150,250,350,450]
#jump = [300, 500, 700, 900, 1100]
#jump = [30, 50, 80, 100, 200, 300, 400]
#jump = [30, 50, 80, 100, 300]
#jump = [20, 40, 60, 80, 100, 120, 150, 250]
#jump = [20, 40, 80, 150]
#jump = [7.5, 7.6, 7.7, 7.8, 7.9, 8.1, 8.2, 8.3, 8.4]
tresh = [7.5, 7.7, 7.9, 8.1, 8.3]

base = r"/cs/labs/daphna/guy.hacohen/project/models2/"
#base_cross = r"/cs/labs/daphna/guy.hacohen/project/models_cross_fixed/"
#base_cross_fixed = r"/cs/labs/daphna/guy.hacohen/project/models_cross_fixed2/"

# lin
#example_output_path  = base + "sched_${cond}_sgd_lin_balance_lr${lrate}_${jmp}_${inc}_history"

# expo
#example_output_path  = base + "sched_${subset}_${cond}_sgd_expo_balance_${jmp}_${inc}_${strt}_history"

#decrease_lr
#example_output_path  = base + "sched_${subset}_${cond}_sgd_expo_decrease_lr0.025_${jmp}_${inc}_${strt}_history"

#tresh
#example_output_path  = base + "sched_${subset}_${cond}_sgd_expo_tresh_balance_lr${lrate}_${tresh}_${inc}_${strt}_history"
base_debug = r"/cs/labs/daphna/guy.hacohen/project/models_cross_debug/"
base_2 = r"/cs/labs/daphna/guy.hacohen/project/models2/"
base_3 = r"/cs/labs/daphna/guy.hacohen/project/models3/"
base_4 = r"/cs/labs/daphna/guy.hacohen/project/models4/"
#example_output_path = base + r"sched_16_${cond}_sgd_expo_lr${lrate}_150_1.9_0.04_history"
#vanilla_path = base + r"sched_${subset}_vanilla_sgd_expo_lr${lrate}_150_1.9_0.04_history"
#vanilla_path = base + r"sched_16_vanilla_sgd_expo_tresh_balance_lr0.025_7.5_1.9_0.04_history"

#vanilla_path = base + r"adjust_lr_16_vanilla_sgd_expo_lr_0.07_350_1.9_0.04_decay1.7_min1e-4_lrjmp400_history"
#vanilla_path = base + r"adjust_lr_16_vanilla_long_sgd_expo_lr_0.07_100_1.9_0.04_decay1.7_min1e-3_lrjmp400_history"
#vanilla_path = base_3 + r"adjust_lr_stl10_16_vanilla_sgd_expo_balance_lr_0.15_100_1.9_0.02_decay1.5_min1e-4_lrjmp600_history"

#vanilla_path = base + r"adjust_lr_0_vanilla_sgd_expo_lr_0.05_100_1.9_0.04_decay1.5_min1e-4_lrjmp600_history"
# adjust lr

example_output_path = base_3 + r"adjust_lr_${subset}_${cond}_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path = base_cross + r"cross_val_${subset}_${cond}_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"


vanilla_path = base_3 + r"adjust_lr_16_vanilla_sgd_expo_balance_lr_0.08_100_1.9_0.04_decay2_min1e-4_lrjmp500_history"
#example_output_path = base_3 + r"adjust_lr_cifar10_${cond}_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_vanilla = base_3 + r"adjust_lr_stl10_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.02_decay${decay}_min${minimal}_lrjmp${lrjump}_history"

#example_output_path_curriculum = base_4 + r"adjust_lr_16_vanilla_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"


#example_output_path_curriculum = base_4 + r"adjust_lr_long_16_curriculum_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# original curriculum
example_output_path_curriculum = base_4 + r"adjust_lr_long_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_curriculum = base_4 + r"adjust_lr_replong_same_network_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_curriculum = base_3 + r"rep_bootstrap_long_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_num_boots3_repeats5_orig_lr20.05_lr_batch_size2600_decay21.7_history_boost1"
#example_output_path_curriculum = base_3 + r"rep_bootstrap_long_16_combined_min_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_num_boots1_repeats5_orig_lr20.05_lr_batch_size2600_decay21.7_history_boost1"

#example_output_path_curriculum = base_4 + r"adjust_lr_long_same_network_16_curriculum_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_curriculum = base_4 + r"adjust_lr_16_self_pace_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"


#example_output_path_curriculum = base_4 + r"change_lr_exponent0.5_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_curriculum = base_3 + r"cycle_lr_cifar10_curriculum_sgd_expo_balance_decay${decay}_lrjmp${lrjump}_minlr1e-3_maxlr0.09_history"
#example_output_path_curriculum = base_3 + r"adjust_lr_cifar10_36epochs_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_curriculum = base_3 + r"adjust_lr_rbootstrap_16_curriculum_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min1e-3_lrjmp${lrjump}_num_boots_repeats5_history_boost1"
#example_output_path_vanilla = base_3 + r"cycle_lr_cifar10_vanilla_sgd_expo_balance_decay${decay}_lrjmp${lrjump}_minlr1e-3_maxlr0.09_history"
#example_output_path_vanilla = base_3 + r"adjust_lr_cifar10_36epochs_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_anti = base_3 + r"adjust_lr_16_anti_self_pace_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"

#cifar10_vgg_curriculum_no_augmentation_lr_0.1_jmp600_inc1.9_strt0.1_history
#cifar100_vgg_curriculum_no_augmentation_lr_0.1_jmp300_inc1.9_strt0.003_history

#example_output_path_curriculum = base_4 + r"cifar100_vgg_long_curriculum_single_step_no_augmentation_lr_${lrate}_jmp${jmp}_strt${strt}_history"
#example_output_path_vanilla = base_3 + r"cifar100_vgg_vanilla_no_augmentation_history"


#example_output_path_curriculum = base_4 + r"adjust_lr_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_vanilla = base_4 + r"adjust_lr_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_vanilla = base_4 + r"adjust_lr_long_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_random = base_3 + r"adjust_lr_16_random_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#single shot random
#example_output_path_random = base_4 + r"adjust_lr_long_16_random_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_anti = base_3 + r"adjust_lr_16_anti_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"


##validation
#example_output_path_curriculum = base_4 + r"validation_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#same network
#example_output_path_curriculum = base_4 + r"validation_16_same_network_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_vanilla = base_4 + r"validation_16_vanilla_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"






#example_output_path_vanilla = base_3 + r"test_hardness_long1_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_vanilla_easy = base_3 + r"test_hardness_long1_easy_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_vanilla_mid = base_3 + r"test_hardness_long1_mid_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_vanilla_hard = base_3 + r"test_hardness_long1_hard_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"

#LRATES = [0.15, 0.1, 0.05, 0.025, 0.01]
#LRATES = [0.1, 0.07, 0.05, 0.035, 0.025]
#LRATES = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02] +[0.038, 0.035, 0.033]
#LRATES = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02] + [0.075, 0.073, 0.071, 0.07, 0.069, 0.067, 0.065, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032]
#LRATES = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02]

LRATES=[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02] + [0.015, 0.01, 0.11, 0.12, 0.13] + [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 2]

#LRATES=[0.07, 0.1, 0.15, 0.2, 0.25]
#LRATES = [0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01]
#LRATES = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
#LRATES = [0.15, 0.1, 0.07, 0.05, 0.025, 0.01]
#LRATES = [0.1, 0.05]
#LRATES = [0.12, 0.1, 0.08, 0.07, 0.05]
#LRATES = [0.05, 0.08, 0.11, 0.14]
#LRATES = [0.08, 0.07, 0.05, 0.025, 0.015]
#LRATES = [0.12, 0.1, 0.08, 0.07, 0.05, 0.025, 0.015]
#LRATES = [0.07, 0.05, 0.025, 0.01.5]
#LRATES = [0.03]
#LRATES = [0.05, 0.025]
#LRATES = [0.1, 0.05, 0.01]
#LRJUMPS = [20, 50, 100, 250, 500]
#LRJUMPS = [300, 500, 600]

LRJUMPS = [200, 300, 400, 500, 600] + [3000, 4000, 5000, 10000]

#LRJUMPS = [200, 400, 600]
#LRJUMPS = [300, 400, 500, 600, 700]
#LRJUMPS = [500, 800, 1200]
#LRJUMPS = [600]
#LRJUMPS = [600, 800, 1000, 1200, 1400]
#DECAYS = [15, 10, 5, 2, 1.5]
#DECAYS = [2, 1.7, 1.5, 1.3, 1.2, 1.1]
#DECAYS = [1.5, 1.3, 1.2, 1.1]
#DECAYS = [2.5, 2, 1.7, 1.5, 1.3, 1.1]
#DECAYS = [1.3, 1.5]
#DECAYS = [5, 2, 1.5, 1.1]
#DECAYS = [2, 1.7, 1.5, 1.3, 1.1]

DECAYS = [2, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1] + [1, 2.5]

#DECAYS = [2, 1.6, 1.1]
#DECAYS = [2]
#DECAYS = [1.6]
#DECAYS = [2, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.03]
#DECAYS = [2, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.03] + [1.35, 1.32, 1.3, 1.27, 1.25, 1.22, 1.2, 1.17, 1.15]
#DECAYS=[1.35, 1.32, 1.3, 1.27, 1.25]
#DECAYS=[1.35, 1.32, 1.3, 1.27, 1.25, 1.22, 1.2, 1.17, 1.15]
#LRATES=[0.075, 0.073, 0.071, 0.07, 0.069, 0.067, 0.065, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032]



#DECAYS = [1.25, 1.2, 1.15]
#DECAYS=(2 1.8 1.7 1.6 1.5 1.4 1.3 1.2 1.1 1.05 1.03)
#MINIMALS = ["1e-3", "1e-4", "1e-5"]
#MINIMALS = ["1e-3", "1e-4", "1e-5"]
MINIMALS = ["1e-3", "1e-4"]
#MINIMALS = ["1e-4"]
#CONDITION = ["vanilla", "curriculum", "anti", "random"]
CONDITION = ["curriculum"]
#STARTING = [0.04]
#STARTING = [0.05]
#STARTING = [0.04, 0.06, 0.08, 0.1, 0.12]
#STARTING = [0.08, 0.1]
SUBSETS = [16]
#JUMPS = [100, 400]
#JUMPS = [100]
#JUMPS = [10, 20, 50, 70, 100, 200, 300, 400, 600]
#JUMPS = [100, 300, 600, 1000, 1500, 2000]
#INCREASES = [1.1, 1.5, 1.9, 2.5]
#INCREASES = [1.9, 2.3]
INCREASES = [1.9]
#
#LRATES = [0.14, 0.12, 0.1, 0.08, 0.06, 0.04]
#JUMPS = [1500, 2500]
#JUMPS = [500, 1500, 2500] + [300, 400, 600, 700]
#STARTING = [0.003, 0.006, 0.01, 0.02, 0.04, 0.08, 0.1]

#LRATES=[0.05]
##LRATES=[0.05]
STARTING=[0.04, 0.06, 0.08, 0.1, 0.12, 0.15]
#STARTING=[0.05]
#LRATES=[0.08, 0.06, 0.04]
JUMPS=[20, 50, 70, 100, 200, 300, 400, 600] 
#JUMPS=[20, 50, 70, 100, 200, 300, 400, 600] + [250, 500, 1000, 1500, 2500, 5000]
#INCREASES=[1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.5]

#example_output_path = base_3 + r"adjust_lr_long_${subset}_${cond}_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"

CHANGE_JMPS1 = [20, 30, 40, 50, 60, 70, 90, 100] + [5, 10, 15, 20, 25, 30, 35, 40, 50, 70] + [25, 75, 125, 150, 200]
CHANGE_JMPS2 = [20, 30, 40, 50, 60, 70, 90, 100] + [5, 10, 15, 20, 25, 30, 35, 40, 50, 70] + [200]
CHANGE_JMPS3 = [50, 100, 150, 400]
CHANGE_JMPS4 = [50, 100, 150, 400]
CHANGE_JMPS5 = [50, 100, 150, 400]

in_graph_params = {
        "lrate": LRATES,
        "decay": DECAYS,
        "lrjump": LRJUMPS,
        "minimal": MINIMALS,
        "inc": INCREASES,
        "jmp": JUMPS,
        "strt": STARTING,
        "cng_jmp1": CHANGE_JMPS1,
        "cng_jmp2": CHANGE_JMPS2,
        "cng_jmp3": CHANGE_JMPS3,
        "cng_jmp4": CHANGE_JMPS4,
        "cng_jmp5": CHANGE_JMPS5,
        }

between_graph_params = {
        "cond": CONDITION,
#        "inc": INCREASES,
#        "jmp": JUMPS,
#        "strt": STARTING,
        "subset": SUBSETS,
#        "decay": DECAYS,
#        "lrjump": LRJUMPS,
#        "minimal": MINIMALS,
#        "lrate": LRATES,
        }

#print("curriculum grid:")
#curriculum_grid_search = choose_lr_parameters(example_output_path_curriculum, in_graph_params)
##curriculum_grid_search = curriculum_grid_search.replace("cross_val", "cross_val_test").replace("_0.05_", "_0.04_")
#print("vanilla grid:")
#vanilla_grid_search = choose_lr_parameters(example_output_path_vanilla, in_graph_params)

#vanilla_grid_search = vanilla_grid_search.replace("cross_val", "cross_val_test").replace("_0.05_", "_0.04_")
#print("random grid:")
#random_grid_search = choose_lr_parameters(example_output_path_random, in_graph_params)
#print("anti grid:")
#anti_grid_search = choose_lr_parameters(example_output_path_anti, in_graph_params)




#example_output_path_random = base_3 + r"adjust_lr_16_random_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
#example_output_path_anti = base_3 + r"adjust_lr_16_anti_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"



##### SUMMARY
# original curriculum
example_output_path_curriculum = base_4 + r"adjust_lr_long_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_random = base_4 + r"adjust_lr_long_16_random_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_anti = base_4 + r"adjust_lr_long_16_anti_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# original vanila
example_output_path_vanilla = base_4 + r"adjust_lr_long_16_vanilla_sgd_expo_balance_lr_${lrate}_100_1.9_0.04_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# same network
example_output_path_same_network = base_4 + r"adjust_lr_long_norm_same_network_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# single step
example_output_path_single_step = base_4 + r"adjust_lr_long_16_curriculum_sgd_single_step_balance_lr_${lrate}_${jmp}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# self pace
example_output_path_self_pace = base_4 + r"adjust_lr_long_16_self_pace_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
example_output_path_anti_self_pace = base_4 + r"adjust_lr_long_16_anti_self_pace_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history"
# jmp change
example_output_path_change_jumps = base_4 + r"adjust_lr_long_16_curriculum_sgd_expo_change_jumps_${cng_jmp1}_${cng_jmp2}_${cng_jmp3}_${cng_jmp4}_${cng_jmp5}_balance_lr_0.03_100_2_0.04_decay1.7_min1e-4_lrjmp600_history"
example_output_path_change_jumps_random = base_4 + r"adjust_lr_long_16_random_sgd_expo_change_jumps_${cng_jmp1}_${cng_jmp2}_${cng_jmp3}_${cng_jmp4}_${cng_jmp5}_balance_lr_0.03_100_2_0.04_decay1.7_min1e-4_lrjmp600_history" 
# 2 jumps
example_output_path_2_first_jumps = base_4 + r"adjust_lr_longer_16_curriculum_sgd_expo_2_first_jumps_${cng_jmp1}_${cng_jmp2}_balance_lr_0.035_100_1.9_${strt}_decay1.6_min1e-4_lrjmp600_history"
#example_output_path_2_first_jumps = base_4 + r"cifar100_vgg_long_curriculum_single_step_no_augmentation_lr_0.9_jmp500_strt${strt}_inc1.9_chng1_${cng_jmp1}_chng2_${cng_jmp2}_history"
print("curriculum grid:")
curriculum_grid_search = choose_lr_parameters(example_output_path_curriculum, in_graph_params)
#print("anti grid:")
#anti_grid_search = choose_lr_parameters(example_output_path_anti, in_graph_params)
#print("random grid:")
#random_grid_search = choose_lr_parameters(example_output_path_random, in_graph_params)
print("vanilla grid:")
vanilla_grid_search = choose_lr_parameters(example_output_path_vanilla, in_graph_params)
#print("same_network grid:")
#same_network_grid_search = choose_lr_parameters(example_output_path_same_network, in_graph_params)
#print("single_step grid:")
#single_step_grid_search = choose_lr_parameters(example_output_path_single_step, in_graph_params)
#print("self_path grid:")
#self_path_grid_search = choose_lr_parameters(example_output_path_self_pace, in_graph_params)
#print("anti_self_path grid:")
#anti_self_path_grid_search = choose_lr_parameters(example_output_path_self_pace, in_graph_params)
#print("change_jumps grid:")
#change_jumps_grid_search = choose_lr_parameters(example_output_path_change_jumps, in_graph_params)
#print("change_jumps_random grid:")
#change_jumps_random_grid_search = choose_lr_parameters(example_output_path_change_jumps_random, in_graph_params)
print("two_first grid:")
two_first_jumps_random_grid_search = choose_lr_parameters(example_output_path_2_first_jumps, in_graph_params)



#vanilla_easy_grid_search = choose_lr_parameters(example_vanilla_easy, in_graph_params)
#vanilla_mid_grid_search = choose_lr_parameters(example_vanilla_mid, in_graph_params)
#vanilla_hard_grid_search = choose_lr_parameters(example_vanilla_hard, in_graph_params)
#print(vanilla_easy_grid_search)
#print(vanilla_mid_grid_search)
#print(vanilla_hard_grid_search)

#curriculum_grid_search = r"/cs/labs/daphna/guy.hacohen/project/models3/adjust_lr_long_16_curriculum_sgd_expo_balance_lr_0.03_100_1.9_0.04_decay2_min1e-3_lrjmp600_history"
#vanilla_grid_search = r"/cs/labs/daphna/guy.hacohen/project/models3/adjust_lr_long_16_vanilla_sgd_expo_balance_lr_0.08_100_1.9_0.04_decay2_min1e-4_lrjmp500_history"
#random_grid_search = r"/cs/labs/daphna/guy.hacohen/project/models3/adjust_lr_long_16_random_sgd_expo_balance_lr_0.045_100_1.9_0.04_decay1.7_min1e-3_lrjmp300_history"
#anti_grid_search = r"/cs/labs/daphna/guy.hacohen/project/models3/adjust_lr_long_16_anti_sgd_expo_balance_lr_0.03_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
#curriculum_grid_search = choose_lr_parameters(example_output_path, in_graph_params)
#curriculum_grid_search = curriculum_grid_search.replace("cross_val", "cross_val3")


#print("curriculum:", curriculum_grid_search)
#print("vanilla:", vanilla_grid_search)
#print("random:", random_grid_search)
#print("anti:", anti_grid_search)


def plot_history_list(history_list, names, title, bold_first=False, tag="acc"):
    histories = []
    for history_path in history_list:
        with open(history_path, "rb+") as history_file:
            history = pickle.load(history_file)
        histories.append(history)
    draw_results.plot_keras_history_2(histories[0], histories[1:],
                                  train=False, name1=names[0],
                                  names2=names[1:],
                                  bold=bold_first,
                                  y_tag=tag)
    # draw_results.plot_keras_history_2(history_curriculum2)
    
    plt.grid()
    plt.title(title)
    plt.show()
    if tag == "loss":
        std = False
    else:
        std = True
    bar_plot_histories(histories, names, "val_" + tag, std=std)
    plt.title(title)
#    plt.ylim([0.54, 0.58])
    plt.show()


def cycle_set_param_lr_scheduler_generator(min_lr, max_lr, step_size_batchs):
    gamma = (max_lr / min_lr) ** (1/step_size_batchs)
    cycle_size = step_size_batchs * 2
    step_lr_vals = [min_lr * gamma**(batch) for batch in range(step_size_batchs)]
    lr_vals = np.concatenate((step_lr_vals, step_lr_vals[::-1]))

    def cycle_lr_scheduler(initial_lr, batch, history):
        return lr_vals[batch % cycle_size]

    return cycle_lr_scheduler

def plot_lr_vs_acc(history_path, min_lr, max_lr, title):
    with open(history_path, "rb+") as history_file:
        history = pickle.load(history_file)
    num_batchs = history['batch_num'][-1]
    lr_scheduler = cycle_set_param_lr_scheduler_generator(min_lr, max_lr, num_batchs)
    plt.semilogx([lr_scheduler(None, i, None) for i in history['batch_num']], history["val_acc"])
    plt.xlabel("lr")
    plt.ylabel("top-1 accuracy")
#    plt.xticks(np.linspace(min_lr, max_lr, 10))
    plt.grid()
    plt.title(title)
    plt.show()
    plt.plot([lr_scheduler(None, i, None) for i in history['batch_num']], history["val_acc"])
    plt.xlabel("lr")
    plt.ylabel("top-1 accuracy")
#    plt.xticks(np.linspace(min_lr, max_lr, 10))
    plt.grid()
    plt.title(title)
    plt.show()
    
#plot_lr_vs_acc(base_3 + r"vanilla_find_cycle_parameters_cifar10_history",
#               1e-5, 0.9, "vanilla accuracy as function of lr")

#lrates = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
#lrates = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
lrates = [0.08, 0.1]
jmps = [300, 400, 500, 600, 700]

#plot_history_list([base_3 + "cifar100_vgg_vanilla_no_augmentation_history"] + [base_3 + "cifar100_vgg_curriculum_no_augmentation_lr_" + str(lr) + "_jmp" + str(jmp) + "_inc1.9_strt0.08_history" for lr, jmp in itertools.product(lrates, jmps)],
#                  ["vanilla vgg"] + ["curriculum vgg lr  " + str(lr) + " jmp " + str(jmp) for lr, jmp in itertools.product(lrates, jmps)],
#                  "vanilla vs curriculum - vgg model")

#max_lr_list = [0.04]
#max_lr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
#max_lr_list = [0.06]
#plot_history_list(([base_3 + r"vanilla_cycle_decaying_min_1e-4_max_" + str(lr) + "_history"
#                    for lr in max_lr_list] + [vanilla_grid_search]),
#                   (["cycle vanilla " + str(lr) for lr in max_lr_list] +
#                    ["grid search vanilla"]),
#                    "compare cycle lr vs grid search - vanilla subset 16")
#plot_history_list(([base_3 + r"curriculum_cycle_decaying_min_1e-4_max_" + str(lr) + "_history"
#                    for lr in max_lr_list] + [curriculum_grid_search]),
#                   (["cycle curriculum " + str(lr) for lr in max_lr_list] +
#                    ["grid search curriculum"]),
#                    "compare cycle lr vs grid search - curriculum subset 16")
#plot_history_list([base_3 + r"curriculum_cycle_decaying_min_1e-4_max_0.06_history",
#                   base_3 + r"vanilla_cycle_decaying_min_1e-4_max_0.07_history",
#                   curriculum_grid_search,
#                   vanilla_grid_search],
#                  ["cycle curriculum",
#                   "cycle vanilla",
#                   "grid search curriculum",
#                   "grid search vanilla"],
#                   "compare cycle lr vs grid search")
#plot_history_list(([base_3 + "vanilla_cycle_decaying_min_1e-3_max_0.09_cifar10_history",
#                    vanilla_grid_search]),
#                   (["cycle vanilla "] +
#                    ["grid search vanilla"]),
#                    "compare cycle lr vs grid search - vanilla subset 16")
#plot_history_list(([base_3 + "cycle_lr_cifar10_vanilla_sgd_expo_balance_100_1.9_0.04_minlr1e-3_maxlr0.09_history",
#                    base_3 + "cycle_lr_cifar10_curriculum_sgd_expo_balance_100_1.9_0.04_minlr1e-3_maxlr0.09_history"]),
##                    vanilla_grid_search,
##                    curriculum_grid_search]),
#                   (["cycle vanilla", "cycle curriculum"]),
##                    ["grid search vanilla", "grid search curriculum"]),
#                    "compare cycle lr vs grid search")

#plot_history_list([base_3 + "adjust_lr_rbootstrap_16_curriculum_sgd_expo_balance_lr_0.03_100_1.9_0.04_decay2_min1e-3_lrjmp600_num_boots_repeats1_history_boost0",
#                   base_3 + "adjust_lr_rbootstrap_16_curriculum_sgd_expo_balance_lr_0.03_100_1.9_0.04_decay2_min1e-3_lrjmp600_num_boots_repeats1_history_boost1",
#                   base_3 + "adjust_lr_rbootstrap_16_curriculum_sgd_expo_balance_lr_0.03_100_1.9_0.04_decay2_min1e-3_lrjmp600_num_boots_repeats1_history_boost2",
#                   base_3 + "adjust_lr_rbootstrap_16_curriculum_sgd_expo_balance_lr_0.03_100_1.9_0.04_decay2_min1e-3_lrjmp600_num_boots_repeats1_history_boost3",
#                   vanilla_grid_search,
#                   base_3 + "adjust_lr_rep_same_network_16_curriculum_sgd_expo_balance_lr_0.035_100_1.9_0.04_decay2_min1e-4_lrjmp600_history"],
#                   ["boost0", "boost1", "boost2", "boost3", "vanilla", "same_network"],
#                    "bootstrap subset 16")

plot_history_list([vanilla_grid_search,
                   curriculum_grid_search,
#                   anti_grid_search,
#                   random_grid_search,
#                   same_network_grid_search,
#                   single_step_grid_search,
#                   self_path_grid_search,
#                   anti_self_path_grid_search,
##                   change_jumps_grid_search,
##                   change_jumps_random_grid_search,
                   two_first_jumps_random_grid_search
                   
                   ],
                   ["vanilla",
                    "curriculum",
#                    "anti",
#                    "random",
#                    "same_net", 
#                    "single_step",
#                    "self_pace",
#                    "anti_self_pace",
#                    "change_jumps",
#                    "change_jumps_random",
                    "two_first"
                    ],
                    "",
#                    "compare all major conditions",
                    bold_first=True)

#plot_history_list([base_3 + r"adjust_lr_stl10_16_curriculum_sgd_expo_balance_lr_0.05_200_1.9_0.08_decay1.1_min1e-4_lrjmp800_history",
#                   base_3 + r"adjust_lr_stl10_16_vanilla_sgd_expo_balance_lr_0.03_100_1.9_0.02_decay1.5_min1e-4_lrjmp800_history"],
#    ["curriculum", "vanilla"], "compare curriculum/vanilla in stl10")

#plot_history_list([vanilla_grid_search, vanilla_easy_grid_search, vanilla_mid_grid_search, vanilla_hard_grid_search],
#                  ["vanilla regular", "vanilla easy grid", "vanilla mid grid", "vanilla hard grid"],
#                  "test hardness comparison")

#plot_history_list([r"/cs/labs/daphna/guy.hacohen/project/models3/debug_boost_history_boost0",
#                   r"/cs/labs/daphna/guy.hacohen/project/models3/debug_boost_history_boost1",
#                   r"/cs/labs/daphna/guy.hacohen/project/models3/debug_boost_history_boost2"],
#                  ["debug boost 0", "debug boost 1", "debug boost 2"],
#                  "debug boost")

#orders = ["inception", "resnet", "xception", "vgg19", "vgg16", "freq", "prototype", "same_network"]
orders = ["inception", "resnet", "same_network"]

#plot_history_list([base_4 + "adjust_lr_long_" + order + "_16_curriculum_sgd_single_step_balance_lr_0.035_300_0.1_decay1.6_min1e-4_lrjmp500_history" for order in orders] + [vanilla_grid_search],
#                   orders + ["vanilla"],
#                   "compare different order choices - single step")
#
##adjust_lr_16_curriculum_sgd_expo_balance_lr_0.035_100_2.5_0.04_decay1.7_min1e-4_lrjmp400_history
#jumps = ["20", "50", "100", "200", "400"]
#plot_history_list( [vanilla_grid_search] + [base_4 + "adjust_lr_16_curriculum_sgd_expo_balance_lr_0.035_"+jmp+"_1.9_0.04_decay1.7_min1e-4_lrjmp400_history" for jmp in jumps],
#                   ["vanilla"] + jumps,
#                   "compare different jumps",
#                   bold_first=True)
#
#
#plot_history_list( [base_4 + r"vanilla_basic_lr_change_debug_history", base_4 + r"random_basic_lr_change_debug_history"],
#                   ["vanilla", "random"],
#                   "change lr random playing with basic",
#                   bold_first=True)
#
#plot_history_list( [vanilla_grid_search,
#                    base_4 + r"adjust_lr_long_16_vanilla_sgd_expo_balance_lr_0.045_100_1.9_0.04_decay1.6_min1e-4_lrjmp500_history",
#                    base_4 + r"adjust_lr_replong_same_network_16_curriculum_sgd_expo_balance_lr_0.045_100_1.9_0.1_decay1.6_min1e-4_lrjmp500_history",
#                    base_3 + r"adjust_lr_rbootstrap_16_curriculum_sgd_expo_balance_lr_0.045_100_1.9_0.1_decay1.6_min1e-4_lrjmp500_num_boots1_repeats1_history_boost0",
#                    base_3 + r"adjust_lr_rbootstrap_16_curriculum_sgd_expo_balance_lr_0.045_100_1.9_0.1_decay1.6_min1e-4_lrjmp500_num_boots1_repeats1_history_boost1",
#                    base_3 + r"debug_boots_history",
#                    base_3 + r"debug_boots3_history"],
##                    base_3 + r"debug_same_net_history"],
#                   ["vanilla", "sub-vanilla", "same_network", "bootstrap0", "bootstrap1", "debug_boots", "debug_boots2"],
#                   "compare same network with boot",
#                   bold_first=True)

#exps = ["0.0" + str(i) for i in range(1,10)] + ["0." + str(i) for i in range(10,100)]
#exps = ["0.1", "0.3", "0.5", "0.7", "0.9"]
#exps_paths = [ base_4 + r"change_lr_exponent"+str(exp)+"_16_random_sgd_expo_balance_lr_0.05_50_1.7_0.12_decay1.8_min1e-4_lrjmp600_history" for exp in exps]
#relevent_exps = [exp for i, exp in enumerate(exps) if os.path.exists(exps_paths[i])]
#plot_history_list( [vanilla_grid_search]+ [ base_4 + r"change_lr_exponent"+str(exp)+"_16_random_sgd_expo_balance_lr_0.05_50_1.7_0.12_decay1.8_min1e-4_lrjmp600_history" for exp in relevent_exps],
#                   ["vanilla"] + relevent_exps,
#                   "change_lr with different exponents",
#                   bold_first=True)

#consts = ["0.02", "0.025", "0.03", "0.035", "0.04", "0.045"]
#consts_paths = [ base_4 + r"change_lr_const"+str(const)+"_16_curriculum_sgd_expo_balance_lr_0.05_20_1.7_0.04_decay1.8_min1e-4_lrjmp600_history" for const in consts]
#relevent_conts = [const for i, const in enumerate(consts) if os.path.exists(consts_paths[i])]
#plot_history_list( [vanilla_grid_search]+ [ base_4 + r"change_lr_const"+str(const)+"_16_curriculum_sgd_expo_balance_lr_0.05_20_1.7_0.04_decay1.8_min1e-4_lrjmp600_history" for const in relevent_conts],
#                   ["vanilla"] + relevent_conts,
#                   "change_lr with different consts",
#                   bold_first=True)
##                   tag="loss")

#exps = ["0.0" + str(i) for i in range(1,10)] + ["0." + str(i) for i in range(10,100)]
#exps = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]
#exps_paths = [ base_4 + r"change_lr_exponent"+str(exp)+"_16_curriculum_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.8_min1e-4_lrjmp600_history" for exp in exps]
#relevent_exps = [exp for i, exp in enumerate(exps) if os.path.exists(exps_paths[i])]
#plot_history_list( [vanilla_grid_search]+ [ base_4 + r"change_lr_exponent"+str(exp)+"_16_curriculum_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.8_min1e-4_lrjmp600_history" for exp in relevent_exps],
#                   ["vanilla"] + relevent_exps,
#                   "change_lr with different exponents",
#                   bold_first=True)
##                   tag="loss")

#networks = ["inception", "resnet", "vgg16"]
#outputs = [ base_4 + r"adjust_lr_rep_"+net+"_16_curriculum_sgd_expo_balance_lr_${lrate}_${jmp}_${inc}_${strt}_decay${decay}_min${minimal}_lrjmp${lrjump}_history" for net in networks]
#
#plot_history_list( [vanilla_grid_search] + [choose_lr_parameters(example_out, in_graph_params) for example_out in outputs],
#                   ["vanilla"] + networks,
#                   "compare curriculum of different orders",
#                   bold_first=True)


def print_all_random_orders(min_inx=0):
    index_path = os.path.join(r"/cs/labs/daphna/guy.hacohen/project/random_models/index")
    with open(index_path, 'rb') as file_pi:
        max_order_idx = pickle.load(file_pi)
    base = r"/cs/labs/daphna/guy.hacohen/project/models3/"
    random_base = r"/cs/labs/daphna/guy.hacohen/project/random_models/"
    paths = [base + r"adjust_lr_16_vanilla_sgd_expo_balance_lr_0.08_100_1.9_0.04_decay2_min1e-4_lrjmp500_history"]
    paths += [random_base + "random" + str(i) + "_history" for i in range(min_inx, max_order_idx)]
    names = ["vanilla"]
    names += ["order " + str(i) for i in range(min_inx, max_order_idx)]
    
    plot_history_list(paths, names, "random orders comparison")
    
    
#print_all_random_orders(min_inx=5)
#history_paths = [base + r"boosting"+str(i)+"_16_curriculum_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
#                 for i in [1, 2, 3, 4, 5, 7]]
#names = ["boost " + str(i) for i in [1, 2, 3, 4, 5, 7]]



#plot_history_list(history_paths, names)
##lr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02]
#lr_list = [0.08, 0.06, 0.04, 0.02]
#paths = [base_3 + r"adjust_lr_14_vanilla_sgd_expo_balance_lr_0.06_100_1.9_0.04_decay1.3_min1e-3_lrjmp400_history"]
#paths += [base_3 + r"adjust_lr_14_curriculum_sgd_expo_balance_lr_"+ str(lr) +"_100_1.9_0.04_decay2_min1e-3_lrjmp600_history"
#          for lr in lr_list]
#
#names = ["vanilla"] + ["lr " + str(lr) for lr in lr_list]
#plot_history_list(paths, names, "compare lr for subset 14")
### compare best lr vanilla with best lr curriculum
#

vanilla_best = base + r"adjust_lr_16_vanilla_long_sgd_expo_lr_0.07_100_1.9_0.04_decay1.7_min1e-3_lrjmp400_history"

anti_best = base + r"adjust_lr_16_anti_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
vanilla_sub = base + r"adjust_lr_16_vanilla_long_sgd_expo_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
curriculum_sub = base + r"adjust_lr_16_curriculum_long_sgd_expo_balance_lr_0.07_100_1.9_0.04_decay1.7_min1e-3_lrjmp400_history"
curriculum_best = base + r"adjust_lr_16_curriculum_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
curriculum_best_train = base + r"adjust_lr_16_curriculum_sgd_expo_balance_lr_0.1_100_1.9_0.04_decay2_min1e-3_lrjmp500_history"
random_vanilla = base + r"adjust_lr_16_random_long_sgd_expo_balance_lr_0.07_100_1.9_0.04_decay1.7_min1e-3_lrjmp400_history"
random_curriculum = base + r"adjust_lr_16_random_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
freq_curriculum = base + r"adjust_lr_16_curriculum_freq_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
prototype_curriculum = base + r"adjust_lr_16_curriculum_prototype_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
small_network_curriculum = base + r"adjust_lr_16_curriculum_small_network_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
same_network_curriculum = base + r"adjust_lr_16_curriculum_same_network_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
same_network_repeat = base + r"adjust_lr_16_curriculum_same_network_longlong_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
cross_vanilla_sub = base + r"cross_validate_16_vanilla_sgd_expo_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
cross_vanilla_best = base +  r"cross_validate_16_vanilla_sgd_expo_lr_0.07_100_1.9_0.04_decay1.7_min1e-3_lrjmp400_history"
random_best = base + r"adjust_lr_16_random_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.1_min1e-3_lrjmp250_history"
anti_best = base + r"adjust_lr_16_anti_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.1_min1e-4_lrjmp250_history"
boosting = base + r"boosting2_16_curriculum_long_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"

#cross_vanilla_best = base + r"adjust_lr_16_vanilla_sgd_expo_lr_0.07_100_1.9_0.04_decay1.7_min1e-4_lrjmp500_history"
## random test
cross_vanilla_best = base_debug + r"cross_val_fixed_test_16_vanilla_sgd_expo_balance_lr_0.07_100_1.9_0.04_decay1.8_min1e-4_lrjmp500_history"
cross_vanilla_sub = base + r"cross_validate_16_vanilla_sgd_expo_lr_0.025_100_1.9_0.04_decay1.5_min1e-3_lrjmp500_history"
cross_curriculum_sub = base + r"cross_validate_16_curriculum_sgd_expo_balance_lr_0.07_100_1.9_0.04_decay1.7_min1e-3_lrjmp400_history"
#cross_curriculum_best = base + r"adjust_lr_16_curriculum_sgd_expo_balance_lr_0.04_100_1.9_0.04_decay1.7_min1e-4_lrjmp600_history"
## random test
cross_curriculum_best = base_debug + r"cross_val_fixed_test_16_curriculum_sgd_expo_balance_lr_0.04_100_1.9_0.04_decay1.61_min1e-4_lrjmp500_history"
#cross_curriculum_best = base_debug + r"cross_val_fixed_test_16_curriculum_sgd_expo_balance_lr_0.045_100_1.9_0.04_decay1.6_min1e-4_lrjmp600_history"
#cross_curriculum_best = base_debug + r"cross_val_fixed_test_16_curriculum_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.3_min1e-3_lrjmp300_history"
#cross_curriculum_best = base + r"adjust_lr_16_curriculum_sgd_expo_balance_lr_0.07_100_1.9_0.04_decay1.3_min1e-4_lrjmp300_history"
cross_anti_best = base + r"cross_validate_16_anti_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.1_min1e-4_lrjmp250_history"
cross_random_best = base + r"cross_validate_16_random_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.1_min1e-3_lrjmp250_history"



## subset 14
#
#
#vanilla_best = base + r"adjust_lr_14_vanilla_long_sgd_expo_lr_0.08_100_1.9_0.04_decay1.1_min1e-3_lrjmp600_history"
#anti_best = base + r"adjust_lr_14_anti_sgd_expo_balance_lr_0.07_100_1.9_0.04_decay1.1_min1e-3_lrjmp50_history"
#random_best = base + r"adjust_lr_14_random_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.1_min1e-3_lrjmp100_history"
#vanilla_sub = base + r"adjust_lr_14_vanilla_long_sgd_expo_lr_0.05_100_1.9_0.04_decay1.3_min1e-3_lrjmp400_history"
#curriculum_sub = base + r"adjust_lr_14_curriculum_long_sgd_expo_balance_lr_0.08_100_1.9_0.04_decay1.1_min1e-3_lrjmp600_history"
#curriculum_best = base + r"adjust_lr_14_curriculum_long_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.3_min1e-3_lrjmp400_history"
#random_vanilla = base + r"adjust_lr_14_random_long_sgd_expo_balance_lr_0.08_100_1.9_0.04_decay1.1_min1e-3_lrjmp600_history"
#random_curriculum = base + r"adjust_lr_14_random_long_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.3_min1e-3_lrjmp400_history"

### subset 0
##
#vanilla_best = base + r"adjust_lr_0_vanilla_sgd_expo_lr_0.05_100_1.9_0.04_decay1.5_min1e-4_lrjmp600_history"
#anti_best = base + r"adjust_lr_0_anti_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.1_min1e-4_lrjmp50_history"
#random_best = base + r"adjust_lr_0_random_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.1_min1e-3_lrjmp50_history"
#vanilla_sub = base + r"adjust_lr_0_vanilla_sgd_expo_lr_0.025_100_1.9_0.04_decay1.7_min1e-4_lrjmp500_history"
#curriculum_sub = base + r"adjust_lr_0_curriculum_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.5_min1e-4_lrjmp600_history"
#curriculum_best = base + r"adjust_lr_0_curriculum_sgd_expo_balance_lr_0.025_100_1.9_0.04_decay1.5_min1e-4_lrjmp600_history"
#random_vanilla = base + r"adjust_lr_0_random_long_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.3_min1e-3_lrjmp500_history"
#random_curriculum = base + r"adjust_lr_0_random_long_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.7_min1e-3_lrjmp300_history"

# subset 7

#vanilla_best = base + r"adjust_lr_7_vanilla_sgd_expo_lr_0.1_100_1.9_0.04_decay1.1_min1e-4_lrjmp100_history"
#anti_best = base + r"adjust_lr_7_anti_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.1_min1e-4_lrjmp100_history"
#random_best = base + r"adjust_lr_7_random_sgd_expo_balance_lr_0.07_100_1.9_0.04_decay1.1_min1e-3_lrjmp100_history"
#vanilla_sub = base + r"adjust_lr_7_vanilla_sgd_expo_lr_0.05_100_1.9_0.04_decay1.5_min1e-4_lrjmp500_history"
#curriculum_sub = base + r"adjust_lr_7_curriculum_sgd_expo_balance_lr_0.1_100_1.9_0.04_decay1.1_min1e-4_lrjmp100_history"
#curriculum_best = base + r"adjust_lr_7_curriculum_sgd_expo_balance_lr_0.05_100_1.9_0.04_decay1.5_min1e-4_lrjmp500_history"


with open(vanilla_best, "rb+") as history_file:
    history_vanilla_best = pickle.load(history_file)
with open(vanilla_sub, "rb+") as history_file:
    history_vanilla_sub = pickle.load(history_file)
with open(curriculum_sub, "rb+") as history_file:
    history_curriculum_sub = pickle.load(history_file)
with open(curriculum_best, "rb+") as history_file:
    history_curriculum_best = pickle.load(history_file)
with open(random_vanilla, "rb+") as history_file:
    history_random_vanilla = pickle.load(history_file)
with open(random_curriculum, "rb+") as history_file:
    history_random_curriculum = pickle.load(history_file)
with open(freq_curriculum, "rb+") as history_file:
    history_freq_curriculum = pickle.load(history_file)
with open(prototype_curriculum, "rb+") as history_file:
    history_prototype_curriculum = pickle.load(history_file)
with open(small_network_curriculum, "rb+") as history_file:
    history_small_network_curriculum = pickle.load(history_file)
with open(same_network_curriculum, "rb+") as history_file:
    history_same_network_curriculum = pickle.load(history_file)
with open(same_network_repeat, "rb+") as history_file:
    history_same_network_repeat = pickle.load(history_file)
with open(anti_best, "rb+") as history_file:
    history_anti_best = pickle.load(history_file)
with open(cross_vanilla_sub, "rb+") as history_file:
    history_cross_vanilla_sub = pickle.load(history_file)
with open(cross_vanilla_best, "rb+") as history_file:
    history_cross_vanilla_best = pickle.load(history_file)
with open(anti_best, "rb+") as history_file:
    history_anti_best = pickle.load(history_file)
with open(random_best, "rb+") as history_file:
    history_random_best = pickle.load(history_file)
with open(boosting, "rb+") as history_file:
    history_boosting = pickle.load(history_file)
    
#with open(curriculum_grid_search, "rb+") as history_file:
#    history_curriculum_grid_search = pickle.load(history_file)
#with open(vanilla_grid_search, "rb+") as history_file:
#    history_vanilla_grid_search = pickle.load(history_file)
#with open(random_grid_search, "rb+") as history_file:
#    history_random_grid_search = pickle.load(history_file)
#with open(anti_grid_search, "rb+") as history_file:
#    history_anti_grid_search = pickle.load(history_file)


with open(cross_vanilla_best, "rb+") as history_file:
    history_cross_vanilla_best = pickle.load(history_file)
with open(cross_vanilla_sub, "rb+") as history_file:
    history_cross_vanilla_sub = pickle.load(history_file)
with open(cross_curriculum_sub, "rb+") as history_file:
    history_cross_curriculum_sub = pickle.load(history_file)
with open(cross_curriculum_best, "rb+") as history_file:
    history_cross_curriculum_best = pickle.load(history_file)
with open(cross_anti_best, "rb+") as history_file:
    history_cross_anti_best = pickle.load(history_file)
with open(cross_random_best, "rb+") as history_file:
    history_cross_random_best = pickle.load(history_file)

with open(curriculum_best_train, "rb+") as history_file:
    history_curriculum_best_train = pickle.load(history_file)


    
histories = [
#             history_vanilla_best,
#             history_vanilla_sub,
#             history_curriculum_sub,
#             history_curriculum_best,
#             history_random_vanilla,
#             history_random_curriculum,
#             history_freq_curriculum,
#             history_prototype_curriculum,
#             history_small_network_curriculum,
#             history_same_network_curriculum,
#             history_same_network_repeat,
#             history_anti_best,
#             history_cross_vanilla_sub,
#             history_cross_vanilla_best,
#             history_random_best,
#             history_anti_best,
#             history_boosting,
#             history_cross_vanilla_best,
#             history_cross_vanilla_sub,
#             history_cross_curriculum_sub,
#             history_cross_curriculum_best,
#             history_cross_anti_best,
#             history_cross_random_best,
#             history_curriculum_best_train,
#             history_curriculum_grid_search,
#             history_vanilla_grid_search,
#             history_anti_grid_search,
#             history_random_grid_search,
             ]
names = [
#         'vanilla_best',
#         'vanilla_sub',
#         'curriculum_sub',
#         'curriculum_best',
#         'random_vanilla',
#         'random_curriculum',
#         "freq_curriculum",
#         "history_prototype_curriculum",
#         "history_small_network_curriculum",
#         "same_network_curriculum",
#         "same_network_repeat",
#         "anti_best",
#         'vanilla_cross_sub',
#         'vanilla_cross_best',
#         "random_best",
#         "anti_best",
#         "boosting",
#         "cross vanilla best",
#         "cross vanilla sub",
#         "cross curriculum sub",
#         "cross curriculum best",
#         "cross anti best",
#         "cross random best",
#         "curriculum_best_train",
#         "curriculum grid search",
#         "vanilla_grid_search",
#         "anti grid search",
#         "random_grid_search",
         ]
         
#draw_results.plot_keras_history_2(histories[0], histories[1:],
#                                  train=False, name1=names[0],
#                                  names2=names[1:])
## draw_results.plot_keras_history_2(history_curriculum2)
#
#
#plt.grid()
#plt.title("compare vanilla and curriculum with best learning rates")
#plt.show()
#
#bar_plot_histories(histories, names, "val_acc", std=True)
#plt.title("compare vanilla and curriculum with best learning rates")
#plt.show()




##in_graph_params = {"jmp": jump}
#in_graph_params = {"tresh": tresh}
##in_graph_params = {"lrate": [0.005, 0.015, 0.025, 0.035, 0.05]}
#
#increase = [1.9]
##increase = [0.1]
#conditions = ["curriculum"]
##conditions = ["vanilla"]
#between_graph_params = {"cond": conditions,
#                        "inc": increase,
#                        "strt": [0.04],
##                        "lrate": [0.025, 0.015],
##                        "lrate": [0.005, 0.015, 0.025, 0.035, 0.05],
#                        "lrate": [0.025],
#                        "subset": [16]}

#example_output_path = "/cs/labs/daphna/guy.hacohen/project/models2/sched_${subset}_${condition}_sgd_expo_lr${lrate}_${jump}_${increase}_${start}_history"



#create_graphs(between_graph_params, example_output_path_curriculum, in_graph_params, vanilla_grid_search)





def svm_layers_results(path_list, name_list, max_repeat, start_layer=11, layers_in_net=31):
    results_list = []
    for path in path_list:
        cur_results = []
        for i in range(max_repeat):
            try:
                cur_path = path + "_repeat" + str(i) + "_svm_layers"
                
                with open(cur_path, "rb+") as history_file:
                    svm_layers = pickle.load(history_file)
            except:
                continue
            cur_results.append([svm_layers[j][1] for j in range(len(svm_layers))])
                
        if len(cur_results[0]) != layers_in_net - start_layer:
            cur_results = [i[start_layer:] for i in cur_results]
            
        results_list.append(cur_results)
    return results_list
        

def plot_svm_layers(path_list, name_list, max_repeat,
                    train_dataset, test_dataset, start_layer=11, layers_in_net=31):
    results_list = svm_layers_results(path_list, name_list,
                                      max_repeat, start_layer=start_layer,
                                      layers_in_net=layers_in_net)
        
    def plot_res(results, name):
        
        results_mean = np.mean(results, axis=0)
        results_e = scipy.stats.sem(results, axis=0)
        
        layers = list(range(start_layer, start_layer+len(results_mean)))
        plt.errorbar(layers, results_mean, results_e, linestyle='None', marker='^', label=name)
        plt.title("svm layers accuracy\nnetwork trained on dataset" +
                  str(train_dataset) + 
                  ", accuracy on dataset " + 
                  str(test_dataset))
        plt.xlabel("Layer")
        plt.ylabel("Top 1 Accuracy")
        plt.xticks(layers)
        plt.xlim([start_layer - 1, layers_in_net + 12])
    
    plt.figure()
    for results, name in zip(results_list, name_list):
        plot_res(results, name)
    plt.legend(loc="best")
#    plt.savefig(r"/cs/labs/daphna/guy.hacohen/project/graphs/to_daphna/10.7.18/svm_layers_train_"
#                + str(train_dataset)
#                + "_test_"
#                + str(test_dataset)
#                + ".png")
    plt.show()


#def plot_svm_each_layer(path_list, name_list, param_names, max_repeat,
#                    train_dataset, test_dataset, start_layer=11, layers_in_net=31):
#    results_list = svm_layers_results(path_list, name_list,
#                                  max_repeat, start_layer=start_layer,
#                                  layers_in_net=layers_in_net)
#    def plot_res(layer_results, x_param, name):
#        
#        results_mean = np.mean(layer_results, axis=0)
#        results_e = scipy.stats.sem(layer_results, axis=0)
#        
#        layers = list(range(start_layer, start_layer+len(results_mean)))
#        plt.errorbar(layers, results_mean, results_e, linestyle='None', marker='^', label=name)
#        plt.title("svm layers accuracy\nnetwork trained on dataset" +
#                  str(train_dataset) + 
#                  ", accuracy on dataset " + 
#                  str(test_dataset))
#        plt.xlabel("Layer")
#        plt.ylabel("Top 1 Accuracy")
#        plt.xticks(layers)
#    print(results_list)
#    
#    for param in range(len(param_names)):
#        param_results = results_list[param]
#        layer_results = []
#        for layer in range(layers_in_net - start_layer):
#            cur_layer_res = []
#            for repeat in param_results:
#                cur_layer_res.append(repeat[layer])
#            layer_results.append(cur_layer_res)
#        
#            
#    for layer in range(start_layer, layers_in_net):
#        layer_accuracies = []
#        for i in range(len(name_list)):
#            
#        print(layer_accuracies)
#    
#plot_svm_layers(base_3 + "svm_layers_train_14_test_16_vanilla_balance_lr_0.06_decay1.3_min1e-3_lrjmp400",
#                base_3 + "svm_layers_train_14_test_16_curriculum_balance_lr_0.04_decay2_min1e-3_lrjmp600", 50)

#plot_svm_layers(base_3 + "svm_layers_train_0_test_16_vanilla_balance_lr_0.07_decay1.8_min1e-3_lrjmp600",
#                base_3 + "svm_layers_train_0_test_16_curriculum_balance_lr_0.04_decay1.8_min1e-3_lrjmp400", 50)

train_dataset = 16
test_dataset = 0
#for test_dataset in range(20):
vanilla_path = (base_3 + "svm_layers_train_" + str(train_dataset) + "_test_" +
                str(test_dataset) + "_vanilla_balance_lr_0.08_decay2_min1e-4_lrjmp500")

suboptimal_vanilla_path = (base_3 + "svm_layers_train_" + str(train_dataset) + "_test_" +
                           str(test_dataset) + "_vanilla_balance_lr_0.03_decay2_min1e-3_lrjmp600")

curriculum_path = (base_3 + "svm_layers_train_" + str(train_dataset) + "_test_" +
                str(test_dataset) + "_curriculum_balance_lr_0.03_decay2_min1e-3_lrjmp600")

suboptimal_curriculum_path = (base_3 + "svm_layers_train_" + str(train_dataset) + "_test_" +
                              str(test_dataset) + "_curriculum_balance_lr_0.08_decay2_min1e-4_lrjmp500")

vanilla_change_lr_path_list = []
vanilla_change_lr_name_list = []
curriculum_change_lr_path_list = []
curriculum_change_lr_name_list = []
#change_lr_vals = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
change_lr_vals = [0.02, 0.03, 0.04, 0.06, 0.08, 0.1]
for lr in change_lr_vals:
    vanilla_change_lr_path_list.append(base_3 + "svm_layers_train_" +
                                       str(train_dataset) + "_test_" +
                                       str(test_dataset) +
                                       "_vanilla_balance_lr_" +
                                       str(lr) + "_decay2_min1e-4_lrjmp500")
    vanilla_change_lr_name_list.append("vanilla lr: " + str(lr))
    curriculum_change_lr_path_list.append(base_3 + "svm_layers_train_" +
                                          str(train_dataset) + "_test_" +
                                          str(test_dataset) +
                                          "_curriculum_balance_lr_" +
                                          str(lr) + "_decay2_min1e-3_lrjmp600")
    curriculum_change_lr_name_list.append("curriculum lr: " + str(lr))



#plot_svm_layers(vanilla_change_lr_path_list,
#                vanilla_change_lr_name_list,
#                50,
#                train_dataset,
#                test_dataset)
#
#plot_svm_layers(curriculum_change_lr_path_list,
#                curriculum_change_lr_name_list,
#                50,
#                train_dataset,
#                test_dataset)






#def reduce_order(order, train_idx, test_idx):
#    new_order = np.array(order)[train_idx]
#    
#    for idx in sorted([order[i] for i in test_idx], reverse=True):
#        new_order[new_order > idx] -= 1
#    return new_order
#
#def get_cross_validation_indexes(dataset, num_folds):
#    """
#    gets dataset and number of folds, and return
#    2 matrixes, train and test idxes.
#    each matrix has num_folds rows, each row is a specific
#    train set indexes or test set indexes of given fold.
#    """
#    indexes_path = os.path.join(dataset.data_dir, dataset.name + "_crossval_" + str(num_folds) + "_folds")
#    if os.path.exists(indexes_path):
#        with open(indexes_path, "rb") as input_file:
#            train_idxes, test_idxes = pickle.load(input_file)
#        return train_idxes, test_idxes
#    else:
#        data_size = dataset.test_size + dataset.train_size
#        assert(data_size%num_folds == 0)
#        new_data_order = list(range(data_size))
#        np.random.shuffle(new_data_order)
#        fold_size = data_size // num_folds
#        train_idxes = np.zeros((num_folds, data_size - fold_size), dtype=np.int64)
#        test_idxes = np.zeros((num_folds, fold_size), dtype=np.int64)
#        for i, start_idx in enumerate(range(0, data_size, fold_size)):
#            test_idx = new_data_order[start_idx:(start_idx+fold_size)]
#            train_idx = new_data_order[:start_idx] + new_data_order[(start_idx+fold_size):]
#            test_idxes[i, :] = test_idx
#            train_idxes[i, :] = train_idx
#        with open(indexes_path, "wb") as output_file:
#            pickle.dump((train_idxes, test_idxes), output_file)
#        return train_idxes, test_idxes
#
#dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16)
#train_idxes, test_idxes = get_cross_validation_indexes(dataset, 10)
#dataset.split_data_cross_validation([], "_all_data_in_train")
#(transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
#
#train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
#                                                             transfer_values_test, dataset.y_test, dataset,
#                                                             network_name="inception")
#order_inception = list(transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train))
#
#fold_idx = 0
##dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16)
##dataset.split_data_cross_validation(test_idxes[fold_idx, :], "_fold_" + fold_idx)

#new_order = reduce_order(order_inception, train_idxes[fold_idx, :], test_idxes[fold_idx, :])

#
#with open(r"/cs/labs/daphna/guy.hacohen/project/models2/learn_lr_vanilla_15_history", "rb+") as history_file:
#    history = pickle.load(history_file)
#    
#epochs = 120
##epochs = 200
#lr = [1.005] * (25*epochs); lr[0] = 1e-6
#lr = np.cumprod(lr)
#plt.semilogx(lr, history["loss"])
##plt.plot(lr, history["loss"])
#
#plt.ylim([0, 10])
#plt.xlim([1e-4, 0.5])