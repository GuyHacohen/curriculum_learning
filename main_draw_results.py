import draw_results
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


def rebuild_history(num_epochs):
    history = {}
    stop_iter = 10
    for epoch_stop in range(stop_iter, num_epochs + stop_iter, stop_iter):
        model_dir = r"/cs/labs/daphna/guy.hacohen/project/models"
        history_path = r"stl10_early_stop_epoch_" + str(epoch_stop) + "_out_of_100_3_history"
        with open(os.path.join(model_dir, history_path), "rb+") as history_file:
            cur_history = pickle.load(history_file)
        for key in cur_history.keys():
            if key not in history:
                history[key] = cur_history[key]
            else:
                history[key] += cur_history[key]
    return history


# history = rebuild_history(100)
# for key in history.keys():
#     print(key)
#     print(len(history[key]))
# draw_results.plot_keras_history(history)


# with open(r"/cs/labs/daphna/guy.hacohen/project/models/long_epochs_cifar100_curriculum_super16_9_9_history", "rb+") as history_file:
#     history_curriculum = pickle.load(history_file)
# with open(r"/cs/labs/daphna/guy.hacohen/project/models/long_epochs_cifar_100_super16_history", "rb+") as history_file:
#     history_vanilla = pickle.load(history_file)
# with open(r"/cs/labs/daphna/guy.hacohen/project/models/long_epochs_cifar100_anticurriculum_super16_9_9_history", "rb+") as history_file:
#     history_anticurriculum = pickle.load(history_file)
# with open(r"/cs/labs/daphna/guy.hacohen/project/models/long_epochs_cifar100_random_super16_9_9_history", "rb+") as history_file:
#     history_random = pickle.load(history_file)

# draw_results.plot_keras_history_2(history_vanilla, [history_curriculum, history_anticurriculum, history_random],
#                                   train=False, name1='vanilla', names2=['curriculum', 'anti-curriculum', 'random'])
# # draw_results.plot_keras_history_2(history_curriculum2)
# plt.show()

 


#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_curriculum_sgd_expo_50_2_0.1_history", "rb+") as history_file:
#    history_curriculum = pickle.load(history_file)
#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_lin_50_0.05_history", "rb+") as history_file:
#    history_vanilla = pickle.load(history_file)
##with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_anti_sgd_history", "rb+") as history_file:
##    history_anticurriculum = pickle.load(history_file)
##with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_random_sgd_history", "rb+") as history_file:
##    history_random = pickle.load(history_file)
#
#draw_results.plot_keras_history_2(history_vanilla, [history_curriculum],
#                                  train=False, name1='vanilla', names2=['curriculum'])
## draw_results.plot_keras_history_2(history_curriculum2)
#plt.show()




#jumps = [50, 100, 150, 200, 250, 300, 250, 400, 450, 500]
#increases = [0.05, 0.1, 0.15, 0.2, 0.25]
#conditions = ["anti"]
#
#for increase in increases:
#    histories = []
#    names = []
#    for jump in jumps:
#        for condition in conditions:
#            path = r"/cs/labs/daphna/guy.hacohen/project/models1/sched_" + condition + "_sgd_lin_" + str(jump) + "_" + str(increase) + "_history"
#            try:
#                with open(path, "rb+") as history_file:
#                    history = pickle.load(history_file)
#                histories.append(history)
#                names.append(condition + "_" + str(jump) + "_" + str(increase))
#            except:
#                pass
#    with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_lin_50_0.05_history", "rb+") as history_file:
#        vanilla_history = pickle.load(history_file)
#        
#    cmap = plt.get_cmap('gnuplot')
#    colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(histories))]
##        colors = [str(np.random.rand()) for history in histories]
#    draw_results.plot_keras_history_2(vanilla_history, histories, 
#                                  train=False, name1="vanilla", names2=names,
#                                  bold=True, colors=colors)
#    plt.title("increase: " + str(increase))
#    # draw_results.plot_keras_history_2(history_curriculum2)
#    plt.savefig(r"/cs/labs/daphna/guy.hacohen/project/graphs/sched/linear/sched_jump_anti_" + str(increase) + ".png")
#
#    plt.show()
#
#def see_all_data_batch_expo(batches_to_increase, increase_amount, starting_percent):
#    cur_percent = starting_percent
#    cur_batch = 0
#    while cur_percent < 1:
#        cur_percent *= increase_amount
#        cur_batch += batches_to_increase
#    return cur_batch

#jumps = [50, 100, 150, 200]
##jumps = [100]
##increases = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
#increases = [1.8, 1.9, 2]
#starts = [0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
#conditions = ["curriculum"]


#jumps = [50,100,150,200,250,300,350,400,450,500]
#increases = [1.8,1.9,2,2.1,2.2,2.4,2.6,3,3.3,3.5,4,6]
#starts = [0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
#conditions = ["curriculum"]


#jumps = [150]
#increases= [1.9]
#starts = [0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
#conditions = ["curriculum", "anti", "random"]
#for condition in conditions:
#    for increase in increases:
#        for jump in jumps:
#            histories = []
#            names = []
#            all_data_batchs = []
#        
#            for start in starts:
#                path = r"/cs/labs/daphna/guy.hacohen/project/models1/sched_14_" + condition + "_sgd_expo_" + str(jump) + "_" + str(increase) + "_" + str(start) + "_history"
#                try:
#                    with open(path, "rb+") as history_file:
#                        history = pickle.load(history_file)
#                    histories.append(history)
#                    names.append(condition + "_" + str(jump) + "_" + str(increase) + "_" + str(start))
#                    all_data_batchs.append(see_all_data_batch_expo(jump, increase, start))
#                except:
#                    pass
#        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_14_vanilla_sgd_expo_150_1.1_0.04_history", "rb+") as history_file:
##        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_lin_50_0.05_history", "rb+") as history_file:
#            vanilla_history = pickle.load(history_file)
#        
#        cmap = plt.get_cmap('rainbow')
#        colors = [cmap(i) for i in np.linspace(0, 1, len(histories))]
##        colors = [str(np.random.rand()) for history in histories]
#        draw_results.plot_keras_history_2(vanilla_history, histories, 
#                                      train=False, name1="vanilla", names2=names,
#                                      bold=True, colors=colors)
##                                      y_tag='loss', ylabel='loss')
#        
#        for i in range(len(histories)):
#            plt.axvline(x=all_data_batchs[i], color=colors[i])
#        plt.xlim([0, 2900])
#        plt.title("increase: " + str(increase) + " jump: " + str(jump) + " condition:" + condition)
##         draw_results.plot_keras_history_2(history_curriculum2)
##        plt.savefig(r"/cs/labs/daphna/guy.hacohen/project/graphs/sched/exponent/sched_jump_loss_" + str(jump) + "_increase_" + str(increase) + "_" + condition + ".png")
#    
#        plt.show()


jumps = [50,100,150,200,250,300,350,400,450,500]
increases= [1.9]
starts = [0.04]
conditions = ["curriculum", "anti", 'random']
for condition in conditions:
    for increase in increases:
        for start in starts:
            histories = []
            names = []
            all_data_batchs = []
        
            for jump in jumps:
                path = r"/cs/labs/daphna/guy.hacohen/project/models1/sched_14_" + condition + "_sgd_expo_" + str(jump) + "_" + str(increase) + "_" + str(start) + "_history"
                try:
                    with open(path, "rb+") as history_file:
                        history = pickle.load(history_file)
                    histories.append(history)
                    names.append(condition + "_" + str(jump) + "_" + str(increase) + "_" + str(start))
                    all_data_batchs.append(see_all_data_batch_expo(jump, increase, start))
                except:
                    pass
        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_14_vanilla_sgd_expo_150_1.1_0.04_history", "rb+") as history_file:
#        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_lin_50_0.05_history", "rb+") as history_file:
            vanilla_history = pickle.load(history_file)
        
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(histories))]
#        colors = [str(np.random.rand()) for history in histories]
        draw_results.plot_keras_history_2(vanilla_history, histories, 
                                      train=False, name1="vanilla", names2=names,
                                      bold=True, colors=colors)
#                                      y_tag='loss', ylabel='loss')
        
        for i in range(len(histories)):
            plt.axvline(x=all_data_batchs[i], color=colors[i])
        plt.xlim([0, 2900])
        plt.title("increase: " + str(increase) + " start: " + str(start) + " condition:" + condition)
#         draw_results.plot_keras_history_2(history_curriculum2)
#        plt.savefig(r"/cs/labs/daphna/guy.hacohen/project/graphs/sched/exponent/sched_jump_loss_" + str(jump) + "_increase_" + str(increase) + "_" + condition + ".png")
    
        plt.show()


#jumps = [50]
#increases= [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4,3,4,6]
#starts = [0.04]
#conditions = ["curriculum", "anti"]
#for condition in conditions:
#    for jump in jumps:
#        for start in starts:
#            histories = []
#            names = []
#            all_data_batchs = []
#        
#            for increase in increases:
#                path = r"/cs/labs/daphna/guy.hacohen/project/models1/sched_" + condition + "_sgd_expo_" + str(jump) + "_" + str(increase) + "_" + str(start) + "_history"
#                try:
#                    with open(path, "rb+") as history_file:
#                        history = pickle.load(history_file)
#                    histories.append(history)
#                    names.append(condition + "_" + str(jump) + "_" + str(increase) + "_" + str(start))
#                    all_data_batchs.append(see_all_data_batch_expo(jump, increase, start))
#                except:
#                    pass
#        with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_lin_50_0.05_history", "rb+") as history_file:
#            vanilla_history = pickle.load(history_file)
#        
#        cmap = plt.get_cmap('gnuplot')
#        colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(histories))]
##        colors = [str(np.random.rand()) for history in histories]
#        draw_results.plot_keras_history_2(vanilla_history, histories, 
#                                      train=False, name1="vanilla", names2=names,
#                                      bold=True, colors=colors)
##                                      y_tag='loss', ylabel='loss')
#        
#        for i in range(len(histories)):
#            plt.axvline(x=all_data_batchs[i], color=colors[i])
#        plt.xlim([0, 2900])
#        plt.title("start: " + str(start) + " jump: " + str(jump) + " condition:" + condition)
##         draw_results.plot_keras_history_2(history_curriculum2)
##        plt.savefig(r"/cs/labs/daphna/guy.hacohen/project/graphs/sched/exponent/sched_jump_loss_" + str(jump) + "_increase_" + str(increase) + "_" + condition + ".png")
#    
#        plt.show()


#
#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_curriculum_sgd_history", "rb+") as history_file:
#    history_curriculum = pickle.load(history_file)
#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_history", "rb+") as history_file:
#    history_vanilla = pickle.load(history_file)
#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_anti_sgd_history", "rb+") as history_file:
#    history_anticurriculum = pickle.load(history_file)
#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_random_sgd_history", "rb+") as history_file:
#    history_random = pickle.load(history_file)
#
#draw_results.plot_keras_history_2(history_vanilla, [history_curriculum, history_anticurriculum, history_random],
#                                  train=False, name1='vanilla', names2=['curriculum', 'anti-curriculum', 'random'])
## draw_results.plot_keras_history_2(history_curriculum2)
#plt.show()

#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_curriculum_sgd_history", "rb+") as history_file:
#    history_curriculum = pickle.load(history_file)
#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_vanilla_sgd_history", "rb+") as history_file:
#    history_vanilla = pickle.load(history_file)
#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_anti_sgd_history", "rb+") as history_file:
#    history_anticurriculum = pickle.load(history_file)
#with open(r"/cs/labs/daphna/guy.hacohen/project/models1/sched_random_sgd_history", "rb+") as history_file:
#    history_random = pickle.load(history_file)
#
#draw_results.plot_keras_history_2(history_vanilla, [history_curriculum, history_anticurriculum, history_random],
#                                  train=False, name1='vanilla', names2=['curriculum', 'anti-curriculum', 'random'])
## draw_results.plot_keras_history_2(history_curriculum2)
#plt.show()

# # for order_name in ['gad', 'model', "freq", "prototype", "vgg16", "vgg19", "inception", "xception", "resnet"]:
# for order_name in ["gad"]:
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/ensemble_curriculum_adam_adap10_" + order_name + "_nets10_history", "rb+") as history_file:
#         history_curriculum = pickle.load(history_file)
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/ensemble_vanilla_adam_adap10_" + order_name + "_nets10_history", "rb+") as history_file:
#         history_vanilla = pickle.load(history_file)
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/ensemble_anti_adam_adap10_" + order_name + "_nets10_history", "rb+") as history_file:
#         history_anticurriculum = pickle.load(history_file)
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/ensemble_random_adam_adap10_" + order_name + "_nets10_history", "rb+") as history_file:
#         history_random = pickle.load(history_file)
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/ensemble_curriculum_adam_gad_prototype_inception_history", "rb+") as history_file:
#         history_ensemble_diff = pickle.load(history_file)
#
#     draw_results.plot_keras_history_2(history_vanilla, [history_curriculum, history_anticurriculum, history_random, history_ensemble_diff],
#                                       train=False, name1='vanilla', names2=['curriculum', 'anti-curriculum', 'random', 'ensemble_different'])
#     # draw_results.plot_keras_history_2(history_curriculum2)
#
#     axes = plt.gca()
#     axes.set_ylim([0.35,0.7])
#     plt.title(order_name)
#     plt.show()


# for subset in range(20):
#     if subset != 9:
#         continue
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/debug_curriculum_adam_adap10_gad_subset" + str(subset) + "_history", "rb+") as history_file:
#         history_curriculum = pickle.load(history_file)
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/debug_vanilla_adam_adap10_gad_subset" + str(subset) + "_history", "rb+") as history_file:
#         history_vanilla = pickle.load(history_file)
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/debug_anti_adam_adap10_gad_subset" + str(subset) + "_history", "rb+") as history_file:
#         history_anticurriculum = pickle.load(history_file)
#     with open(r"/cs/labs/daphna/guy.hacohen/project/models/debug_random_adam_adap10_gad_subset" + str(subset) + "_history", "rb+") as history_file:
#         history_random = pickle.load(history_file)

#     draw_results.plot_keras_history_2(history_vanilla, [history_curriculum, history_anticurriculum, history_random],
#                                       train=False, name1='vanilla', names2=['curriculum', 'anti-curriculum', 'random'])
#     # draw_results.plot_keras_history_2(history_curriculum2)

#     axes = plt.gca()
#     min_y = 0.35
#     max_y = 0.8
#     axes.set_yticks(np.arange(min_y, max_y, 0.025))
#     axes.set_ylim([min_y, max_y])
#     plt.grid()
#     plt.title("ensemble gad order, subset: " + str(subset))
#     plt.show()



# with open(r"/cs/labs/daphna/guy.hacohen/project/models/debug_ensemble_history", "rb+") as history_file:
#     history_curriculum = pickle.load(history_file)
#
# draw_results.plot_keras_history_2(history_curriculum,
#                                   train=False, name1='ensemble')
# # draw_results.plot_keras_history_2(history_curriculum2)
#
# axes = plt.gca()
# axes.set_ylim([0.35,0.6])
# plt.show()


# with open(r"/cs/labs/daphna/guy.hacohen/project/gad_files/curriculum_learning/cifar100/subset1/results/large_acc8/adam/0.001/0.005/curriculum/exp0/trainHistoryDict0", "rb+") as history_file:
#     history = pickle.load(history_file)
#     history_curriculum = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
#     for i in range(len(history)):
#         for key in history[i].keys():
#             history_curriculum[key].append(history[i][key])
#         history_curriculum["batch_num"] = list(range(len(history)))
#
# with open(r"/cs/labs/daphna/guy.hacohen/project/gad_files/curriculum_learning/cifar100/subset1/results/large_acc8/adam/0.001/0.005/None/exp0/trainHistoryDict0", "rb+") as history_file:
#     history = pickle.load(history_file)
#     history_vanilla = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
#     for i in range(len(history)):
#         for key in history[i].keys():
#             history_vanilla[key].append(history[i][key])
#             history_vanilla["batch_num"] = list(range(len(history)))
#
# with open(r"/cs/labs/daphna/guy.hacohen/project/gad_files/curriculum_learning/cifar100/subset1/results/large_acc8/adam/0.001/0.005/anti-curriculum/exp0/trainHistoryDict0", "rb+") as history_file:
#     history = pickle.load(history_file)
#     history_anticurriculum = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
#     for i in range(len(history)):
#         for key in history[i].keys():
#             history_anticurriculum[key].append(history[i][key])
#             history_anticurriculum["batch_num"] = list(range(len(history)))
#
# with open(r"/cs/labs/daphna/guy.hacohen/project/gad_files/curriculum_learning/cifar100/subset1/results/large_acc8/adam/0.001/0.005/control-curriculum/exp0/trainHistoryDict0", "rb+") as history_file:
#     history = pickle.load(history_file)
#     history_random = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
#     for i in range(len(history)):
#         for key in history[i].keys():
#             history_random[key].append(history[i][key])
#     history_random["batch_num"] = list(range(len(history)))
#
#
# draw_results.plot_keras_history_2(history_vanilla, [history_curriculum, history_anticurriculum, history_random],
#                                   train=False, name1='vanilla', names2=['curriculum', 'anti-curriculum', 'random'])
# # draw_results.plot_keras_history_2(history_curriculum2)
# plt.show()



#with open(r"/cs/labs/daphna/guy.hacohen/project/models/debug_sgd_none_history", "rb+") as history_file:
#    history_vanilla = pickle.load(history_file)
#with open(r"/cs/labs/daphna/guy.hacohen/project/models/debug_sgd_curriculum_history", "rb+") as history_file:
#    history_curriculum = pickle.load(history_file)
#
#draw_results.plot_keras_history_2(history_vanilla, [history_curriculum],
#                                  train=False, name1='vanilla', names2=['curriculum'])
## draw_results.plot_keras_history_2(history_curriculum2)
#plt.show()