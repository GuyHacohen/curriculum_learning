import numpy as np
import matplotlib.pyplot as plt


def plot_keras_history(history):
    # loss, val_loss = history['loss'], history['val_loss']
    # epochs = len(loss)

    fig, axs = plt.subplots(1, 1, figsize=(10,5))

    # graph_from_history(history, y_tag="loss", ylabel='Loss', axs=axs[1])

    plt.plot(range(10, 110, 10), [0.651875, 0.69475, 0.69525, 0.7005, 0.703375, 0.708625, 0.7105, 0.7125, 0.712375, 0.71375], label='ensemble')
    graph_from_history(history, axs=axs, train=False)

    plt.tight_layout()
    plt.show()

def graph_from_history(history, x_tag='epoch', y_tag='acc', train=True, test=True, label_train='train',
                       label_test='test', xlabel='Epoch number', ylabel='Top-1 Accuracy', axs=None,
                       error=False, bold=False, color=None):
    if bold:
        color="black"
    if x_tag == 'epoch':
        loss = history['loss']
        epochs = len(loss)
        x = list(range(1, epochs+1))
    else:
        x = history[x_tag]
    y = history[y_tag]
    y_val = history["val_" + y_tag]
#    y_val = history[y_tag]
    
    if axs is None:
        fig, axs = plt.plot(figsize=(10,5))
    if train:
        if len(y) > len(x):
            x = range(len(y))
        if not error:
            if bold:
                axs.plot(x, y, label=label_train, color="black", ms="20")
            else:
                axs.plot(x, y, label=label_train, color=color)
        else:
            e = history["std_"+y_tag]
            axs.errorbar(x, y, e, linestyle='None', marker='^', label=label_train)
    if test:
        if len(y_val) > len(x):
            x = range(len(y_val))
        if not error:
            if bold:
                axs.plot(x, y_val, label=label_test, color="black", ms="20")
            else:
                axs.plot(x, y_val, label=label_test, color=color)
        else:
            e = history["std_val_"+y_tag]
            if len(x) > 100:
                space = np.array([int(i) for i in np.linspace(0, len(x)-1, 100)])
            else:
                space = np.array(list(range(len(x))))
            axs.errorbar(np.array(x)[space], np.array(y_val)[space], np.array(e)[space],
                         marker='_', label=label_test, capsize=2, elinewidth=1,
                         markeredgewidth=0.5, markersize=0.01, color=color)

    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
#    axs.legend(loc="best")


def plot_keras_history_2(history, histories_2 = [], train=True, name1='', names2=[],
                         error=False, bold=False, colors=None, y_tag='acc', ylabel='Top-1 Accuracy',
                         figsize=(10, 5), visible_x=True, visible_y=True,
                         legend=True, x_title=True, y_title=True, no_background=False,
                         legend_loc="best"):
    # loss, val_loss = history['loss'], history['val_loss']
    # epochs = len(loss)
    
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.xaxis.grid()
    axs.xaxis.set_visible(visible_x)
    axs.yaxis.set_visible(visible_y)
    # graph_from_history(history, y_tag="loss", ylabel='Loss', axs=axs[1])


    graph_from_history(history, x_tag='batch_num', xlabel='Batch Number', axs=axs,
                       train=train, test=not train, label_test=name1,
                       label_train=name1, error=error, bold=bold,
                       y_tag=y_tag, ylabel=ylabel)

    if histories_2:
        for idx, (history_2, name2) in enumerate(zip(histories_2, names2)):
            if colors is not None:
                color = colors[idx]
            else:
                color = None
            graph_from_history(history_2, x_tag='batch_num', xlabel='Batch Number',
                               axs=axs, train=train, test=not train, label_test=name2,
                               label_train=name2, error=error,
                               color=color, y_tag=y_tag, ylabel=ylabel)



    if not x_title:
        axs.xaxis.set_label_text('')
    if not y_title:
        axs.yaxis.set_label_text('')
#    handles, labels = axs.get_legend_handles_labels()
#    # sort both labels and handles by labels
#    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: -t[0]))
#    axs.legend(handles, labels)
    if legend:
        plt.legend(loc=legend_loc)
    if no_background:
        fig.patch.set_facecolor('None')
        fig.patch.set_alpha(0.0)
#    plt.xlim([0, 5000])
    try:
        plt.tight_layout()
    except:
        pass


def plot_cifar_100(x, window_size=3, order=None):
    for i in range(min(window_size**2, x.shape[0])):
#        plt.subplot(330 + 1 + i)
        plt.subplot(window_size, window_size, i+1)
        if order is None:
            image = x[i]
        else:
            image = x[order[i]]
        plt.imshow(image, interpolation='nearest')
    # show the plot
    plt.figure(figsize=(20,10))
    plt.show()


def plot_stl10(x, window_size=3, order=None):
    for i in range(min(window_size**2, x.shape[0])):
#        plt.subplot(330 + 1 + i)
        plt.subplot(window_size, window_size, i+1)
        if order is None:
            image = x[i]
        else:
            image = x[order[i]]
        plt.imshow(image, interpolation='nearest')
    # show the plot
    plt.figure(figsize=(20,10))
    plt.show()