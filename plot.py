# -*- coding:utf-8 -*-


import argparse
from calendar import c
from pydoc import doc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
import matplotlib
import torch
from matplotlib.colors import LinearSegmentedColormap
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=0.5, **textkw):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
 
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
 
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
 
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
 
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
 
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j]>0.01:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None),fontsize=16, **kw)
                texts.append(text)
 
    return texts


def plot_cka_matrix(cka_matrix, title="matrix",
                    xlabel="input_circuit", ylabel="output_circuit",
                    xlayerids=None, ylayerids=None,
                    every_nth_tick=1, ax=None, fig=None):
    """" Helper Function for Plotting CKA Matrices. """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(120,80))
    im = ax.imshow(cka_matrix.detach().numpy(), vmin=0, vmax=1,cmap='Blues')

    if ylayerids is None:
        yticklabels = list([i for i in range(cka_matrix.shape[0])])
    else:
        assert len(ylayerids) == cka_matrix.shape[0]
        yticklabels = list([str(i) for i in ylayerids])

    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=14)

    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth_tick != 0:
            label.set_visible(False)

    if xlayerids is None:
        xticklabels = [i for i in range(cka_matrix.shape[1])]
    else:
        assert len(xlayerids) == cka_matrix.shape[1]
        xticklabels = [str(i) for i in xlayerids]

    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=28)
    ax.set_xlabel(xlabel, fontsize=14)

    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth_tick != 0:
            label.set_visible(False)
    ax.set_title(title, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Weight", rotation=270, labelpad=30, fontsize=18)
    cbar.ax.tick_params(labelsize=26)
    fig.tight_layout()
    return fig,im





def matrix_plot(matrix,x_labels,y_labels,save_path):
    

    fig, ax = plt.subplots()

    # 创建自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom blue', ['white', 'blue'], N=256)

    # 用颜色显示矩阵值的大小
    cax = ax.matshow(matrix, cmap=cmap)

    # 设置颜色条
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    # 标签
    ax.set_xticklabels(x_labels,rotation=45)
    ax.set_yticklabels(y_labels)

    # 使标签显示在图表中央
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)