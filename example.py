import matplotlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os


def customplot(adj_left=.13, adj_bottom=.13, figsize=[10.72, 8.205], axes_size=31, tick_size=24, legend_size=24,
               co=False):
    params = {'font.family': 'sans-serif',
              'font.sans-serif': 'Verdana',
              'xtick.labelsize': tick_size,
              'ytick.labelsize': tick_size,
              'axes.labelsize': axes_size,
              'figure.figsize': figsize,
              'legend.loc': 'best',
              'legend.fontsize': legend_size,
              'legend.fancybox': False}
    matplotlib.rcParams.update(params)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=adj_left, bottom=adj_bottom, right=.95, top=.94)
    if co:
        co = open('configs/list_co.txt', 'r')
        co = co.readlines()
        co = [f'#{co[i].strip().replace(" ", "")}' for i in range(len(co))]
        return fig, ax, co
    else:
        return fig, ax


fig, ax = customplot()
plt.plot(np.arange(1, 100), label='TES1')
plt.plot(np.arange(1, 100) * 2, label='TES2')
plt.xlabel('XXX')
plt.ylabel('DDD')
plt.legend()
plt.show()