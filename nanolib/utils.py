# -*- coding: utf-8 -*-
"""
This file is part of NANOLIB


NANOLIB was primarily developed at Nanosense by:
    Shidiq Nur Hidayat (s.hidayat@nanosense-id.com)

Created on Tue Jul 14 18:23:51 2020

@author: Shidiq Nur Hidayat
"""

import matplotlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from sklearn.metrics import confusion_matrix


def papers(legend_size=22, loc='best', classic=True):
    if classic:
        plt.style.use('classic')
    params = {
        "axes.formatter.useoffset": False,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "axes.labelsize": 28,
        "axes.labelweight": "bold",
        "figure.dpi": 100,
        "figure.figsize": [10.72, 8.205],
        "legend.loc": loc,
        "legend.fontsize": legend_size,
        "legend.fancybox": True,
        "mathtext.fontset": 'custom',
        "mathtext.default": 'regular',
        "figure.autolayout": True,
        "patch.edgecolor": "#000000",
        "text.color": "#000000",
        "axes.edgecolor": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }
    matplotlib.rcParams.update(params)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


# noinspection PyDefaultArgument
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


def train_test_split(x, y, test_size=0.2, random_state=99):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_size,
                                                        random_state=random_state)
    return x_train, x_test, y_train, y_test


class TooMuchUnique(ValueError):
    pass


def saveimg(fig, file='foo', dpi=600, filetype="pdf"):
    path_ = os.path.dirname(file)
    if not os.path.exists(path_):
        os.makedirs(path_)

    if isinstance(filetype, str):
        fig.savefig(f"{file}.{filetype}", dpi=dpi)
    elif isinstance(filetype, list):
        for i in filetype:
            fig.savefig(f"{file}.{filetype[i]}", dpi=dpi)


# noinspection PyUnusedLocal
def plotroc(y_test, y_score, lw=3, colors=None, multiclass=True):
    if colors is None:
        colors = ['aqua', 'darkorange', 'cornflowerblue']

    from sklearn.metrics import roc_curve, auc
    from numpy import interp

    if multiclass:
        n_classes = y_test.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fig, ax = customplot()
        ax = plt.plot(fpr["micro"], tpr["micro"],
                      label='micro-average ROC curve (area = {0:0.2f})'
                            ''.format(roc_auc["micro"]),
                      color='deeppink', linestyle=':', linewidth=4)

        ax = plt.plot(fpr["macro"], tpr["macro"],
                      label='macro-average ROC curve (area = {0:0.2f})'
                            ''.format(roc_auc["macro"]),
                      color='navy', linestyle=':', linewidth=4)

        colors = itertools.cycle(colors)
        for i, color in zip(range(n_classes), colors):
            ax = plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                          label='ROC curve of class {0} (area = {1:0.2f})'
                                ''.format(i, roc_auc[i]))

        ax = plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax = plt.xlim([-0.05, 1.0])
        ax = plt.ylim([0.0, 1.05])
        ax = plt.xlabel('False Positive Rate')
        ax = plt.ylabel('True Positive Rate')
        ax = plt.legend(loc="lower right")
    else:
        # fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = customplot()
        ax = plt.plot(fpr, tpr, color='darkorange', lw=lw,
                      label='ROC curve (area = %0.2f)' % roc_auc)
        ax = plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax = plt.xlim([-0.05, 1.0])
        ax = plt.ylim([0.0, 1.05])
        ax = plt.xlabel('False Positive Rate')
        ax = plt.ylabel('True Positive Rate')
        ax = plt.legend(loc='lower right')

    return fig, (fpr, tpr, roc_auc)


# noinspection PyUnresolvedReferences
def plot_confusion_matrix(cm_, classes, cmap=plt.cm.RdPu, fontsize=26, xrot=0, yrot=0):
    plt.imshow(cm_, interpolation='nearest', cmap=cmap, aspect='equal')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrot)
    plt.yticks(tick_marks, classes, rotation=yrot)

    fmt = 'd'
    thresh = cm_.max() / 2.
    for i, j in itertools.product(range(cm_.shape[0]), range(cm_.shape[1])):
        plt.text(j, i, format(cm_[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_[i, j] > thresh else "black",
                 fontsize=fontsize,
                 )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def runkfoldcv(estimator, x, y, scaling=None, cv=None, random_state=99):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RepeatedKFold

    if scaling is None:
        scaling = StandardScaler()

    if cv is None:
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=random_state)

    score = list()
    for train, test in cv.split(x):
        x_train = scaling.fit_transform(x[train])
        x_test = scaling.transform(x[test])
        y_train = y[train]
        y_test = y[test]
        estimator.fit(x_train, y_train)
        score.append(estimator.score(x_test, y_test))

    print("Baseline: %.2f%% (%.2f%%)" % (np.mean(score) * 100, np.std(score) / np.sqrt(len(score)) * 100))

    return estimator, score


def readini(file, section, key):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(file)
    return config.get(section, key)


def writeini(file, section, key, string):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(file)
    config[section][key] = string

    with open(file, 'w') as configfile:
        config.write(configfile)


def printprogressbar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def sample(x, n=None, random_state=None):
    from random import sample
    from random import seed

    if not isinstance(x, list):
        items = [x for x in range(x)]
    else:
        items = x

    if n is None or n > len(items):
        n = len(items)

    seed(random_state)
    return sample(items, n)


def match(a, b):
    return [b.index(x) + 1 if x in b else None for x in a]


def url(foldername='run'):
    import os
    get_folder = os.path.join(os.curdir, foldername)
    os.makedirs(get_folder, exist_ok=True)
    return get_folder


# noinspection PyPep8Naming
def ClassificationReport(ytrue, ypred, plotcm=False, file=None, **options):
    from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef, confusion_matrix

    a = options.get('adj_left', 0.1)
    b = options.get('adj_bottom', 0.2)
    figsize = options.get('figsize', [4, 4])
    axes_size = options.get('axes_size', 22)
    # noinspection PyUnresolvedReferences
    cmap = options.get('cmap', plt.cm.RdPu)
    fontsize = options.get('fontsize', 26)
    xrot = options.get('xrot', 0)
    show = options.get('show', True)

    acc = cohen_kappa_score(ytrue, ypred)
    cm = confusion_matrix(ytrue, ypred)
    matt = matthews_corrcoef(ytrue, ypred)
    cr = classification_report(ytrue, ypred)

    print(f'\nConfusion matrix:\n{cm}')
    print(f'\nCohen kappa score        : {np.round(acc, 3)}')
    print(f'Matthews correlation coef: {np.round(matt, 3)}\n')
    print(cr)

    if file is not None:
        from pycm import ConfusionMatrix

        path_ = os.path.dirname(file)
        if not os.path.exists(path_):
            os.makedirs(path_)

        CM = ConfusionMatrix(ytrue, ypred)
        CM.save_html(file)

    if plotcm:
        fig, _ = customplot(adj_bottom=b, adj_left=a, figsize=figsize,
                            axes_size=axes_size)
        plot_confusion_matrix(cm, classes=np.unique(ytrue), cmap=cmap,
                              fontsize=fontsize, xrot=xrot)
        if show:
            plt.show()
        return fig


# noinspection PyPep8Naming
def FullClassificationReport(model, xtrain, xtest, ytrain, ytest, bypass=False, scoring=None, **options):
    a = options.get('adj_left', 0.1)
    b = options.get('adj_bottom', 0.2)
    figsize = options.get('figsize', [4, 4])
    axes_size = options.get('axes_size', 22)
    # noinspection PyUnresolvedReferences
    cmap = options.get('cmap', plt.cm.RdPu)
    fontsize = options.get('fontsize', 26)
    xrot = options.get('xrot', 0)
    cmreport = options.get('cmreport', None)
    savefigs = options.get('savefigs', None)
    filetype = options.get('filetype', "pdf")
    dpi = options.get('dpi', 600)

    print('Training Report')
    if bypass:
        ptrain = xtrain
    else:
        ptrain = model.predict(xtrain)

    fig1 = ClassificationReport(ytrain, ptrain, plotcm=True, show=True,
                                adj_left=a, adj_bottom=b, figsize=figsize, axes_size=axes_size,
                                cmap=cmap, fontsize=fontsize, xrot=xrot, file=f'{cmreport}_training')
    print(f'{cmreport}_training')

    print('Testing Report')
    if bypass:
        ptest = xtest
    else:
        ptest = model.predict(xtest)
    fig2 = ClassificationReport(ytest, ptest, plotcm=True, show=True,
                                adj_left=a, adj_bottom=b, figsize=figsize, axes_size=axes_size,
                                cmap=cmap, fontsize=fontsize, xrot=xrot, file=f'{cmreport}_testing')

    print('All Report')
    x = np.concatenate((xtrain, xtest))
    y = np.concatenate((ytrain, ytest))
    if bypass:
        p = x
    else:
        p = model.predict(x)
    fig3 = ClassificationReport(y, p, plotcm=True, show=True,
                                adj_left=a, adj_bottom=b, figsize=figsize, axes_size=axes_size,
                                cmap=cmap, fontsize=fontsize, xrot=xrot, file=f'{cmreport}_all')

    if scoring is None:
        from sklearn.metrics import accuracy_score
        scoring = accuracy_score

    a = scoring(ytrain, ptrain)
    b = scoring(ytest, ptest)
    c = scoring(y, p)
    df = pd.DataFrame({'Data': ['Train', 'Test', 'All'], 'Value': [a, b, c]})
    fig4, ax = customplot()
    ax.bar(df['Data'].values, df['Value'].values)
    ax.set_ylim(np.min(df['Value'].values) - 0.1, 1.0)

    if savefigs is not None:
        saveimg(fig1, file=f'{savefigs}_training', filetype=filetype, dpi=dpi)
        saveimg(fig2, file=f'{savefigs}_testing', filetype=filetype, dpi=dpi)
        saveimg(fig3, file=f'{savefigs}_all', filetype=filetype, dpi=dpi)
        saveimg(fig4, file=f'{savefigs}_barplot', filetype=filetype, dpi=dpi)

    return fig1, fig2, fig3, fig4


def timenow():
    import time
    return time.asctime(time.localtime(time.time()))


# noinspection PyPep8Naming
def Logging(logPath, fileName):
    """
    logging function
    :param logPath:
    :param fileName:
    :return:
    """
    import logging

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger


def cmetrics(real_values, pred_values, beta=0.4):
    CM = confusion_matrix(real_values, pred_values)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Population = TN + FN + TP + FP
    Prevalence = round((TP + FP) / Population, 2)
    Accuracy = round((TP + TN) / Population, 4)
    Precision = round(TP / (TP + FP), 4)
    NPV = round(TN / (TN + FN), 4)
    FDR = round(FP / (TP + FP), 4)
    FOR = round(FN / (TN + FN), 4)
    check_Pos = Precision + FDR
    check_Neg = NPV + FOR
    Recall = round(TP / (TP + FN), 4)
    FPR = round(FP / (TN + FP), 4)
    FNR = round(FN / (TP + FN), 4)
    TNR = round(TN / (TN + FP), 4)
    check_Pos2 = Recall + FNR
    check_Neg2 = FPR + TNR
    LRPos = round(Recall / FPR, 4)
    LRNeg = round(FNR / TNR, 4)
    DOR = round(LRPos / LRNeg)
    F1 = round(2 * ((Precision * Recall) / (Precision + Recall)), 4)
    FBeta = round((1 + beta ** 2) * ((Precision * Recall) / ((beta ** 2 * Precision) + Recall)), 4)
    MCC = round(((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 4)
    BM = Recall + TNR - 1
    MK = Precision + NPV - 1
    mat_met = pd.DataFrame({
        'Metric': ['TP', 'TN', 'FP', 'FN', 'Prevalence', 'Accuracy', 'Precision', 'NPV', 'FDR', 'FOR', 'check_Pos',
                   'check_Neg', 'Recall', 'FPR', 'FNR', 'TNR', 'check_Pos2', 'check_Neg2', 'LR+', 'LR-', 'DOR', 'F1',
                   'FBeta', 'MCC', 'BM', 'MK'],
        'Value': [TP, TN, FP, FN, Prevalence, Accuracy, Precision, NPV, FDR, FOR, check_Pos, check_Neg, Recall, FPR,
                  FNR, TNR, check_Pos2, check_Neg2, LRPos, LRNeg, DOR, F1, FBeta, MCC, BM, MK]})
    return mat_met
