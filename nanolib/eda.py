# -*- coding: utf-8 -*-
"""
This file is part of NANOLIB


NANOLIB was primarily developed at Nanosense by:
    Shidiq Nur Hidayat (s.hidayat@nanosense-id.com)

Created on Tue Jul 14 18:23:51 2020

@author: Shidiq Nur Hidayat
"""

import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from nanolib.utils import customplot, train_test_split


class Stats:
    """
    ===============================
    Data Manipulation and statistic
    ===============================

    Stat(df, feature=int, key=string)

    Methods:

    - print_()
    - stats_()
    - corr_()
    - boxplot_()
    - matrixplot_()
    - swarmplot_()
    - stripplot_()
    - corrplot_(adj_left, adj_bottom, size)
    - dataframe_()
    - xy_()
    - xy_encoder(config=None)
    - xy_binarize()
    - signalmean(rot, adj_left, adj_bottom, scale)
    - signalminmaxmean()
    - saveinfo(file)
    - xy_split(encoder=False, binarized=False, test_size=0.2, random_state=None, config=None)
    - hierarchical_clustering(max_d)

    """

    def __init__(self, df, feature=2, key='label'):
        self.feature = feature
        self.key = key
        featurename = list(df)[0:self.feature]
        featurename.append(self.key)
        self.df = df[featurename]
        self.df = self.df.sort_values(by=[self.key])
        self.X = self.df.values[:, 0:self.feature].astype(float)
        self.y = self.df[self.key].values
        self.case = None

    def __repr__(self):
        return 'You must understand your data in order to get the best results.\n'

    def print_(self):
        # Dimensions of Your Data
        print('Understand Your Data\n\n')
        print(f'Dimensions of Your Data\n'
              f'Number of rows    : {self.df.shape[0]}\n'
              f'Number of columns : {self.df.shape[1]}\n\n')

        # Data Type For Each Attribute
        print(f'Data Type For Each Attribute\n'
              f'{self.df.dtypes}\n\n')

        # Descriptive Statistics
        print(f'Descriptive Statistics\n'
              f'{self.df.describe()}\n\n')

        # Class Distribution
        print(f'Class Distribution\n'
              f'{self.df.groupby(self.key).size()}\n\n')

        # Correlations Between Attributes
        pd.set_option('display.width', 100)
        pd.set_option('precision', 3)
        correlations = self.df.corr(method='pearson')
        print(f'Correlations Between Attributes\n'
              f'{correlations}\n\n')

        # Skew of Univariate Distributions
        print(f'Skew of Univariate Distributions\n'
              f'{self.df.skew()}\n\n')

    def stats_(self):
        return self.df.describe()

    def corr_(self):
        return self.df.corr(method='pearson')

    def boxplot_(self, separate=False, adj_left=.1, adj_bottom=.1):
        if separate:
            params = {'font.family': 'serif',
                      'font.serif': 'DejaVu Serif',
                      'xtick.labelsize': 20,
                      'ytick.labelsize': 20,
                      'axes.labelsize': 28,
                      'figure.figsize': [10.72, 8.205],
                      'legend.loc': 'best',
                      'legend.fontsize': 18,
                      'legend.fancybox': False}
            matplotlib.rcParams.update(params)
            self.df.groupby(self.key).boxplot()
        else:
            customplot(adj_bottom=adj_bottom, adj_left=adj_left)
            dd = pd.melt(self.df, id_vars=[self.key], value_vars=list(self.df)[0:self.feature], var_name='Features')
            ax = sns.boxplot(x=self.key, y='value', data=dd, hue='Features')
            return ax.get_figure()

    def matrixplot_(self, adj_left=.1, adj_bottom=.1):
        fig, _ = customplot(adj_bottom=adj_bottom, adj_left=adj_left)
        matplotlib.pyplot.close()
        fig = sns.pairplot(self.df, hue=self.key)
        return fig

    def swarmplot_(self, adj_left=.1, adj_bottom=.1):
        customplot(adj_bottom=adj_bottom, adj_left=adj_left)
        dd = pd.melt(self.df, [self.key], var_name='Features')
        ax = sns.swarmplot(x='Features', y='value', data=dd, hue=self.key)
        return ax.get_figure()

    def stripplot_(self, adj_left=.1, adj_bottom=.1):
        dd = pd.melt(self.df, [self.key], var_name='Features')
        customplot(adj_bottom=adj_bottom, adj_left=adj_left)
        sns.stripplot(x="value", y="Features", hue=self.key,
                      data=dd, dodge=True, jitter=True,
                      alpha=.25, zorder=1)
        ax = sns.pointplot(x="value", y="Features", hue=self.key,
                           data=dd, dodge=.532, join=False, palette="dark",
                           markers="d", scale=.75, ci=None)
        handles, labels = ax.get_legend_handles_labels()
        n = len(np.unique(self.y))
        ax.legend(handles[0:n], labels[0:n], loc='best',
                  handletextpad=0, columnspacing=1,
                  frameon=True)
        return ax.get_figure()

    def corrplot_(self, adj_left=.1, adj_bottom=.1, size=20):
        corr_ = self.df.corr(method='pearson')
        customplot(adj_bottom=adj_bottom, adj_left=adj_left)
        ax = sns.heatmap(corr_, vmax=1, vmin=-1, cmap='YlGnBu', annot=True, annot_kws={"size": size})
        return ax.get_figure()

    def dataframe_(self):
        return self.df

    def xy_(self):
        return self.X, self.y

    def xy_encoder(self, config=None):
        from sklearn.preprocessing import LabelEncoder
        
        if config is None:
            le = LabelEncoder()
            le.fit(list(self.df[self.key].unique()))
            y = le.transform(self.y)
            print(self.df[self.key].unique())
            print(np.unique(y))
        else:
            from sklearn.utils import column_or_1d
            
            class MyLabelEncoder(LabelEncoder):
                
                def fit(self, y):
                    y = column_or_1d(y, warn=True)
                    self.classes_ = pd.Series(y).unique()
                    return self
            
            le = MyLabelEncoder()
            le.fit(config)
            y = le.transform(self.y)
            warnings.warn('User encoder activated')
            
        return self.X, y

    def xy_binarize(self):
        from sklearn.preprocessing import label_binarize
        y = label_binarize(self.y, classes=list(np.unique(self.y)))
        print(self.df[self.key].unique())
        return self.X, y

    def signalmean(self, rot=90, adj_left=0.14, adj_bottom=0.24, scale=False):
        label = self.df[self.key].unique()
        nama = list(self.df)[0:self.feature]
        mean_df = pd.DataFrame()
        scaling = StandardScaler()
        scaling.fit(self.df.values[:, 0:self.feature].astype(float))

        for i, item in enumerate(label):
            ind = self.df[self.key].isin([item])
            temp = self.df[ind]
            temp = temp[nama]
            if scale:
                temp[nama] = scaling.transform(temp[nama])
            temp = temp.apply(np.mean, axis=0)
            temp = pd.DataFrame(data=temp, columns=[item]).transpose()
            mean_df = mean_df.append(temp)

        mean_df = mean_df.transpose()
        params = {'font.family': 'serif',
                  'font.serif': 'DejaVu Serif',
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'axes.labelsize': 28,
                  'figure.figsize': [10.72, 8.205],
                  'legend.loc': 'best',
                  'legend.fontsize': 18,
                  'legend.fancybox': False}
        matplotlib.rcParams.update(params)
        ax = mean_df[label].plot(kind='bar', legend=True, rot=rot)
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        plt.subplots_adjust(left=adj_left, bottom=adj_bottom, right=.97, top=.97)

        return mean_df, ax.get_figure()

    def signalminmaxmean(self):
        label = self.df[self.key].unique()
        nama = list(self.df)[0:self.feature]
        minDF = pd.DataFrame()
        maxDF = pd.DataFrame()
        meanDF = pd.DataFrame()

        for i, item in enumerate(label):
            ind = self.df[self.key].isin([item])
            temp = self.df[ind]
            temp = temp[nama]
            temp = temp.apply(np.min, axis=0)
            temp = pd.DataFrame(data=temp, columns=[item]).transpose()
            minDF = minDF.append(temp)

            temp = self.df[ind]
            temp = temp[nama]
            temp = temp.apply(np.max, axis=0)
            temp = pd.DataFrame(data=temp, columns=[item]).transpose()
            maxDF = maxDF.append(temp)

            temp = self.df[ind]
            temp = temp[nama]
            temp = temp.apply(np.mean, axis=0)
            temp = pd.DataFrame(data=temp, columns=[item]).transpose()
            meanDF = meanDF.append(temp)

        minDF = minDF.transpose()
        maxDF = maxDF.transpose()
        meanDF = meanDF.transpose()

        return minDF, meanDF, maxDF

    def saveinfo(self, file='info.txt'):
        import io
        from contextlib import redirect_stdout
        import os

        path_ = os.path.dirname(file)
        if not os.path.exists(path_):
            os.makedirs(path_)

        f = io.StringIO()

        with open(file, 'w') as f:
            with redirect_stdout(f):
                self.print_()
        return f'Information saved in {file}'

    def xy_split(self, encoder=False, binarized=False, test_size=0.2, random_state=None, config=None):
        if encoder:
            x, y = self.xy_encoder(config=config)
        elif binarized:
            x, y = self.xy_binarize()
        else:
            x, y = self.xy_()

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state=random_state)
        
        return xtrain, xtest, ytrain, ytest

    def hierarchical_clustering(self, max_d=0):
        from scipy.cluster.hierarchy import dendrogram, linkage
        X = self.X
        linked = linkage(X, 'ward')
        labelList = np.arange(0, X.shape[0])
        fig, ax = customplot()
        dendrogram(linked,
                   truncate_mode='lastp',
                   p=X.shape[0],
                   orientation='top',
                   labels=labelList,
                   distance_sort='descending',
                   show_leaf_counts=True)
        plt.axhline(y=max_d, c='k')
        return ax.get_figure()


class KennardStone:
    """
    =============================
    Kennard-Stone Algorithm Class
    =============================

    KennardStone(k)

    Methods:
    - fit(x, y)
    - datasplit()
    - print()

    """

    def __init__(self, **options):
        self.X = None
        self.y = None
        self.k = options.get('k', 0)
        self.X_train = list()
        self.X_val = list()
        self.y_train = list()
        self.y_val = list()
        self.selectedsample = list()
        self.remainingsample = list()

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def fit(self, x, y):
        self.X = x
        self.y = y
        x_variables = self.X
        k = self.k
        x_variables = np.array(x_variables)
        original_x = x_variables
        distance_to_average = ((x_variables - np.tile(x_variables.mean(axis=0), (x_variables.shape[0], 1))) ** 2).sum(
            axis=1)
        max_distance_sample_number = np.where(distance_to_average == np.max(distance_to_average))
        max_distance_sample_number = max_distance_sample_number[0][0]
        selected_sample_numbers = list()
        selected_sample_numbers.append(max_distance_sample_number)
        remaining_sample_numbers = np.arange(0, x_variables.shape[0], 1)
        x_variables = np.delete(x_variables, selected_sample_numbers, 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, 0)
        for iteration in range(1, k):
            selected_samples = original_x[selected_sample_numbers, :]
            min_distance_to_selected_samples = list()
            for min_distance_calculation_number in range(0, x_variables.shape[0]):
                distance_to_selected_samples = (
                        (selected_samples - np.tile(x_variables[min_distance_calculation_number, :],
                                                    (selected_samples.shape[0], 1))) ** 2).sum(axis=1)
                min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
            max_distance_sample_number = np.where(
                min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
            max_distance_sample_number = max_distance_sample_number[0][0]
            selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
            x_variables = np.delete(x_variables, max_distance_sample_number, 0)
            remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)

        self.selectedsample = selected_sample_numbers
        self.remainingsample = remaining_sample_numbers
        return self.selectedsample, self.remainingsample

    def datasplit(self):
        self.X_train = self.X[self.remainingsample, :]
        self.y_train = self.y[self.remainingsample]

        self.X_val = self.X[self.selectedsample, :]
        self.y_val = self.y[self.selectedsample]

        dist_y_train = pd.DataFrame(self.y_train, columns=['Training'])['Training'].value_counts()
        dist_y_val = pd.DataFrame(self.y_val, columns=['Validation'])['Validation'].value_counts()
        print(dist_y_train)
        print(dist_y_val)

        return self.X_train, self.X_val, self.y_train, self.y_val

    def print(self):
        dist_y_train = pd.DataFrame(self.y_train, columns=['Training'])['Training'].value_counts()
        dist_y_val = pd.DataFrame(self.y_val, columns=['Validation'])['Validation'].value_counts()
        print(dist_y_train)
        print(dist_y_val)


class StatsMethod:
    """
    =========================================================
    Statistic Hypothesis testing and Five-Number of statistic
    =========================================================

    StatsMethod()

    Methods:

    - nonparamsignificance(x1, x2)
    
    """

    def __init__(self):
        self.d1 = None
        self.d2 = None

    def nonparamsignificance(self, *arrays):
        from scipy.stats import mannwhitneyu, kruskal, wilcoxon
        import pingouin as pg

        self.d1 = arrays[0]
        self.d2 = arrays[1]
        metode = []
        p = []
        stat = []
        ket = []

        def keterangan(pval):
            alpha = 0.05
            if pval > alpha:
                return 'Same distribution (fail to reject H0)'
            else:
                return 'Different distribution (reject H0)'

        metode.append('Mann-Whitney U test')
        a, b = mannwhitneyu(self.d1, self.d2)
        stat.append(a)
        p.append(b)
        ket.append(keterangan(b))

        metode.append('Kruskal-Wallis H Test')
        a, b = kruskal(self.d1, self.d2)
        stat.append(a)
        p.append(b)
        ket.append(keterangan(b))

        metode.append('Wilcoxon')
        a, b = wilcoxon(self.d1, self.d2, correction=True)
        stat.append(a)
        p.append(b)
        ket.append(keterangan(b))

        results = {
            'Method': metode,
            'Statistic': stat,
            'p-value': p,
            'Conclusion': ket,
        }
        results = pd.DataFrame(results)
        print('5-Number of statistic for D1:')
        self.fivenumberplus(self.d1)

        print('5-Number of statistic for D2:')
        self.fivenumberplus(self.d2)

        print(results)
        print(pg.wilcoxon(self.d1, self.d2, tail='two-sided'))
        return results

    @staticmethod
    def fivenumberplus(x):
        from numpy import percentile
        Q = percentile(x, [25, 50, 75])
        print('Min   : %.3f' % x.min())
        print('Q1    : %.3f' % Q[0])
        print('Median: %.3f' % Q[1])
        print('Q3    : %.3f' % Q[2])
        print('Max   : %.3f' % x.max())
        print('Mean  : %.3f' % x.mean())

    @staticmethod
    def multisignificancetest(A1, A2):

        def func(d1, d2, alpha=0.05):
            from scipy.stats import kruskal
            _, p = kruskal(d1, d2)
            if p > alpha:
                # same distribution (fail to reject H0)
                return 0
            else:
                # different distribution (reject H0 = H1)
                return 1

        m, n = A1.shape

        result = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                temp = func(A1[:, i], A2[:, j])
                result[i, j] = temp

        label = [f'F{i}' for i in range(n)]
        result = pd.DataFrame(data=result, index=label, columns=label)

        plt.figure(figsize=(7, 7))
        sns.set(font_scale=2.0)
        ax = sns.heatmap(result, annot=True, cbar=False, annot_kws={'size': 20})

        return result, ax.get_figure()
