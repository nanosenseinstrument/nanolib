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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nanolib.utils import TooMuchUnique, customplot


class CalcPCA:
    """
    ============================
    Principal component analysis
    ============================

    CalcPCA(round_, featurename, scaler)

    Methods:

    - fit(x, y)
    - getvarpc()
    - getcomponents()
    - getbestfeature()
    - plotpc(PC, adj_left, adj_bottom, acending)
    - screenplot(adj_left, adj_Bottom)

    """

    def __init__(self, **options):        
        self.x = None
        self.y = None
        self.vardf = pd.DataFrame()
        self.pcadf = pd.DataFrame()
        self.eigpc = pd.DataFrame()
        self.round_ = options.get('round_', 1)
        self.featurename = options.get('featurename', None)
        self.scaler = options.get('scaler', StandardScaler())
        self.pca = PCA()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.pca!r})'
        )

    def fit(self, x, y):
        self.x = x
        self.y = y

        if self.scaler is not None:
            scaler = self.scaler
            self.x = scaler.fit_transform(self.x)

        self.pca = PCA()
        self.pca.fit(self.x)
        pcscore = self.pca.transform(self.x)
        pcname = [f'PC{i + 1}' for i in range(pcscore.shape[1])]
        # pcname = [f'PC{i + 1}' for i in range(self.x.shape[1])]
        if self.featurename is None:
            self.featurename = [f'Feature{i + 1}' for i in range(self.x.shape[1])]
        # var_exp = [round(i * 100, self.round_) for i in sorted(self.pca.explained_variance_ratio_, reverse=True)]
        var_exp = np.round(self.pca.explained_variance_ratio_ * 100, decimals=self.round_)
        self.vardf = pd.DataFrame({'Var (%)': var_exp, 'PC': pcname})
        # pcscore = self.pca.transform(self.x)
        pcaDF = pd.DataFrame(data=pcscore, columns=pcname)
        Y = pd.DataFrame(data=self.y, columns=['label'])
        self.pcadf = pd.concat([pcaDF, Y], axis=1)
        self.eigpc = pd.DataFrame(data=np.transpose(self.pca.components_),
                                  columns=pcname,
                                  index=self.featurename)
        return self.pca

    def getvarpc(self):
        return self.pcadf, self.vardf, self.eigpc

    def getcomponents(self):
        loading_score = pd.DataFrame(data=self.pca.components_, columns=[self.featurename])
        return loading_score

    def getbestfeature(self, PC=0, n=3):
        loading_score = pd.Series(self.pca.components_[PC], index=self.featurename)
        sorted_loading_score = loading_score.abs().sort_values(ascending=False)
        top_score = sorted_loading_score[0:n].index.values
        print(loading_score[top_score])

    def plotpc(self, **options):
        PC = options.get('PC', ['PC1', 'PC2'])
        a = options.get('adj_left', 0.1)
        b = options.get('adj_bottom', 0.15)
        ascending = options.get('ascending', True)
        self.pcadf = self.pcadf.sort_values(by=['label'], ascending=ascending)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'C0', 'C1', 'C2']
        markers = ["o", "v", "s", "p", "P", "*", "h", "H", "X", "D"]
        targets = list(self.pcadf['label'].unique())

        if len(targets) > 10:
            raise TooMuchUnique(str(targets))

        colors = colors[:len(targets)]
        markers = markers[:len(targets)]

        xlabs = f'{PC[0]} ({float(self.vardf.values[self.vardf["PC"] == PC[0], 0])}%)'
        ylabs = f'{PC[1]} ({float(self.vardf.values[self.vardf["PC"] == PC[1], 0])}%)'

        fig, ax = customplot(adj_left=a, adj_bottom=b)
        for target, color, mark in zip(targets, colors, markers):
            indicesToKeep = self.pcadf['label'] == target
            ax.scatter(self.pcadf.loc[indicesToKeep, PC[0]],
                       self.pcadf.loc[indicesToKeep, PC[1]],
                       c=color,
                       marker=mark,
                       s=50,
                       )

        plt.xlabel(xlabs)
        plt.ylabel(ylabs)
        plt.legend(targets)
        return fig

    def screenplot(self, **options):
        a = options.get('adj_left', 0.1)
        b = options.get('adj_bottom', 0.2)
        fig, _ = customplot(adj_bottom=b, adj_left=a)
        plt.bar(x='PC', height='Var (%)', data=self.vardf)
        plt.xticks(rotation='vertical')
        plt.xlabel('Principal Component')
        plt.ylabel('Percentage of Variance')
        return fig
    

class CalcLDA:
    """
    ============================
    Linear discriminant analysis
    ============================

    CalcLDA(round_, scaler, cv)

    Methods:

    - fit(x, y)
    - getvarkd()
    - getscore()
    - plotlda(adj_left, adj_bottom, acending)

    """

    def __init__(self, **options):
        self.x = None
        self.xval = None
        self.y = None
        self.yval = None
        self.ldaval = None
        self.dual = None

        self.round_ = options.get('round_', 1)
        self.vardf = pd.DataFrame()
        self.ldadf = pd.DataFrame()
        self.lda = LinearDiscriminantAnalysis()
        self.scaler = options.get('scaler', StandardScaler())
        self.cv = options.get('cv', 10)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.lda!r})'
        )

    def fit(self, *arrays):
        if len(arrays) == 2:
            self.x = arrays[0]
            self.y = arrays[1]
            self.dual = False
        else:
            self.x = arrays[0]
            self.xval = arrays[1]
            self.y = arrays[2]
            self.yval = arrays[3]
            self.ldaval = None
            self.dual = True

        scaler = self.scaler
        X = scaler.fit_transform(self.x)
        self.lda.fit(X, self.y)
        ldax = self.lda.transform(X)
        ldname = [f'LD{i + 1}' for i in range(ldax.shape[1])]
        self.ldadf = pd.DataFrame(ldax, columns=ldname)
        Y = pd.DataFrame(data=self.y, columns=['label'])
        self.ldadf = pd.concat([self.ldadf, Y], axis=1)

        tot = sum(self.lda.explained_variance_ratio_)
        var_exp = [round((i / tot) * 100, self.round_) for i in sorted(self.lda.explained_variance_ratio_,
                                                                       reverse=True)]
        self.vardf = pd.DataFrame({'Var (%)': var_exp, 'LD': ldname})

        if self.dual:
            Xval = scaler.transform(self.xval)
            ldax = self.lda.transform(Xval)
            self.ldaval = pd.DataFrame(ldax, columns=ldname)
            Y = pd.DataFrame(data=self.yval, columns=['label'])
            self.ldaval = pd.concat([self.ldaval, Y], axis=1)

    def getvarld(self):
        if self.dual:
            ldaDF1 = pd.concat([
                self.ldadf,
                pd.DataFrame(data=self.ldadf['label'].values, columns=['Class']),
            ], axis=1)
            ldaDF1['Class'] = "Training"

            ldaDF2 = pd.concat([
                self.ldaval,
                pd.DataFrame(data=self.ldaval['label'].values, columns=['Class']),
            ], axis=1)
            ldaDF2['Class'] = "Testing"

            ldaDF = pd.concat([ldaDF1, ldaDF2], axis=0)
        else:
            ldaDF = self.ldadf

        return ldaDF, self.vardf

    def getscore(self):
        from sklearn.model_selection import cross_val_score
        return cross_val_score(LinearDiscriminantAnalysis(), self.x, self.y, cv=self.cv)

    def plotlda(self, **options):
        import seaborn as sns
        
        a = options.get('adj_left', 0.1)
        b = options.get('adj_bottom', 0.15)
        ascending = options.get('ascending', True)

        self.ldadf = self.ldadf.sort_values(by=['label'], ascending=ascending)
        nlabel = np.unique(self.y)
        if len(nlabel) < 3:
            fig, _ = customplot(adj_left=a, adj_bottom=b)
            s = options.get('size', 10)

            if self.dual:
                self.ldaval = self.ldaval.sort_values(by=['label'], ascending=ascending)
                sns.stripplot(x="label", y="LD1", color='k', size=s, data=self.ldadf)
                sns.stripplot(x="label", y="LD1", marker='^', color='red', size=s, data=self.ldaval)
            else:
                sns.stripplot(x="label", y="LD1", size=s, data=self.ldadf)

            plt.xlabel('Classes')
            plt.axhline(y=0, linewidth=1.5, color='black', linestyle='--')
            return fig
        else:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'C0', 'C1', 'C2']
            markers = ["o", "v", "s", "p", "P", "*", "h", "H", "X", "D"]
            targets = list(self.ldadf['label'].unique())
            s = options.get('size', 90)
            if len(targets) > 10:
                raise TooMuchUnique(str(targets))

            colors = colors[:len(targets)]
            markers = markers[:len(targets)]

            xlabs = f'LD1 ({self.vardf.values[0, 0]}%)'
            ylabs = f'LD2 ({self.vardf.values[1, 0]}%)'
            fig, ax = customplot(adj_left=a, adj_bottom=b)
            for target, color, mark in zip(targets, colors, markers):
                indicesToKeep = self.ldadf['label'] == target
                ax.scatter(self.ldadf.loc[indicesToKeep, 'LD1'],
                           self.ldadf.loc[indicesToKeep, 'LD2'],
                           c=color,
                           marker=mark,
                           s=s,
                           )
            if self.dual:
                # plot tidak urut, tambahkan line ini: (14/05)
                self.ldaval = self.ldaval.sort_values(by=['label'], ascending=ascending)
                for target, color, mark in zip(targets, colors, markers):
                    indicesToKeep = self.ldaval['label'] == target
                    ax.scatter(self.ldaval.loc[indicesToKeep, 'LD1'],
                               self.ldaval.loc[indicesToKeep, 'LD2'],
                               # c=color,
                               marker=mark,
                               s=s,
                               facecolors='none',
                               edgecolors=color,
                               )
            plt.legend(targets)
            plt.xlabel(xlabs)
            plt.ylabel(ylabs)

            return fig