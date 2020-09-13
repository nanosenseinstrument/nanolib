# -*- coding: utf-8 -*-
"""
This file is part of NANOLIB


NANOLIB was primarily developed at Nanosense by:
    Shidiq Nur Hidayat (s.hidayat@nanosense-id.com)

Created on Tue Jul 14 18:23:51 2020

@author: Shidiq Nur Hidayat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from torclib.utils import customplot
from pycm import ConfusionMatrix


class CalcROC:
    """
    ===================================
    Receiver operational curve analysis
    ===================================

    CalcROC(colors, scaler)

    Methods:

    - fit(x, y)
    - fig_()
    - results_()
    
    """

    def __init__(self, estimator, **options):
        self.colors = options.get('colors', None)
        self.estimator = estimator
        self.fig = []
        self.fpr = []
        self.tpr = []
        self.roc_auc = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = options.get('scaler', StandardScaler())

    def fit(self, *arrays):
        from sklearn.multiclass import OneVsRestClassifier
        from torclib.utils import plotroc

        self.X_train = arrays[0]
        self.X_test = arrays[1]
        self.y_train = arrays[2]
        self.y_test = arrays[3]

        clf = OneVsRestClassifier(self.estimator)
        
        XTrain = self.scaler.fit_transform(self.X_train)
        XTest = self.scaler.transform(self.X_test)

        y_score = clf.fit(XTrain, self.y_train).decision_function(XTest)
        self.fig, res = plotroc(self.y_test, y_score, colors=self.colors)
        self.fpr, self.tpr, self.roc_auc = res

    def fig_(self):
        return self.fig

    def results_(self):
        return self.fpr, self.tpr, self.roc_auc

    def __repr__(self):
        return f'ROC Analysis of {self.estimator!r} \nwith return (fig, fpr, tpr, roc_auc)'


class Train:
    """
    ==========================================================
    Training procedure for classification and regression model
    ==========================================================

    Train(random_state, cv, mode, colors, scaler, multiclassroc, verbose)

    Methods:

    - fit(x, y)
    - evaluate(x, y)
    - predict(x)
    - classification_report(external=bool, intercv=bool)
    - scores_()
    - estimator_()
    - plotcm(external, intercv, print_stats, adj_left, adj_bottom, fromatplot, title)
    - save_stats(file)
    - pointplot_(adj_left, adj_bottom, cross_val_predict)
    - results()
    - plotregression_(adj_left, adj_bottom_cross_val_predict)

    """

    def __init__(self, estimator, **options):
        from sklearn.model_selection import StratifiedKFold
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.intercv = None
        self.estimator = estimator
        self.random_state = options.get('random_state', 99)
        self.cv = options.get('cv', StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state))
        self.mode = options.get('mode', 'classification')
        self.colors = options.get('colors', None)
        self.scaler = options.get('scaler', StandardScaler())
        self.multiroc = options.get('multiclassroc', True)
        self.verbose = options.get('verbose', 2)
        self.compact = options.get('compact', True)
        self.internal = list()
        self.external = list()
        self.training = list()
        self.crossValPredict = list()
        self.clf = list()
        self.CM = list()
        self.predictCVDF = pd.DataFrame()
        self.predictTrainDF = pd.DataFrame()
        self.predictTestDF = pd.DataFrame()

    def __repr__(self):
        if self.compact:
            return f'Scaling  : {self.scaler!r}\nEstimator: {self.estimator!r}\n'
        else:
            return f'Estimator: {self.estimator!r}\n'

    def fit(self, *arrays):
        from sklearn.pipeline import Pipeline
        from torclib.utils import plotroc
        
        if len(arrays) == 2:
            self.X_train = arrays[0]
            self.y_train = arrays[1]
            self.intercv = True
        else:
            self.X_train = arrays[0]
            self.X_test = arrays[1]
            self.y_train = arrays[2]
            self.y_test = arrays[3]
            self.intercv = False

        # classification using accuracy metric
        if self.mode == 'classification':
            if self.compact:
                pipe = Pipeline([('scaler', self.scaler), ('clf', self.estimator)])
            else:
                pipe = self.estimator
                
            self.internal = cross_val_score(pipe, self.X_train, self.y_train, n_jobs=-1,
                                            cv=self.cv, scoring=None)
            self.crossValPredict = cross_val_predict(pipe, self.X_train, self.y_train,
                                                     cv=self.cv, n_jobs=-1)
            self.clf = pipe.fit(self.X_train, self.y_train)
            self.training = self.clf.predict(self.X_train)

            predTrain = self.clf.predict(self.X_train)
            self.predictTrainDF = pd.concat([
                pd.DataFrame(data=self.y_train, columns=['Actual']),
                pd.DataFrame(data=predTrain, columns=['Prediction']),
            ], axis=1)

            self.predictCVDF = pd.concat([
                pd.DataFrame(data=self.y_train, columns=['Actual']),
                pd.DataFrame(data=self.crossValPredict, columns=['Prediction'])
            ], axis=1)

            print('Metric - accuracy_score:')

            internalcv = cross_val_score(pipe, self.X_train, self.y_train, n_jobs=-1, cv=self.cv, scoring=None)

            print(f'Mean of Internal-Validation  : {round(np.mean(internalcv), 3)}')
            print(f'Stdev of Internal-Validation : {round(np.std(internalcv), 3)}')
            print(f'Training score               : {round(accuracy_score(self.y_train, self.training), 3)}\n')
            print('Metric - F1-Score (macro):')
            scoring = make_scorer(f1_score, average='macro')
            internalcv = cross_val_score(pipe, self.X_train, self.y_train, n_jobs=-1, cv=self.cv, scoring=scoring)

            print(f'Mean of Internal-Validation  : {round(np.mean(internalcv), 3)}')
            print(f'Stdev of Internal-Validation : {round(np.std(internalcv), 3)}')
            print(f'Training score               : {round(f1_score(self.y_train, self.training, average="macro"), 3)}\n')

            if not self.intercv:
                self.external = self.clf.predict(self.X_test)

                self.predictTestDF = pd.concat([
                    pd.DataFrame(data=self.y_test, columns=['Actual']),
                    pd.DataFrame(data=self.external, columns=['Prediction'])
                ], axis=1)

                print(f'External-validation accuracy score  : {round(accuracy_score(self.y_test, self.external), 3)}\n')
                print(f'External-validation F1-score (macro): {round(f1_score(self.y_test, self.external, average="macro"), 3)}\n')

        # regression using rsquared metric
        elif self.mode == 'regression':
            if self.compact:
                pipe = Pipeline([('scaler', self.scaler), ('clf', self.estimator)])
            else:
                pipe = self.estimator

            self.crossValPredict = cross_val_predict(pipe, self.X_train, self.y_train,
                                                     cv=self.cv, n_jobs=-1)

            self.predictCVDF = pd.concat([
                pd.DataFrame(data=self.y_train, columns=['Actual']),
                pd.DataFrame(data=self.crossValPredict, columns=['Prediction'])
            ], axis=1)

            self.internal = cross_val_score(pipe, self.X_train, self.y_train, n_jobs=-1,
                                            cv=self.cv, scoring='r2')
            print('Internal-Validation Score')
            print(f'Mean of R2 score        : {np.mean(self.internal)}')
            print(f'Stdev of R2 score       : {np.std(self.internal)}')

            print(f'Mean of adj-R2 score    : {np.mean(self.adj_r2_squared(self.internal, self.X_train, self.y_train))}')
            print(f'Stdev of adj-R2 score   : {np.std(self.adj_r2_squared(self.internal, self.X_train, self.y_train))}')

            self.internal = cross_val_score(pipe, self.X_train, self.y_train, n_jobs=-1,
                                            cv=self.cv, scoring='neg_root_mean_squared_error')
            print(f'Mean of RMSE score      : {np.mean(self.internal)}')
            print(f'Stdev of RMSE score     : {np.std(self.internal)}\n')

            self.internal = cross_val_score(pipe, self.X_train, self.y_train, n_jobs=-1,
                                            cv=self.cv, scoring='neg_mean_absolute_error')
            print(f'Mean of MAE score       : {np.mean(self.internal)}')
            print(f'Stdev of MAE score      : {np.std(self.internal)}\n')

            self.clf = pipe.fit(self.X_train, self.y_train)
            self.training = self.clf.predict(self.X_train)

            pred_training = pd.DataFrame(data=self.training, columns=['Prediction'])
            self.predictTrainDF = pd.concat([pd.DataFrame(data=self.y_train, columns=['Actual']),
                                             pred_training], axis=1)
            print('Training Score           ')
            print(f'R2 score                : {r2_score(self.y_train, self.training)}')
            print(f'adj-R2 score            : {self.adj_r2_squared(r2_score(self.y_train, self.training), self.X_train, self.y_train)}')
            print(f'RMSE score              : {mean_squared_error(self.y_train, self.training)}')
            print(f'MAE score               : {mean_absolute_error(self.y_train, self.training)}\n')

            if not self.intercv:
                self.external = self.clf.predict(self.X_test)

                pred_external = pd.DataFrame(data=self.external, columns=['Prediction'])
                self.predictTestDF = pd.concat([pd.DataFrame(data=self.y_test, columns=['Actual']),
                                                pred_external], axis=1)

                print('External-Validation Score')
                print(f'R2 score                : {r2_score(self.y_test, self.external)}')
                print(f'adj-R2 score            : {self.adj_r2_squared(r2_score(self.y_test, self.external), self.X_test, self.y_test)}')
                print(f'RMSE score              : {mean_squared_error(self.y_test, self.external)}')
                print(f'MAE score               : {mean_absolute_error(self.y_test, self.external)}\n')

        elif self.mode == 'roc':
            from sklearn.multiclass import OneVsRestClassifier
            clf = OneVsRestClassifier(self.estimator)
            
            if self.compact:
                pipe = Pipeline([('scaler', self.scaler), ('clf', clf)])
            else:
                pipe = clf
                
            y_score = clf.fit(self.X_train, self.y_train).decision_function(self.X_test)
            fig, res = plotroc(self.y_test, y_score, colors=self.colors, multiclass=self.multiroc)
            return fig, res
        else:
            return print(f'Your input scoring is {self.scoring} that not fit with one of accuracy, rsquared, and roc')
    
    def predict(self, *array):
        return self.clf.predict(array[0])
    
    def evaluate(self, *arrays):
        self.X_test = arrays[0]
        self.y_test = arrays[1]
        self.external = self.clf.predict(self.X_test)
        
        if self.mode == 'classification':
            self.predictTestDF = pd.concat([
                    pd.DataFrame(data=self.y_test, columns=['Actual']),
                    pd.DataFrame(data=self.external, columns=['Prediction'])
                ], axis=1)

            print(f'External-validation accuracy score  : {accuracy_score(self.y_test, self.external)}\n')
            # print(f'External-validation F1-score (micro): {f1_score(self.y_test, self.external, average="micro")}\n')
            print(f'External-validation F1-score (macro): {f1_score(self.y_test, self.external, average="macro")}\n')
        
        elif self.mode == 'regression':
            pred_external = pd.DataFrame(data=self.external, columns=['Prediction'])
            self.predictTestDF = pd.concat([pd.DataFrame(data=self.y_test, columns=['Actual']),
                                            pred_external], axis=1)

            print('External-Validation Score')
            print(f'R2 score                : {r2_score(self.y_test, self.external)}')
            print(f'adj-R2 score            : {self.adj_r2_squared(r2_score(self.y_test, self.external), self.X_test, self.y_test)}')
            print(f'RMSE score              : {mean_squared_error(self.y_test, self.external)}')
            print(f'MAE score               : {mean_absolute_error(self.y_test, self.external)}\n')

    def scores_(self):
        if self.intercv:
            return self.internal, self.training, self.crossValPredict
        else:
            return self.internal, self.training, self.crossValPredict, self.external

    def estimator_(self):
        return self.clf
    
    def classification_report(self, external=False, intercv=False, file=None):
        from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef
        if self.mode == 'classification':
            if external:
                y_true = self.y_test
                y_pred = self.clf.predict(self.X_test)
                info = 'external-validation'
            else:
                y_true = self.y_train
                if intercv:
                    info = 'internal-validation'
                    y_pred = self.crossValPredict
                else:
                    info = 'training-validation'
                    y_pred = self.clf.predict(self.X_train)
            
            acc = cohen_kappa_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            matt = matthews_corrcoef(y_true, y_pred)
            cr = classification_report(y_true, y_pred, digits=3)
            
            print(f'\nConfusion matrix for {info}:\n{cm}')
            print(f'\nCohen kappa score for {info}: {np.round(acc, 3)}')
            print(f'Matthews correlation coef for {info}: {np.round(matt, 3)}\n')
            print(cr)

            if file != None:
                self.CM = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
                self.CM.save_html(f'{str(file)}-{info}')

    def plotcm(self, external=False, intercv=False, **options):
        from torclib.utils import plot_confusion_matrix
        
        a = options.get('adj_left', 0.1)
        b = options.get('adj_bottom', 0.2)
        ps = options.get('print_stats', False)
        title = options.get('title', False)
        figsize = options.get('figsize', [5, 5])
        axes_size = options.get('axes_size', 22)

        if self.mode == 'classification':
            if external:
                y_true = self.y_test
                y_pred = self.clf.predict(self.X_test)
            else:
                y_true = self.y_train
                if intercv:
                    y_pred = self.crossValPredict
                else:
                    y_pred = self.clf.predict(self.X_train)

            cm_ = confusion_matrix(y_true, y_pred)

            np.set_printoptions(precision=2)
            class_names = np.unique(y_true)

            self.CM = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
            if ps:
                print(self.CM)
            
            fig, _ = customplot(adj_bottom=b, adj_left=a, figsize=figsize, axes_size=axes_size)
            
            if title:
                if external:
                    plt.title('External-Validation')
                else:
                    if intercv:
                        plt.title('Internal-Validation')
                    else:
                        plt.title('Training Results')

            plot_confusion_matrix(cm_, classes=class_names)
            plt.show()
            return fig
        else:
            print('Just for classification')

        # noinspection PyUnresolvedReferences

    def save_stats(self, **kwargs):
        fullpathname = kwargs.get('file', 'cm')
        self.CM.save_html(fullpathname)

    def pointplot_(self, **options):
        import math
        yint = range(min(self.y_train), math.ceil(max(self.y_train)) + 1)
        a = options.get('adj_left', 0.12)
        b = options.get('adj_bottom', 0.12)
        mode = options.get('cross_val_predict', False)

        if mode:
            trainDF = self.predictCVDF
        else:
            trainDF = self.predictTrainDF

        trainDF = trainDF.sort_values(by='Actual')
        if self.intercv:
            fig, ax = customplot(adj_left=a, adj_bottom=b)
            plt.plot(trainDF.Actual.values, '--r')
            plt.plot(trainDF.Prediction.values, '-bo')
            plt.xlabel('Samples')
            plt.ylabel('Class prediction')
            plt.yticks(yint)
            return fig
        else:
            testDF = self.predictTestDF
            testDF = testDF.sort_values(by='Actual')
            fig, ax = customplot(adj_bottom=b, adj_left=a)

            plt.subplot(2, 1, 1)
            plt.plot(trainDF.Actual.values, '--r')
            plt.plot(trainDF.Prediction.values, '-bo')
            plt.ylabel('Train prediction')
            plt.yticks(yint)

            plt.subplot(2, 1, 2)
            plt.plot(testDF.Actual.values, '--r')
            plt.plot(testDF.Prediction.values, '-bo')
            plt.ylabel('Test prediction')
            plt.xlabel('Samples')
            plt.yticks(yint)
            return fig

    def results(self):
        if self.intercv:
            return self.predictTrainDF, self.predictCVDF
        else:
            return self.predictTrainDF, self.predictCVDF, self.predictTestDF

    @staticmethod
    def abline(slope, intercept):
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')
    
    @staticmethod
    def adj_r2_squared(r2, X, y):
        return 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

    def plotregression_(self, **options):
        a = options.get('adj_left', 0.12)
        b = options.get('adj_bottom', 0.12)
        mode = options.get('cross_val_predict', False)

        fig, ax = customplot(adj_left=a, adj_bottom=b)
        if mode:
            ax.scatter(self.predictCVDF.Actual, self.predictCVDF.Prediction,
                       s=70, c='b', marker='o', label='Training')
        else:
            ax.scatter(self.predictTrainDF.Actual, self.predictTrainDF.Prediction,
                       s=70, c='b', marker='o', label='Training')

        if not self.intercv:
            ax.scatter(self.predictTestDF.Actual, self.predictTestDF.Prediction,
                       s=70, c='r', marker='v', label='Testing')
            ax.legend()
        self.abline(1, 0)
        plt.xlabel('Actual')
        plt.ylabel('Prediction')
        return fig


class Battle:
    """
    ===============================================
    Compare PreProcessing and Classification models
    ===============================================

    Battle()

    Methods:

    - fit(x, y)

    """

    def __init__(self):
        self.X = None
        self.y = None
        self.results = pd.DataFrame()

    def fit(self, X, y):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer, MaxAbsScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from itertools import product
        import warnings

        warnings.filterwarnings("ignore")

        self.X = X
        self.y = y

        list_ = {
            'scaler': [StandardScaler(), RobustScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
            'clf': [LogisticRegression(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(),
                    GaussianNB(), SVC()],
        }

        combination = [dict(zip(list_.keys(), v)) for v in product(*list_.values())]

        kCV = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

        listScaler = []
        listClf = []
        listAcc = []
        listStdev = []

        for i in range(len(combination)):
            pipe = Pipeline([('scaler', combination[i]['scaler']), ('clf', combination[i]['clf'])])
            acc = cross_val_score(pipe, self.X, self.y, cv=kCV)
            listScaler.append(str(combination[i]['scaler'].__class__))
            listClf.append(str(combination[i]['clf'].__class__))
            listAcc.append(np.mean(acc))
            listStdev.append(np.std(acc))

        self.results = pd.concat([
            pd.DataFrame(data=listScaler, columns=['Scaler']),
            pd.DataFrame(data=listClf, columns=['Classification']),
            pd.DataFrame(data=listAcc, columns=['Mean of Accuracy']),
            pd.DataFrame(data=listStdev, columns=['Standard Deviation']),
        ], axis=1)