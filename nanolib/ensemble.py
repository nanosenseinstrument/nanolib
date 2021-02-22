# -*- coding: utf-8 -*-
"""
This file is part of NANOLIB


NANOLIB was primarily developed at Nanosense by:
    Shidiq Nur Hidayat (s.hidayat@nanosense-id.com)

Created on Tue Jul 14 18:23:51 2020

@author: Shidiq Nur Hidayat
"""

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from numpy import mean, std, nan, array, sum, argmax
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Voting:
    def __init__(self, models, voting='hard', cv=10, case='classifier', 
                 weights=None, scoring=None, error_score=nan):
        self.models = models
        self.voting = voting
        self.cv = cv
        self.case = case
        self.weights = weights
        self.scoring = scoring
        self.errorscore = error_score
        
    def get_voting(self):
        if self.case == 'classifier':
            ensemble = VotingClassifier(estimators=self.models, 
                                        voting=self.voting,
                                        weights=self.weights)
        else:
            ensemble = VotingRegressor(estimators=self.models,
                                        weights=self.weights)
        return ensemble
    
    def get_models(self):
        models = self.Convert(self.models)
        models['voting'] = self.get_voting()
        return models
    
    def evaluate_models(self, models, X, y):
        score = cross_val_score(models, X, y, cv=self.cv, n_jobs=-1,
                                scoring=self.scoring, 
                                error_score=self.errorscore)
        return score
    
    def evaluate(self, X, y):
        results, names = list(), list()
        models = self.get_models()
        for name, model in models.items():
            score = self.evaluate_models(model, X, y)
            results.append(score)
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, mean(score), std(score)))
        
        plt.boxplot(results, labels=names, showmeans=True)
        plt.show()
    
    @staticmethod
    def Convert(lst): 
        res_dct = {lst[i][0]: lst[i][1] for i in range(len(lst))} 
        return res_dct 
    

class DNNClassifier:
    def __init__(self, model, case='average', n_members=10, epochs=500,
                 verbose=0, loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy']):
        self.basemodel = model
        self.case = case
        self.n_members = n_members
        self.epochs = epochs
        self.verbose = verbose
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
    
    @staticmethod
    def fit_model(model, xtrain, ytrain, loss, optimizer, metrics, epochs, verbose):
        from tensorflow.keras.utils import to_categorical
        
        ytrain_enc = to_categorical(ytrain)
        model.compile(loss=loss, optimizer=optimizer, 
                      metrics=metrics)
        model.fit(xtrain, ytrain_enc, epochs=epochs, verbose=verbose)
        return model
        
    @staticmethod
    def ensemble_predictions(members, xtest):
        yhats = [model.predict(xtest) for model in members]
        yhats = array(yhats)
        summed = sum(yhats, axis=0)
        results = argmax(summed, axis=1)
        return results
    
    def evaluate_n_members(self, members, n_members, xtest, ytest):
        subset = members[:n_members]
        yhat = self.ensemble_predictions(subset, xtest)
        return accuracy_score(ytest, yhat)
    
    def evaluate_members(self, xtrain, ytrain, xtest, ytest):
        from tensorflow.keras.utils import to_categorical

        members = [self.fit_model(model=self.basemodel, xtrain=xtrain, ytrain=ytrain, loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, epochs=self.epochs, verbose=self.verbose) for _ in range(self.n_members)]
        
        single_scores, ensemble_scores = list(), list()
        
        for i in range(1, len(members)+1):
            ensemble_score = self.evaluate_n_members(members, i, xtest, ytest)
            ytest_enc = to_categorical(ytest)
            _, single_score = members[i-1].evaluate(xtest, ytest_enc, verbose=self.verbose)
            print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
            ensemble_scores.append(ensemble_score)
            single_scores.append(single_score)
        
        print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
        
        # plot score vs number of ensemble members
        x_axis = [i for i in range(1, len(members)+1)]
        plt.plot(x_axis, single_scores, marker='o', linestyle='None')
        plt.plot(x_axis, ensemble_scores, marker='o')
        plt.show()