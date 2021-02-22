# -*- coding: utf-8 -*-
"""
This file is part of NANOLIB


NANOLIB was primarily developed at Nanosense by:
    Shidiq Nur Hidayat (s.hidayat@nanosense-id.com)

Created on Tue Jul 14 18:23:51 2020

@author: Shidiq Nur Hidayat
"""

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from nanolib.utils import customplot, printprogressbar


class TuneSVM:
    """
    ===================================
    Hyperparameter tuning for SVM model
    ===================================

    Tune SVM(finetune, clfmode, params, scaler, cv, scoring, verbose)

    Methods:

    - fit(x, y)
    - bestparams()
    - bestsvc()
    - plot_validation_curve(ylim)
    - plot_learning()

    """

    def __init__(self, **options):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        
        self.X = None
        self.y = None
        self.finetune = options.get('finetune', None)
        self.classification = options.get('clfmode', True)
        self.params = options.get('params', [
            {
                'svm__kernel': ['linear', 'rbf'],
                'svm__C': np.logspace(-2, 4, 7),
                'svm__gamma': np.logspace(-6, 0, 7),
            }
        ])
        self.scaler = options.get('scaler', StandardScaler())
        self.cv = options.get('cv', StratifiedKFold(n_splits=10, shuffle=True, random_state=99))
        self.scoring = options.get('scoring', None)
        self.verbose = options.get('verbose', 1)
        self.grid = []

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
        )

    def fit(self, x, y):
        from sklearn.svm import SVC, SVR
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        
        self.X = x
        self.y = y

        if self.finetune:
            self.params = [
                {
                    'svm__kernel': ['rbf'],
                    'svm__C': np.logspace(0, 4, 5),
                    'svm__gamma': np.linspace(0.01, 2, 200),
                }
            ]

        if self.classification:
            steps = [
                ('scaler', self.scaler),
                ('svm', SVC()),
            ]
        else:
            steps = [
                ('scaler', self.scaler),
                ('svm', SVR()),
            ]

        pipe = Pipeline(steps)
        self.grid = GridSearchCV(pipe, self.params, cv=self.cv, n_jobs=-1, scoring=self.scoring,
                                 verbose=self.verbose)
        self.grid.fit(self.X, self.y)

    def bestparams(self):
        return f'Best parameters: {self.grid.best_params_} with score: {round(self.grid.best_score_, 2)}'

    def bestsvc(self):
        return self.grid.best_estimator_

    def plot_validation_curve(self, ylim=None):

        if ylim is None:
            ylim = [0.5, 1.1]

        from sklearn.model_selection import validation_curve

        mode = self.grid.best_params_['svm__kernel']

        if mode == 'linear':
            print('linear')
            param_range = self.params[0].get('svm__C')
            train_scores, test_scores = validation_curve(self.grid.best_estimator_, self.X, self.y,
                                                         param_name='svm__C',
                                                         param_range=param_range,
                                                         scoring='accuracy',
                                                         cv=self.cv,
                                                         n_jobs=-1)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            fig, ax = customplot(adj_bottom=.13, adj_left=.12)
            plt.xlabel(f"Cost (with Gamma: {self.grid.best_params_['svm__gamma']})")
            plt.ylabel("Score")
            plt.ylim(ylim[0], ylim[1])
            lw = 2
            plt.semilogx(param_range, train_scores_mean, label="Training score",
                         color="r", lw=lw)
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r", lw=lw)
            plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                         color="g", lw=lw)
            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g", lw=lw)
            plt.axvline(x=self.grid.best_params_['svm__C'], color='k', linestyle='--')
            plt.plot(self.grid.best_params_['svm__C'], self.grid.best_score_, 'ok')
            ax.text(self.grid.best_params_['svm__C'], self.grid.best_score_,
                    f' {round(self.grid.best_score_, 2)}', fontsize=20)
            plt.legend(loc="best")
            return fig

        else:
            print('radial')
            param_range = self.params[0].get('svm__C')
            train_scores, test_scores = validation_curve(self.grid.best_estimator_, self.X, self.y,
                                                         param_name='svm__C',
                                                         param_range=param_range,
                                                         scoring='accuracy',
                                                         cv=self.cv,
                                                         n_jobs=-1)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            fig1, ax1 = customplot(adj_bottom=.13, adj_left=.12)
            plt.xlabel(f"Cost (with Gamma: {self.grid.best_params_['svm__gamma']})")
            plt.ylabel("Score")
            plt.ylim(ylim[0], ylim[1])
            lw = 2
            plt.semilogx(param_range, train_scores_mean, label="Training score",
                         color="r", lw=lw)
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r", lw=lw)
            plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                         color="g", lw=lw)
            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g", lw=lw)
            plt.axvline(x=self.grid.best_params_['svm__C'], color='k', linestyle='--')
            plt.plot(self.grid.best_params_['svm__C'], self.grid.best_score_, 'ok')
            ax1.text(self.grid.best_params_['svm__C'], self.grid.best_score_,
                     f' {round(self.grid.best_score_, 2)}', fontsize=20)
            plt.legend(loc="best")

            # gamma
            param_range = self.params[0].get('svm__gamma')
            train_scores, test_scores = validation_curve(self.grid.best_estimator_, self.X, self.y,
                                                         param_name='svm__gamma',
                                                         param_range=param_range,
                                                         scoring='accuracy',
                                                         cv=self.cv,
                                                         n_jobs=-1)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            fig2, ax2 = customplot(adj_bottom=.13, adj_left=.12)
            plt.xlabel(f"Gamma (with Cost: {self.grid.best_params_['svm__C']})")
            plt.ylabel("Score")
            plt.ylim(ylim[0], ylim[1])
            lw = 2
            plt.semilogx(param_range, train_scores_mean, label="Training score",
                         color="r", lw=lw)
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r", lw=lw)
            plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                         color="g", lw=lw)
            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g", lw=lw)
            plt.axvline(x=self.grid.best_params_['svm__gamma'], color='k', linestyle='--')
            plt.plot(self.grid.best_params_['svm__gamma'], self.grid.best_score_, 'ok')
            ax2.text(self.grid.best_params_['svm__gamma'], self.grid.best_score_,
                     f' {round(self.grid.best_score_, 2)}', fontsize=20)
            plt.legend(loc="best")
            return fig1, fig2

    def plot_learning(self):
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            self.grid.best_estimator_, self.X, self.y,
            cv=self.cv, n_jobs=-1, train_sizes=np.linspace(.6, 1.0, 5),
            return_times=True,
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # plot learning curve
        fig1, ax1 = customplot(adj_bottom=.13, adj_left=.12)
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", lw=2,
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", lw=2,
                 label="Cross-validation score")
        plt.legend(loc="best")

        # Plot n_samples vs fit_times
        fig2, ax2 = customplot(adj_bottom=.13, adj_left=.15)
        plt.xlabel('Training examples')
        plt.ylabel('Fit times')
        plt.plot(train_sizes, fit_times_mean, 'o-', lw=2)
        plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)

        # Plot fit_time vs score
        fig3, ax3 = customplot(adj_bottom=.13, adj_left=.12)
        plt.xlabel('Fit times')
        plt.ylabel('Score')
        plt.plot(fit_times_mean, test_scores_mean, 'o-', lw=2)
        plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)

        return fig1, fig2, fig3


class anneal:
    def __init__(self, estimator, **options):
        self.X = None
        self.y = None
        self.estimator = estimator
        self.nsol = options.get('nsolution', 10)
        self.k = options.get('k', None)
        self.cv = options.get('cv', 10)
        self.Tinit = options.get('Tinit', 1)
        self.sizeDev = options.get('sizeDev', 1)
        self.limit = options.get('limit', 1)
        self.seed = options.get('random_state', None)
        self.cooling = options.get('cooling', 0.05)
        self.niter = options.get('niter', 100)
        self.verbose = options.get('verbose', 1)
        self.store = None
        self.score = None
        self.bestsol = None

    def __repr__(self):
        return 'Simulated Annealing Feature Extraction'

    def activation(self, subset):
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import confusion_matrix
        
        pred = cross_val_predict(self.estimator, self.X[:, subset], self.y, cv=self.cv)
        return sum(np.diag(confusion_matrix(self.y, pred))) - len(subset)

    def runOneSA(self, prefix='Calculate'):
        nvar = self.X.shape[1]
        best = curr = self.SAstep(None, maxvar=nvar, k=self.k, size_dev=self.sizeDev, limit=self.limit,
                                  random_state=self.seed)
        bestQ = currQ = self.activation(subset=curr)

        Temp = self.Tinit

        for i in range(self.niter):
            if self.verbose == 1:
                printprogressbar(i + 1, self.niter, prefix=prefix + ":", suffix='Complete', length=50)

            new = self.SAstep(curr, nvar, self.k, self.sizeDev, self.limit, self.seed)
            newQ = self.activation(new)
            accept = True

            if newQ < currQ:
                pAccept = np.exp((newQ - currQ) / Temp)
                if np.random.uniform(0, 1) > pAccept:
                    accept = False

            if accept:
                curr = new
                currQ = newQ

                if currQ > newQ:
                    best = curr
                    bestQ = currQ

            Temp = Temp * (1 - self.cooling)
        return {'best': best, 'bestScore': bestQ}

    def solution(self):
        def randomwarning():
            warnings.warn('Random state is seeded, we try to fix different random seed each the solution', stacklevel=2)

        def callwarning():
            randomwarning()

        seedinit = self.seed
        if self.seed is not None:
            callwarning()
            np.random.seed(self.seed)
            rand = np.random.randint(1, 99999, self.nsol)
        else:
            rand = None

        stored_obj = tuple()
        score = list()
        for i in range(self.nsol):

            if rand is None:
                self.seed = None
            else:
                self.seed = rand[i]

            obj = self.runOneSA(prefix=f'Solution {i}')
            stored_obj = stored_obj + (obj,)
            score.append(obj['bestScore'])

        self.store = stored_obj
        self.score = score
        self.seed = seedinit

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.solution()
        maxindex = [i for i, j in enumerate(self.score) if j == max(self.score)]
        self.bestsol = tuple()

        for i, j in enumerate(self.store):
            if i in maxindex:
                self.bestsol = self.bestsol + (j,)
        return self.bestsol

    def support_(self, sol=0):
        return self.bestsol[sol]['best']

    @staticmethod
    def SAstep(curr_set, maxvar, k=None, size_dev=1, limit=2, random_state=None):
        from nanolib.utils import sample, match

        def lessvar():
            warnings.warn('k < 2, k set to 2', stacklevel=2)

        def morevar():
            warnings.warn('k >= maxvar, k set to maxvar - limit', stacklevel=2)

        def callwarning():
            if k < 2:
                lessvar()
            else:
                morevar()

        if curr_set is None:
            if k is None:
                k = maxvar - limit
            if k >= maxvar:
                callwarning()
                k = maxvar - limit
            if k < 2:
                callwarning()
                k = 2

            return sample(maxvar, k, random_state=random_state)
        else:
            new_size = len(curr_set)

            if size_dev > 0:
                items = [x for x in range(-size_dev, size_dev + 1)]
                new_size = new_size + sample(items, 1, random_state=random_state)[0]

            if new_size < 2:
                new_size = 2

            if new_size > maxvar - limit:
                new_size = maxvar - limit

            not_in = [x for x in range(maxvar) if x not in curr_set]
            superset = curr_set + sample(not_in, max([size_dev, 1]), random_state=random_state)
            newset = sample(superset, new_size, random_state=random_state)

            while len(newset) == len(curr_set) and not (None in match(newset, curr_set)):
                newset = sample(superset, new_size)

            return newset


class genetic:
    def __init__(self, estimator, **options):
        self.bestscores, self.avgscores = [], []
        self.bestchromosomes = []
        self.X = None
        self.y = None
        self.nfeatures = None
        self.estimator = estimator
        self.seed = options.get('random_state', None)
        self.ngen = options.get('ngen', 7)
        self.size = options.get('size', 200)
        self.nbest = options.get('nbest', 40)
        self.nrand = options.get('nrand', 40)
        self.nchild = options.get('nchildren', 5)
        self.mutationrate = options.get('mutationrate', 0.05)
        self.case = options.get('regressioncase', False)
        self.cv = options.get('cv', 10)
        self.verbose = options.get('verbose', 1)

        if int((self.nbest + self.nrand) / 2) * self.nchild != self.size:
            raise ValueError('The population size is not stable.')

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def __repr__(self):
        return 'Genetic Algorithm Feature Extraction'

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.nfeatures = X.shape[1]

        self.bestscores, self.avgscores = [], []
        self.bestchromosomes = []

        population = self.initilize()
        for i in range(self.ngen):
            if self.verbose == 1:
                printprogressbar(i + 1, self.ngen, prefix=f'GEN {i + 1} of {self.ngen}: ', suffix='Complete', length=50)
            population = self.generate(population)

        print(f'Genetic Algorithm: {sum(self.bestchromosomes[-1])} '
              f'selected of {len(self.bestchromosomes[-1])} features.')

    def initilize(self):
        population = []
        for i in range(self.size):
            chromosome = np.ones(self.nfeatures, dtype=np.bool)
            mask = np.random.rand(len(chromosome)) < 0.3
            chromosome[mask] = False
            population.append(chromosome)
        return population

    def generate(self, population):
        # selection, crossover and mutation
        sortedscores, sortedpopulation = self.fitness(population)
        population = self.select(sortedpopulation)
        population = self.crossover(population)
        population = self.mutate(population)

        # history
        self.bestchromosomes.append(sortedpopulation[0])
        self.bestscores.append(sortedscores[0])
        self.avgscores.append(np.mean(sortedscores))

        return population

    def fitness(self, population):
        from sklearn.model_selection import cross_val_score
        scores = []
        if self.case:
            # for regression case, less is better.
            for chromosome in population:
                score = -1.0 * np.mean(cross_val_score(self.estimator, self.X[:, chromosome], self.y,
                                                       cv=self.cv, scoring="neg_mean_squared_error"))
                scores.append(score)
            scores, population = np.array(scores), np.array(population)
            ind = np.argsort(scores)
            return list(scores[ind]), list(population[ind, :])
        else:
            # for classification case, bigger is better.
            for chromosome in population:
                score = np.mean(cross_val_score(self.estimator, self.X[:, chromosome], self.y,
                                                cv=self.cv))
                scores.append(score)
            scores, population = np.array(scores), np.array(population)
            ind = np.argsort(-scores)
            return list(scores[ind]), list(population[ind, :])

    def select(self, sortedpopulation):
        nextpopulation = []
        for i in range(self.nbest):
            nextpopulation.append(sortedpopulation[i])
        for i in range(self.nrand):
            nextpopulation.append(random.choice(sortedpopulation))
        random.shuffle(nextpopulation)
        return nextpopulation

    def crossover(self, population):
        nextpopulation = []
        for i in range(int(len(population) / 2)):
            for j in range(self.nchild):
                chromosome1, chromosome2 = population[i], population[len(population) - 1 - i]
                child = chromosome1
                mask = np.random.rand(len(child)) > 0.5
                child[mask] = chromosome2[mask]
                nextpopulation.append(child)
        return nextpopulation

    def mutate(self, population):
        nextpopulation = []
        for i in range(len(population)):
            chromosome = population[i]
            if random.random() < self.mutationrate:
                mask = np.random.rand(len(chromosome)) < 0.05
                chromosome[mask] = False
            nextpopulation.append(chromosome)
        return nextpopulation

    @property
    def support_(self):
        return self.bestchromosomes[-1]

    def plot_scores(self):
        import matplotlib.pyplot as plt
        plt.plot(self.bestscores, label='Best')
        plt.plot(self.avgscores, label='Average')
        plt.legend()
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        plt.show()


class fcbf:
    def __init__(self):
        self.X = None
        self.y = None
        self.threshold = None
        self.F = None
        self.C = None
        self.best = None

    def fit(self, X, y, threshold=-1):
        self.X = X
        self.y = y
        self.threshold = threshold
        maxfeature = self.X.shape[1]
        thres = 0.1

        # Generating Symetrical Uncertainty Matrix
        SUM = np.zeros((maxfeature, maxfeature))
        for i in range(maxfeature):
            for j in range(maxfeature):
                temp = self.calcSU(X[:, i], X[:, j])
                if temp <= thres:
                    temp = 0
                SUM[i, j] = temp

        names = [f'F{i}' for i in range(maxfeature)]
        self.F = pd.DataFrame(data=SUM, columns=names, index=names)

        c_corr = np.zeros((maxfeature, 2))
        for i in range(maxfeature):
            temp = self.calcSU(X[:, i], y)
            c_corr[i, 0] = temp
        ind = c_corr[:, 0].argsort()[::-1]
        c_corr = c_corr[ind,]
        c_corr[:, 1] = ind
        self.C = pd.DataFrame(data=c_corr, columns=['C-Corr', 'Feature'])

        # calculate and select best features
        n = self.X.shape[1]
        slist = np.zeros((n, 3))
        slist[:, -1] = 1

        slist[:, 0] = self.c_correlation()
        idx = slist[:, 0].argsort()[::-1]
        slist = slist[idx,]
        slist[:, 1] = idx
        if self.threshold < 0:
            self.threshold = np.median(slist[-1, 0])
            print("Using minimum SU value as default threshold: {0}".format(self.threshold))
        elif self.threshold >= 1 or self.threshold > max(slist[:, 0]):
            print("No relevant features selected for given threshold.")
            print("Please lower the threshold and try again.")
            exit()

        slist = slist[slist[:, 0] > self.threshold, :]

        cache = {}
        m = len(slist)
        p_su, p, p_idx = self.getFirstElement(slist)
        for i in range(m):
            p = int(p)
            q_su, q, q_idx = self.getNextElement(slist, p_idx)
            if q:
                while q:
                    q = int(q)
                    if (p, q) in cache:
                        pq_su = cache[(p, q)]
                    else:
                        pq_su = self.calcSU(self.X[:, p], self.X[:, q])
                        cache[(p, q)] = pq_su

                    if pq_su >= q_su:
                        slist = self.removeElement(slist, q_idx)
                    q_su, q, q_idx = self.getNextElement(slist, q_idx)

            p_su, p, p_idx = self.getNextElement(slist, p_idx)
            if not p_idx:
                break

        sbest = slist[slist[:, 2] > 0, :2]
        print("\nFeatures selected: {0}".format(len(sbest)))
        print("Selected feature indices:\n{0}".format(sbest))
        self.best = sbest

    def condEntropy(self, d1, d2):
        ud2, ud2c = np.unique(d2, return_counts=True)
        prob_ud2 = ud2c / float(sum(ud2c))
        cond_entropy_d1 = np.array([self.entropy(d1[d2 == v]) for v in ud2])
        return prob_ud2.dot(cond_entropy_d1)

    def mutualInformation(self, d1, d2):
        return self.entropy(d1) - self.condEntropy(d1, d2)

    def calcSU(self, d1, d2):
        return 2.0 * self.mutualInformation(d1, d2) / (self.entropy(d1) + self.entropy(d2))

    def c_correlation(self):
        su = np.zeros(self.X.shape[1])
        for i in np.arange(self.X.shape[1]):
            su[i] = self.calcSU(self.X[:, i], self.y)
        return su

    def correlation_(self):
        return self.F, self.C

    def support_(self):
        return [int(i) for i in list(self.best[:, 1])]

    @staticmethod
    def entropy(d):
        _, vec = np.unique(d, return_counts=True)
        prob_d = np.array(vec / float(sum(vec)))
        return prob_d.dot(-np.log2(prob_d))

    @staticmethod
    def getFirstElement(d):
        t = np.where(d[:, 2] > 0)[0]
        if len(t):
            return d[t[0], 0], d[t[0], 1], t[0]
        return None, None, None

    @staticmethod
    def getNextElement(d, idx):
        t = np.where(d[:, 2] > 0)[0]
        t = t[t > idx]
        if len(t):
            return d[t[0], 0], d[t[0], 1], t[0]
        return None, None, None

    @staticmethod
    def removeElement(d, idx):
        d[idx, 2] = 0
        return d


class featureImportance:
    def __init__(self, **options):
        self.X = None
        self.y = None
        self.important = None
        self.std = None
        self.indices = None
        self.ntree = options.get('n_tress', 250)
        self.seed = options.get('random_state', None)

    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesClassifier

        self.X = X
        self.y = y

        model = ExtraTreesClassifier(n_estimators=self.ntree, random_state=self.seed)
        model.fit(self.X, self.y)
        self.important = model.feature_importances_

        self.std = np.std([tree.feature_importances_ for tree in model.estimators_],
                          axis=0)

        self.indices = np.argsort(self.important)[::-1]

        print("Feature rangking:")

        for f in range(self.X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.important[self.indices[f]]))

    def plot_(self, adj=None, yerr=True):
        from torclib2 import customplot
        import matplotlib.pyplot as plt

        if adj is None:
            aleft = .12
            abottom = .12
        else:
            aleft, abottom = adj[0], adj[1]

        fig, ax = customplot(adj_left=aleft, adj_bottom=abottom)

        if yerr:
            plt.bar(range(self.X.shape[1]), self.important[self.indices],
                    color='r', yerr=self.std[self.indices], align='center')
        else:
            plt.bar(range(self.X.shape[1]), self.important[self.indices],
                    color='r')

        plt.xticks(range(self.X.shape[1]), self.indices)
        plt.xlim([-1, self.X.shape[1]])
        plt.xlabel('Feature')
        plt.ylabel('Score')
        return fig


class QuickGA:
    """
    ========================================
    Fast Genetic Algorithm Feature Selection
    ========================================

    QuickGA(estimator, feature, listName, random_state)

    Methods:

    - fit(x, y)
    - selected()

    """

    def __init__(self, estimator, feature, listName, random_state=None):
        self.estimator = estimator
        self.X = None
        self.y = None
        self.colname = listName
        self.feature = feature
        self.seed = random_state
        self.selector = None

    def fit(self, x, y):
        from genetic_selection import GeneticSelectionCV
        import random as rn
        self.X = x
        self.y = y
        if self.seed is not None:
            np.random.seed(self.seed)
            rn.seed(self.seed)

        # calculate ga
        selector = GeneticSelectionCV(self.estimator,
                                      cv=10,
                                      verbose=1,
                                      scoring="accuracy",
                                      n_population=50,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=50,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      caching=True,
                                      n_jobs=-1)

        self.selector = selector.fit(self.X, self.y)

        print('GA - Selection')
        print(f'Number of selected features: {self.selector.n_features_}')
        print('Selected index:')
        print(pd.Series(self.colname).values[:self.feature][self.selector.support_])

    def selected(self):
        return list(pd.Series(self.colname).values[:self.feature][self.selector.support_])
