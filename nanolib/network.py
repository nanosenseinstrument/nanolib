# -*- coding: utf-8 -*-
"""
This file is part of NANOLIB


NANOLIB was primarily developed at Nanosense by:
    Shidiq Nur Hidayat (s.hidayat@nanosense-id.com)

Created on Tue Jul 14 18:23:51 2020

@author: Shidiq Nur Hidayat
"""

import pandas as pd
# import io
from contextlib import redirect_stdout
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import time
import csv
from tensorflow.keras import backend as K


class DNNLogger:
    """
    =======================================
    DNN Logger Class for DNN Classification
    =======================================
    
    torcdnn(nfeature=2,
            nclasses=2,
            iteration=1,
            hidden=[500,1000],
            activation=['relu', 'relu'],
            dropout=0.2,
            l2=None,
            epochs=100,
            batch_size=5,
            validation_size=None,
            validation_data=True,
            optimizer='adam',
            lastactivation='softmax',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            scaler=StandarScaler,
            iteration=1,
            logdir='D:\\my_logs',
            verbose=1,
            )
    
    Methods:
        - fit(xtrain, xtest, ytrain, ytest)
    """

    def __init__(self, nfeature=2, nclasses=2, **options):
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.info = None
        self.nfeature = nfeature
        self.nclasses = nclasses
        self.hidden = options.get('hidden', [500, 1000])
        self.activation = options.get('activation', ['relu', 'relu'])
        self.dropout = options.get('dropout', 0.2)
        self.l2 = options.get('l2', None)
        self.niter = options.get('iteration', 1)
        self.epochs = options.get('epochs', 100)
        self.batchsize = options.get('batch_size', 5)
        self.validationsize = options.get('validation_size', None)
        self.validationdata = options.get('validation_data', True)
        self.logdir = options.get('logdir', 'D:\\my_logs')
        self.verbose = options.get('verbose', 1)
        self.scaler = options.get('scaler', StandardScaler)
        self.optimizer = options.get('optimizer', 'adam')
        self.metrics = options.get('metrics', ['accuracy'])
        self.lastactivation = options.get('lastactivation', 'softmax')
        self.loss = options.get('loss', 'sparse_categorical_crossentropy')
        self.config()

    def config(self):
        os.makedirs(self.logdir, exist_ok=True)
        neuronx = [f'hidden{i}' for i in range(len(self.hidden))]
        zipneuron = zip(neuronx, self.hidden)
        self.info = dict(zipneuron)
        actx = [f'activation{i}' for i in range(len(self.activation))]
        zipact = zip(actx, self.activation)
        self.info.update(dict(zipact))
        self.info['dropout'] = self.dropout
        self.info['l2'] = f'{self.l2}'
        self.info['epochs'] = self.epochs
        self.info['batchsize'] = self.batchsize
        self.info['optimizer'] = self.optimizer
        self.info['output_activation'] = self.lastactivation
        self.info['loss'] = self.loss
        self.info['scaler'] = f'{self.scaler}'
        self.info['val_size_status'] = f'{self.validationsize}'
        self.info['val_data_status'] = f'{self.validationdata}'
        self.info['logdir'] = self.logdir
        self.info['input_space'] = self.nfeature
        self.info['output_space'] = self.nclasses

        if len(self.hidden) != len(self.activation):
            raise Exception('number of hidden not same as number of activation')

    def createnetwork(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.hidden[0], activation=self.activation[0], input_shape=(self.nfeature,)))
        model.add(tf.keras.layers.Dropout(self.dropout))
        for i in range(len(self.hidden) - 1):
            if self.l2 is None:
                model.add(tf.keras.layers.Dense(self.hidden[i + 1], activation=self.activation[i + 1]))
            else:
                model.add(tf.keras.layers.Dense(self.hidden[i + 1], activation=self.activation[i + 1],
                                                kernel_regularizer=tf.keras.regularizers.l2(self.l2)))
            model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(self.nclasses, activation=self.lastactivation))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def fit(self, *arrays):
        self.xtrain = arrays[0]
        self.xtest = arrays[1]
        self.ytrain = arrays[2]
        self.ytest = arrays[3]

        if self.scaler is not None:
            scaler = self.scaler()
            self.xtrain = scaler.fit_transform(self.xtrain)
            self.xtest = scaler.transform(self.xtest)

        loss_train = []
        loss_test = []
        acc_train = []
        acc_test = []
        list_id = []
        get_run_logdir = []
        # f = io.StringIO()

        for i in range(self.niter):
            print(i)

            run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            get_run_logdir = os.path.join(self.logdir, run_id)

            logcsv = tf.keras.callbacks.CSVLogger(f'{get_run_logdir}.csv', append=True)
            early_stopper = tf.keras.callbacks.EarlyStopping(patience=5)

            model = self.createnetwork()

            if self.validationdata is False:
                if self.validationsize is None:
                    model.fit(self.xtrain, self.ytrain, epochs=self.epochs,
                              batch_size=self.batchsize, verbose=self.verbose,
                              callbacks=[logcsv, early_stopper])
                else:
                    model.fit(self.xtrain, self.ytrain, epochs=self.epochs,
                              batch_size=self.batchsize, verbose=self.verbose,
                              validation_split=self.validationsize,
                              callbacks=[logcsv, early_stopper])
            else:
                model.fit(self.xtrain, self.ytrain, epochs=self.epochs,
                          batch_size=self.batchsize, verbose=self.verbose,
                          validation_data=(self.xtest, self.ytest),
                          callbacks=[logcsv, early_stopper])

            with open(f'{get_run_logdir}.txt', 'w') as f:
                with redirect_stdout(f):
                    results_train = model.evaluate(self.xtrain, self.ytrain, verbose=2)
                    results_test = model.evaluate(self.xtest, self.ytest, verbose=2)
                    print(results_train)
                    print(results_test)

            loss_train.append(results_train[0])
            acc_train.append(results_train[1])
            loss_test.append(results_test[0])
            acc_test.append(results_test[1])
            list_id.append(run_id)

            model.save(f'{get_run_logdir}.h5')

        res = {
            'ID': list_id,
            'loss Train': loss_train,
            'acc Train': acc_train,
            'loss Test': loss_test,
            'acc Test': acc_test, }

        res = pd.DataFrame(data=res)
        res.to_csv(f'{get_run_logdir}_SUMMARY.csv')
        w = csv.writer(open(f'{get_run_logdir}_config.csv', 'w'))
        for key, val in self.info.items():
            w.writerow([key, val])


class BINDNNLogger:
    """
    =======================================
    DNN Logger Class for DNN Classification
    =======================================
    
    torcdnn(nfeature=2,
            nclasses=2,
            iteration=1,
            hidden=[500,1000],
            activation=['relu', 'relu'],
            dropout=0.2,
            l2=None,
            epochs=100,
            batch_size=5,
            validation_size=None,
            validation_data=True,
            optimizer='adam',
            lastactivation='softmax',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            scaler=StandarScaler,
            iteration=1,
            logdir='D:\\my_logs',
            verbose=1,
            )
    
    Methods:
        - fit(xtrain, xtest, ytrain, ytest)
    """

    def __init__(self, nfeature=2, nclasses=2, **options):
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.info = None
        self.nfeature = nfeature
        self.nclasses = nclasses
        self.hidden = options.get('hidden', [500, 1000])
        self.activation = options.get('activation', ['relu', 'relu'])
        self.dropout = options.get('dropout', 0.2)
        self.l2 = options.get('l2', None)
        self.niter = options.get('iteration', 1)
        self.epochs = options.get('epochs', 100)
        self.batchsize = options.get('batch_size', 5)
        self.validationsize = options.get('validation_size', None)
        self.validationdata = options.get('validation_data', True)
        self.logdir = options.get('logdir', 'D:\\my_logs')
        self.verbose = options.get('verbose', 1)
        self.scaler = options.get('scaler', StandardScaler)
        self.optimizer = options.get('optimizer', 'adam')
        self.metrics = options.get('metrics', ['accuracy'])
        self.lastactivation = options.get('lastactivation', 'softmax')
        self.loss = options.get('loss', 'sparse_categorical_crossentropy')
        self.config()

    def config(self):
        os.makedirs(self.logdir, exist_ok=True)
        neuronx = [f'hidden{i}' for i in range(len(self.hidden))]
        zipneuron = zip(neuronx, self.hidden)
        self.info = dict(zipneuron)
        actx = [f'activation{i}' for i in range(len(self.activation))]
        zipact = zip(actx, self.activation)
        self.info.update(dict(zipact))
        self.info['dropout'] = self.dropout
        self.info['l2'] = f'{self.l2}'
        self.info['epochs'] = self.epochs
        self.info['batchsize'] = self.batchsize
        self.info['optimizer'] = self.optimizer
        self.info['output_activation'] = self.lastactivation
        self.info['loss'] = self.loss
        self.info['scaler'] = f'{self.scaler}'
        self.info['val_size_status'] = f'{self.validationsize}'
        self.info['val_data_status'] = f'{self.validationdata}'
        self.info['logdir'] = self.logdir
        self.info['input_space'] = self.nfeature
        self.info['output_space'] = self.nclasses

        if len(self.hidden) != len(self.activation):
            raise Exception('number of hidden not same as number of activation')

    def createnetwork(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.hidden[0], activation=self.activation[0], input_shape=(self.nfeature,)))
        model.add(tf.keras.layers.Dropout(self.dropout))
        for i in range(len(self.hidden) - 1):
            if self.l2 is None:
                model.add(tf.keras.layers.Dense(self.hidden[i + 1], activation=self.activation[i + 1]))
            else:
                model.add(tf.keras.layers.Dense(self.hidden[i + 1], activation=self.activation[i + 1],
                                                kernel_regularizer=tf.keras.regularizers.l2(self.l2)))
            model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(self.nclasses, activation=self.lastactivation))

        model.compile(optimizer=self.optimizer, loss=self.loss,
                      metrics=['acc', self.f1_m, self.precision_m, self.recall_m])
        return model

    def fit(self, *arrays):
        self.xtrain = arrays[0]
        self.xtest = arrays[1]
        self.ytrain = arrays[2]
        self.ytest = arrays[3]

        if self.scaler is not None:
            scaler = self.scaler()
            self.xtrain = scaler.fit_transform(self.xtrain)
            self.xtest = scaler.transform(self.xtest)

        loss_train = []
        acc_train = []
        f1_train = []
        precision_train = []
        recall_train = []
        loss_test = []
        acc_test = []
        f1_test = []
        precision_test = []
        recall_test = []
        list_id = []
        get_run_logdir = []
        # f = io.StringIO()

        for i in range(self.niter):
            print(i)

            run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            get_run_logdir = os.path.join(self.logdir, run_id)

            logcsv = tf.keras.callbacks.CSVLogger(f'{get_run_logdir}.csv', append=True)
            early_stopper = tf.keras.callbacks.EarlyStopping(patience=5)

            model = self.createnetwork()

            if self.validationdata is False:
                if self.validationsize is None:
                    model.fit(self.xtrain, self.ytrain, epochs=self.epochs,
                              batch_size=self.batchsize, verbose=self.verbose,
                              callbacks=[logcsv, early_stopper])
                else:
                    model.fit(self.xtrain, self.ytrain, epochs=self.epochs,
                              batch_size=self.batchsize, verbose=self.verbose,
                              validation_split=self.validationsize,
                              callbacks=[logcsv, early_stopper])
            else:
                model.fit(self.xtrain, self.ytrain, epochs=self.epochs,
                          batch_size=self.batchsize, verbose=self.verbose,
                          validation_data=(self.xtest, self.ytest),
                          callbacks=[logcsv, early_stopper])

            with open(f'{get_run_logdir}.txt', 'w') as f:
                with redirect_stdout(f):
                    loss_train, accuracy_train, f1_score_train, precision_train, recall_train = model.evaluate(
                        self.xtrain, self.ytrain, verbose=2)
                    loss_test, accuracy_test, f1_score_test, precision_test, recall_test = model.evaluate(self.xtest,
                                                                                                          self.ytest,
                                                                                                          verbose=2)
                    print(accuracy_train)
                    print(accuracy_test)

            loss_train.append(loss_train)
            acc_train.append(accuracy_train)
            f1_train.append(f1_score_train)
            precision_train.append(precision_train)
            recall_train.append(recall_train)
            loss_test.append(loss_test)
            acc_test.append(accuracy_test)
            f1_test.append(f1_score_test)
            precision_test.append(precision_test)
            recall_test.append(recall_test)
            list_id.append(run_id)

            model.save(f'{get_run_logdir}.h5')

        res = {
            'ID': list_id,
            'loss Train': loss_train,
            'acc Train': acc_train,
            'f1 Train': f1_train,
            'precision Train': precision_train,
            'recall Train': recall_train,
            'loss Test': loss_test,
            'acc Test': acc_test,
            'f1 Test': f1_test,
            'precision Test': precision_test,
            'recall Test': recall_test,
        }

        res = pd.DataFrame(data=res)
        res.to_csv(f'{get_run_logdir}_SUMMARY.csv')
        w = csv.writer(open(f'{get_run_logdir}_config.csv', 'w'))
        for key, val in self.info.items():
            w.writerow([key, val])

    @staticmethod
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
