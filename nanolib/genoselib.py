import numpy as np
import pandas as pd
from fnmatch import fnmatch
from tkinter import Tk, filedialog
from scipy.stats import zscore
import os


class DataExtract:
    def __init__(self, fun, limit, baseline=0, nsensor=10, mode='diff', envi=0):
        self.fun = fun  # [np.max, np.mean, np.std, np.trapz]
        self.limit = limit  # [100, 500]
        self.baseline = baseline
        self.nsensor = nsensor
        self.mode = mode
        self.envi = envi

    def transform(self, data):
        data = data.drop(list(data)[0], axis=1)
        selected = np.arange(self.limit[0], self.limit[1])

        data_temp = data[list(data)[self.nsensor]].loc[self.limit[0]: self.limit[1]]
        data_humid = data[list(data)[self.nsensor + 1]].loc[self.limit[0]: self.limit[1]]

        storage = pd.DataFrame()
        for i, fun in enumerate(self.fun):
            def fe(y):
                if self.baseline == 0:
                    y0 = y[self.limit[0]]
                else:
                    y0 = np.mean(y[:self.limit[0]])

                # pre-processing
                if self.mode == 'diff':
                    ys = np.subtract(y[selected], y0)
                elif self.mode == 'frac':
                    ys = np.subtract(y[selected], y0)
                    ys = np.divide(ys, y0)
                else:
                    ys = y[selected]

                return fun(ys)

            names = [f'F{i}{x + 1}' for x in range(self.nsensor)]
            temp = data.apply(fe)[:self.nsensor]

            if self.envi == 1:
                temp = temp.append(pd.Series([data_temp.median(), data_humid.median()]), ignore_index=True)
                names = names + ['Temp', 'Humid']
            elif self.envi == 2:
                temp = temp.append(pd.Series([data_temp.median(), data_temp.std(), data_humid.median(), data_humid.std()]), ignore_index=True)
                names = names + ['Temp', 'dTemp', 'Humid', 'dHumid']
            
            temp = pd.DataFrame(temp).transpose()
            temp.columns = names

            storage = pd.concat([storage, temp], axis=1)

        return storage


def listfolder():
    root = Tk()
    root.withdraw()
    logdir = filedialog.askdirectory()
    print(f'dir: {logdir}')
    print('checking selected folder: start...')
    pattern = '*.csv'
    listfiles = os.listdir(logdir)
    rawdata = [item for item in listfiles if fnmatch(item, pattern)]
    print('checking selected folder: finish!')
    return logdir, rawdata


def printprogressbar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledlength = int(length * iteration // total)
    bar = fill * filledlength + '-' * (length - filledlength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class RemoveOutlier:
    def __init__(self, multilabel=True):
        self.multilabel = multilabel

    def transform(self, data, feature=40, key='label'):
        if self.multilabel:
            unik = np.unique(data[key].values)

            store = pd.DataFrame()
            for i in unik:
                ind = data[key].isin([i])
                datai = data[ind]
                z_scores = zscore(datai[list(data)[:feature]])
                abs_z_scores = np.abs(z_scores)
                filtered_entries = (abs_z_scores < 3).all(axis=1)
                datai = datai[filtered_entries]
                store = store.append(datai)

        else:
            z_scores = zscore(data[list(data)[:feature]])
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            store = data[filtered_entries]

        return store


class Windowing:
    """
    windowing
    n = number of window
    ts = intercept time
    tdelay = delay time
    tsampling = sampling time
    fun = numpy function array
    mode = 'norm' (default), 'diff' or 'frac'
    """

    def __init__(self, n=3, ts=10, tdelay=10, tsampling=40, fun=None, mode='norm', envi=0, nsensor=10):
        self.n = n
        self.ts = ts
        self.fun = fun
        self.tdelay = tdelay * 10
        self.tsampling = tsampling * 10
        self.mode = mode
        self.storage = pd.DataFrame()
        self.envi = envi
        self.nsensor = nsensor

    def func(self, y):
        t = len(y) - self.tdelay
        tn = round(t / self.n)
        z = np.arange(self.n)
        store = []

        for i in z:
            if i == 0:
                selected = np.arange(0, tn + self.ts)
            elif i == len(z) - 1:
                selected = np.arange(i * tn - self.ts, t)
            else:
                selected = np.arange(i * tn - self.ts, (i + 1) * tn + self.ts)

            y0 = y[self.tdelay]
            selected += self.tdelay

            if self.mode == 'norm':
                ys = y[selected]
            elif self.mode == 'diff':
                ys = y[selected] - y0
            else:
                ys = (y[selected] - y0) / y0

            store += [f(ys) for f in self.fun]

        return store

    def transform(self):
        logdir, rawdata = listfolder()

        self.storage = pd.DataFrame()
        for item in rawdata:
            data = pd.read_csv(os.path.join(logdir, item))

            data_temp = data[list(data)[self.nsensor + 1]].loc[self.tdelay: (self.tdelay + self.tsampling)]
            data_humid = data[list(data)[self.nsensor + 2]].loc[self.tdelay: (self.tdelay + self.tsampling)]

            data = data[list(data)[1:(self.nsensor + 1)]]
            data = data.loc[:((self.tdelay + self.tsampling) - 1), :]

            res = [self.func(data[i].values) for i in list(data)]
            res = np.asarray(res)
            res = res.flatten()

            if self.envi == 1:
                res = np.append(res, [data_temp.median(), data_humid.median()])
            elif self.envi == 2:
                res = np.append(res, [data_temp.median(), data_temp.std(), data_humid.median(), data_humid.std()])

            res = pd.DataFrame(res).transpose()

            res['label'] = item.split('_')[0]
            res['file'] = item

            self.storage = self.storage.append(res)

        print('DONE!')

    def filetransform(self, data):
        """
        data = DataFrame
        :return values
        """
        data_temp = data[list(data)[self.nsensor + 1]].loc[self.tdelay: (self.tdelay + self.tsampling)]
        data_humid = data[list(data)[self.nsensor + 2]].loc[self.tdelay: (self.tdelay + self.tsampling)]

        data = data[list(data)[1:(self.nsensor + 1)]]
        data = data.loc[:((self.tdelay + self.tsampling) - 1), :]

        res = [self.func(data[i].values) for i in list(data)]
        res = np.asarray(res)
        res = res.flatten()

        if self.envi == 1:
            res = np.append(res, [data_temp.median(), data_humid.median()])
        elif self.envi == 2:
            res = np.append(res, [data_temp.median(), data_temp.std(), data_humid.median(), data_humid.std()])

        res = pd.DataFrame(res).transpose()

        return res.values
