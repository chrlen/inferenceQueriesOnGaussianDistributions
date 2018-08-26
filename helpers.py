import numpy as np
from sklearn.covariance import GraphLasso
import pandas as pd
import scipy.sparse as sp
import multiprocessing as mp
from functools import partial
import statistics as stats
from itertools import repeat

def randomPositiveSemidefinite(n):
    def isPSD(matrix):
        return (all(map(lambda x: x >= 0, np.linalg.eig(matrix)[0])))

    m = np.random.rand(n, n)
    cov = m.T @ m
    if not isPSD(cov):
        print("Cov is not psd!!!")
    return (cov)

def estimatePrecisionFromFile(file, dims, sparsity):
    data = pd.read_csv(file)
    model = GraphLasso()
    model.fit(data)
    return sp.csc_matrix(model.precision_)


def estimatePrecisionFromSet(data, alpha):
    model = GraphLasso(alpha=alpha)
    model.fit(data)
    return sp.csc_matrix(model.precision_)


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def scalar(listOfFrames,func=stats.median):
    listOfFrames=list(listOfFrames)
    res = listOfFrames[0].copy()
    indexSet =[(x,y) for x in range(0,res.shape[0]-1) for y in range(0,res.shape[1]-1)]
    print(res.shape)
    for pair in indexSet:

        vec = list()
        for i in listOfFrames:
            vec.append(i.iloc[pair[0],pair[1]])
        res.iloc[pair[0],pair[1]] = func(vec)
    return(res)
