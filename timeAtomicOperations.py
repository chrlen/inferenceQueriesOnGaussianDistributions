import numpy as np
import scipy.sparse as sp
import os
import pickle
import pandas as pd
import time
import timing
import distribution as dist
import sys
import matplotlib.pyplot as plt
import glob
from functools import reduce
import helpers as hlp
#print('--------> timeAtomicOperations')
modelDir = os.path.dirname(sys.argv[1])
modelPaths = [os.path.join(modelDir + '/', dir) for dir in os.listdir(modelDir)]


runNTimes=5


def evalDir(path,i):
    #print(path)
    frame = pd.DataFrame()
    modelPaths = [os.path.join(path + '/', dir) for dir in os.listdir(path)]
    onlyPickles = filter(lambda x: x.endswith('pickle'), modelPaths)
    counter=0
    for modelPath in onlyPickles:

        model = pickle.load(open(modelPath,'rb'))
        model = dist.Distribution(model.v,model.m,type='canonical')

        dict = {'Dim' : model.m.shape[0]}
        dict['Sparsity'] = model.sparsity()

        #print('inversionTimeSparse')
        dict['inversionTimeSparse'] = timing.inversionTimeSparse(model)
        #dict['inversionTimeSparseCSC'] = timing.inversionTimeSparse(model)

        #print('inversionTimeSparseToDense')
        #dict['inversionTimeSparseToDense'] = timing.inversionTimeSparseToDense(model)

        #print('inversionTimeToDenseSplit')
        #dict['inversionTimeToDenseSplit'] = timing.inversionTimeSparseToDenseSplit(model)

        #print('sparseMatrixDotVectorWithScipySparse')
        dict['sparseMatrixDotVectorWithScipySparse'] = timing.sparseMatrixDotVectorWithScipySparse(model)

        ##print('sparseMatrixDotVectorWithNumpy')
        #dict['sparseMatrixDotVectorWithNumpy'] = timing.sparseMatrixDotVectorWithNumpy(model)

        #print('sparseSubsetFI')
        dict['sparseSubsetFI'] = timing.sparseSubsetFI(model)
        #print('sparseSubsetIX')
        dict['sparseSubsetIX'] = timing.sparseSubsetIX(model)

        #print("sparseMatrixDotMatrix")
        dict['sparseMatrixDotMatrix'] = timing.sparseMatrixDotMatrix(model)


        model.meanForm()
        ##print(model.type)
        #print('inversionTimeDense')
        dict['inversionTimeDense'] = timing.inversionTimeDense(model)
        #print('inversionTimeDenseToSparse')
        #dict['inversionTimeDenseToSparse'] = timing.inversionTimeDenseToSparse(model)
        #print('denseMatrixDotVector')
        dict['denseMatrixDotVector'] = timing.denseMatrixDotVector(model)
        #print('denseMatrixDotVectorFlattened')
        #dict['denseMatrixDotVectorFlattened'] = timing.denseMatrixDotVectorFlattened(model)
        #print('denseMatrixDotMatrix')
        dict['denseMatrixDotMatrix'] = timing.denseMatrixDotMatrix(model)

        frame = frame.append(dict, ignore_index=True)
    frame.to_csv(path + '/' + 'atomicOperations'+ str(i) +'.csv', index=False)

# Ensure sequential order to minimize interference between timings
for i in range(0,runNTimes):
    print("Run: " + str(i))
    for directory in modelPaths:
        evalDir(directory,i)
for directory in modelPaths:
    #print(directory)
    names = glob.glob(directory+ '/' + '*.csv')
    frames = map(pd.read_csv,names)
    res = hlp.scalar(frames)
    res.to_csv(directory + '/' + 'atomicOperations.csv', index=False)
    parts = glob.glob(directory+'/'+ 'atomicOperations[0-9].csv')

    for i in parts:
        os.remove(i)
