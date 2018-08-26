import timing

import os
import pickle
import pandas as pd
import time
import timing
import glob
import distribution as dist
import sys
import matplotlib.pyplot as plt
import helpers as hlp
from functools import reduce
print('--------> timeInferenceOperations')
modelDir = os.path.dirname(sys.argv[1])
modelPaths = [os.path.join(modelDir + '/', dir) for dir in os.listdir(modelDir)]


runNTimes=5


def evalDir(path,i):
    print(path)
    frame = pd.DataFrame()
    modelPaths = [os.path.join(path + '/', dir) for dir in os.listdir(path)]
    onlyPickles = filter(lambda x: x.endswith('pickle'), modelPaths)
    counter=0
    for modelPath in onlyPickles:

        model = pickle.load(open(modelPath,'rb'))
        model = dist.Distribution(model.v,model.m,type='canonical')
        dict = {'Dim' : model.m.shape[0]}
        dict['Sparsity'] = model.sparsity()
        #print('mccc')
        dict['mccc'] = timing.marginalizeCanonicalConditionCanonical(model)
        #print('mccm')
        dict['mccm'] = timing.marginalizeCanonicalConditionMean(model)
        #print('mmcm')
        dict['mmcm'] = timing.marginalizeMeanConditionMean(model)
        #print('mmcc')
        dict['mmcc'] = timing.marginalizeMeanConditionCanonical(model)

        dict['conditionOnlyMean'] = timing.conditionOnlyMean(model)
        dict['conditionOnlyCanonical'] = timing.conditionOnlyCanonical(model)

        dict['mmccConvert'] = timing.marginalizeMeanConditionCanonicalConvert(model)
        dict['convertMmcc'] = timing.ConvertMarginalizeMeanConditionCanonical(model)

        frame = frame.append(dict, ignore_index=True)
    frame.to_csv(path + '/' + 'inferenceOperationsSparse'+ str(i) +'.csv', index=False)

# Ensure sequential order to minimize interference between timings

for i in range(0,runNTimes):
    for directory in modelPaths:
        evalDir(directory,i)
for directory in modelPaths:
    print(directory)
    names = glob.glob(directory+ '/' + '*.csv')
    frames = map(pd.read_csv,names) 
    res = hlp.scalar(frames)
    #res = reduce(lambda a,b: a.add(b),frames)
    res.to_csv(directory + '/' + 'inferenceOperationsSparse.csv', index=False)
    parts = glob.glob(directory+'/'+ 'inferenceOperationsSparse[0-9].csv')

    for i in parts:
        os.remove(i)




