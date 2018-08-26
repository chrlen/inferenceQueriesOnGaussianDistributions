import timing

import os
import pickle
import pandas as pd
import timing
import distributionDense as dist
import sys
import glob
import helpers as hlp
from functools import reduce



runNTimes=5


print('--------> timeInferenceOperationsDense')
modelDir = os.path.dirname(sys.argv[1])
modelPaths = [os.path.join(modelDir + '/', dir) for dir in os.listdir(modelDir)]

def evalDir(path,i):
    print(path)
    frame = pd.DataFrame()
    modelPaths = [os.path.join(path + '/', dir) for dir in os.listdir(path)]
    onlyPickles = filter(lambda x: x.endswith('pickle'), modelPaths)
    counter=0
    for modelPath in onlyPickles:

        model = pickle.load(open(modelPath,'rb'))
        model = dist.DistributionDense(model.v,model.m.todense(),type='canonical')
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
    print(frame.shape)
    frame.to_csv(path + '/' + 'inferenceOperationsDense'+ str(i) +'.csv', index=False)

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
    res.to_csv(directory + '/' + 'inferenceOperationsDense.csv', index=False)
    parts = glob.glob(directory+'/'+ 'inferenceOperationsDense[0-9].csv')

for i in parts:
    os.remove(i)
