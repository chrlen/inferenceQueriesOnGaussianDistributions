import pandas as pd
import time
import random
import pickle
import numpy as np
import os
import helpers as hlp
import glob

runNTimes=5

dims=5500
subsetSize = list(range(2, dims - 10,500))
for run in range(0,runNTimes):
    frame = pd.DataFrame()
    for i in subsetSize:
        print(i)
        m = np.random.rand(dims, dims)

        indexSet = list(range(0, i))
        dict = {'Size': i}
        
        start = time.clock()
        result2  = np.take(np.take(m, indexSet, axis=1), indexSet, axis=0)
        result2 + result2.T
        end = time.clock()
    
    
        mstart = time.clock()
        result2 + result2.T
        mend = time.clock()
        mtime = mend - mstart
    
        dict['take'] = end - start -mtime
    
        start = time.clock()
        result = m[:, indexSet][indexSet, :]
        result + result.T
        end = time.clock()
    
        mstart = time.clock()
        result + result.T
        mend = time.clock()
        mtime = mend - mstart
    
        dict['FancyIndexing'] = end - start -mtime
    
        start = time.clock()
        result3 = m[np.ix_(indexSet,indexSet)]
        result3 + result3.T
        end = time.clock()
    
        mstart = time.clock()
        result3 + result3.T
        mend = time.clock()
        mtime = mend - mstart
    
        dict['ix'] = end - start - mtime

        frame = frame.append(dict, ignore_index=True)
    frame.to_csv('intel/timeNPTake'+ str(run) +'.csv', index=False)

names = glob.glob('intel/timeNPTake[0-9].csv')
frames = map(pd.read_csv,names)
res = hlp.scalar(frames)
res.to_csv('intel/timeNPTake.csv', index=False)
parts = glob.glob('intel/timeNPTake[0-9].csv')
for i in parts:
    os.remove(i)







