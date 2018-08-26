import pandas as pd
import time
import random
import pickle
import numpy as np
import scipy.sparse as sp
import matplotlib as mpl
import distribution as dst
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import helpers as hlp
import glob



#indexSet = indexSet
#wholeSet = range(0, matrix.m.shape[0])
#complementarySet = [i for i in wholeSet if i not in indexSet]

# Or as Set
# indexSet = set(indexSet)
# wholeSet = set(range(0,matrix.shape[0]))
# complementarySet = list(wholeSet-indexSet)
# indexSet = list(indexSet)

frame = pd.DataFrame()
#old = pickle.load(open('models/canonical/602/c(602, 0.00146484375, 0.09504403365439187).pickle', 'rb'))
#canonicalModel = dst.Distribution(old.v, old.m, old.type)
dims=500
m = sp.csc_matrix(np.random.rand(dims, dims))
subsetSize = list(range(2, m.shape[1] - 2,10))
print(m.shape)

runNTimes=5

for x in range(0,runNTimes):
    for i in subsetSize:
        print(i)
        indexSet = list(range(0, i))
        #print(indexSet)
        #print(indexSet)
        dict = {'Size': i}
    
    
        start = time.clock()
        result = m[:, indexSet][indexSet, :]
        result @ result.T
        end = time.clock()
    
        mstart = time.clock()
        result @ result.T
        mend = time.clock()
        mtime = mend - mstart
    
        dict['FancyIndexing'] = end - start -mtime
    
        start = time.clock()
        result3 = m[np.ix_(indexSet,indexSet)]
        result3 @ result3.T
        end = time.clock()
    
        mstart = time.clock()
        result3 @ result3.T
        mend = time.clock()
        mtime = mend - mstart
    
        dict['ix'] = end - start -mtime
        
        frame = frame.append(dict, ignore_index=True)
    
    frame.to_csv('intel/timeNPTakeSparse'+ str(x) +'.csv', index=False)

names = glob.glob('intel/timeNPTakeSparse[0-9].csv')
frames = map(pd.read_csv,names)
res = hlp.scalar(frames)
res.to_csv('intel/timeNPTakeSparse.csv', index=False)
parts = glob.glob('intel/timeNPTakeSparse[0-9].csv')

for i in parts:
   os.remove(i)

