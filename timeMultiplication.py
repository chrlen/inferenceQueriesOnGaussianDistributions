import numpy as np
import time
import pickle
import scipy.sparse as sp
import pandas as pd

import os
import helpers as hlp
import glob


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

m = pickle.load(open('models/canonical/50/s0.1.pickle', 'rb'))
dims=m.m.shape[0]-10
m = m.m.todense()

#m = np.random.rand(dims, dims)

m_csc = sp.csc_matrix(m)
m_csr = sp.csr_matrix(m)


runNTimes=5

for run in range(0,runNTimes):
    subsetSize = list(range(2, m.shape[1] - 2,1000))
    #subsetSize = list(range(2, dims,10))
    frame = pd.DataFrame()
    for i in subsetSize:
        dict = {'Size': i}
        indexSet = list(range(0, i))
    
        t = m.take(indexSet,axis=0).take(indexSet,axis=1)
        t_csc = m_csc[indexSet,:][:,indexSet]
        t_csr = m_csr[indexSet,:][:,indexSet]
    
        v = np.random.rand(i)
    
        start = time.clock()
        result = t.dot(v)
        print(result.shape)
        end=time.clock()
        
        dict['numpy'] = end-start
    
    
    
        start = time.clock()
        result = t_csc.dot(v)
        print(result.shape)
        end=time.clock()
    
        dict['csc'] =end-start
        
    
        start = time.clock()
        result = t_csr.dot(v)
        print(result.shape)
        end=time.clock()
        dict['csr'] =end-start
    
    
    
        frame = frame.append(dict, ignore_index=True)
    
    frame.to_csv('intel/timeMultiplication'+ str(run) +'.csv', index=False)

names = glob.glob('intel/timeMultiplication[0-9].csv')
print(names)
frames = map(pd.read_csv,names)
res = hlp.scalar(frames)
res.to_csv('intel/timeMultiplication.csv', index=False)
parts = glob.glob('intel/timeMultiplication[0-9].csv')
for i in parts:
    os.remove(i)


