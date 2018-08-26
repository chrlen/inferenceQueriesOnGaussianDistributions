import numpy as np
import time
import pickle
import scipy.sparse as sp
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

m = pickle.load(open('models/canonical/50/s0.1.pickle', 'rb'))
dims=m.m.shape[0]-10
m = m.m.todense()

print(m.shape)
#m = np.random.rand(dims, dims)

m_csc = sp.csc_matrix(m)
m_csr = sp.csr_matrix(m)


runNTimes=5

import os
import helpers as hlp
import glob


#subsetSize = list(range(2, m.shape[1] - 2))
for run in range(0,runNTimes):
    subsetSize = list(range(2, dims,1000))
    frame = pd.DataFrame()
    for i in subsetSize:
        print('-------- ' + str(i) + ' --------')
        dict = {'Size': i}
        indexSet = list(range(0, i))
    
        t = m.take(indexSet,axis=0).take(indexSet,axis=1)
        t_csc = m_csc[indexSet,:][:,indexSet]
        t_csr = m_csr[indexSet,:][:,indexSet]
    
        start = time.clock()
        result = np.linalg.inv(t)
        end=time.clock()
        
        dict['numpy'] = end-start
    
    
        start = time.clock()
        result = sp.linalg.inv(t_csc)
        end=time.clock()
    
        dict['csc'] =end-start
        
    
        start = time.clock()
        result = sp.linalg.inv(t_csr)
        end=time.clock()
    
        dict['csr'] =end-start
    
        frame = frame.append(dict, ignore_index=True)
    frame.to_csv('intel/timeInversion'+ str(run) +'.csv', index=False)
names = glob.glob('intel/timeInversion[0-9].csv')
print(names)
frames = map(pd.read_csv,names)
res = hlp.scalar(frames)
res.to_csv('intel/timeInversion.csv', index=False)
parts = glob.glob('intel/timeInversion[0-9].csv')
for i in parts:
    os.remove(i)