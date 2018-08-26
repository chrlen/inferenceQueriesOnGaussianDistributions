import pandas as pd
import time
import random
import os
import helpers as hlp
import glob

wholeSet = list(range(1, 1000))
subsetSize = range(3, 401, 10)
runNTimes=5

for run in range(0,runNTimes):
    frame = pd.DataFrame()
    for i in subsetSize:
        indexSet = random.sample(wholeSet, i)
        dict = {'Size': i}
    
        start = time.clock()
        complementarySet = list(set(wholeSet) - set(indexSet))
        end = time.clock()
        dict['Set'] = end - start
    
        start = time.clock()
        complementarySet = [j for j in wholeSet if j not in indexSet]
        end = time.clock()
    
        dict['List'] = end - start
        frame = frame.append(dict, ignore_index=True)
    frame.to_csv('intel/timeIndex'+ str(run) +'.csv', index=False)

names = glob.glob('intel/timeIndex[0-9].csv')
frames = map(pd.read_csv,names)
res = hlp.scalar(frames)
res.to_csv('intel/timeIndex.csv', index=False)
parts = glob.glob('intel/timeIndex[0-9].csv')
for i in parts:
    os.remove(i)



