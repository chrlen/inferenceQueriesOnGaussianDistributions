import sys
import os
import pandas as pd
from functools import reduce
print('--------> consolidateDataframes')
filename=sys.argv[1]
modelDir = os.path.abspath(sys.argv[2])
initPath = os.getcwd()
print(initPath)
os.chdir(modelDir)
targetDir = sys.argv[3]
print('Searching '+ filename + ' in ' + modelDir)

modelDirs = [dir for dir in os.listdir(modelDir)]
#print(modelDirs)
csvs = [x + '/'+filename for x in modelDirs]
#print(csvs)
csvs = list(filter(os.path.isfile,csvs))
#print(len(csvs))

dataframes = list(map(pd.read_csv,csvs))
for i in csvs:
	os.remove(i)
#print(len(dataframes))
df = pd.concat(dataframes)
os.chdir(initPath)
os.chdir(targetDir)
df = df.sort_values(['Dim','Sparsity'])
df.to_csv(filename,index=False)

	