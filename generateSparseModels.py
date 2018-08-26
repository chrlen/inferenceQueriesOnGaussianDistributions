import distribution as dst
import numpy
import scipy.sparse
import copy
import os
import pickle
import functools as ft
import multiprocessing as mp
import math 
import random

modelsPath = "models/canonical/"

#Define which dimensions should be used
dims = list(range(100, 500, 100))

#Define which sparsities should be used
sparsities = numpy.array(range(1, 10))/10

def genModel(sparsity,dim):
    print(str(dim)+ " " + str(sparsity))
    #Start with identity matrix
    model = numpy.eye(dim)
    #Calculate how many entries we have to set for given sparsity
    numOfNNZ = ((1-sparsity) * (dim * dim))-dim
    #Divide because the matrix is symmetric
    numOfNNZsym = math.ceil(numOfNNZ/2)
    #Generate index set of elements in the upper triangle
    ind = [(i,j) for i in range(1,dim) for j in range(i+1,dim)]
    #Randomly sample elements from the index set of the upper triangle
    r = random.sample(list(range(0,len(ind))),numOfNNZsym)
    #Retrieve the index pairs
    chosenInd = [ind[i] for i in r]
    #For every index pain (x,y) set m[x,y] and m[y,x] to attain symmetry
    for i in chosenInd:
        randomNum = random.random()
        model[i[0],i[1]] = randomNum
        model[i[1],i[0]] = randomNum

    #Calculate information vector
    inf = numpy.linalg.inv(model).dot(numpy.random.rand(dim))
    #Construct compressed sparse column structure
    model = scipy.sparse.csc_matrix(model)
    d = dst.Distribution(inf,model,'canonical')
    with open(modelsPath+ str(dim) + '/s' + str(sparsity) + '.pickle', 'wb') as f:
                pickle.dump(d, f)


for dim in dims:
    subdirPath = modelsPath + str(dim) + '/'
    subdir = os.path.dirname(subdirPath)
    try:
        os.stat(subdir)
    except:
        os.makedirs(subdir)

    part = ft.partial(genModel,dim=dim)

    with mp.Pool(mp.cpu_count()) as p:
        p.map(part,sparsities)






