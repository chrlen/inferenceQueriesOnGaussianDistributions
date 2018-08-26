import time
import scipy
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import math

def conversionTimeMC(meanModel):
    m = meanModel.copy()
    m.meanForm()# Just in Case
    start = time.clock()
    m.canonicalForm()# Convert to canonical Form
    end = time.clock()
    return end - start


def conversionTimeCM(canonicalModel):
    c = canonicalModel.copy()
    c.canonicalForm()
    start = time.clock()
    c.meanForm()
    end = time.clock()
    return end - start


#---------------- Atomic Operations ----------------
#scipy.sparse.linalg.inv(canonicalModel.m)
def inversionTimeSparse(canonicalModel):
    model = canonicalModel.copy()
    start = time.clock()
    result = scipy.sparse.linalg.inv(model.m)
    end = time.clock()
    t = end -start
    #print(t)
    return t

def inversionTimeSparseCSC(canonicalModel):
    model = canonicalModel.copy()
    m = scipy.sparse.csc_matrix(model.m)
    start = time.clock()
    result = scipy.sparse.linalg.inv(m)
    end = time.clock()
    t = end -start
    #print(t)
    return t

def inversionTimeSparseToDense(canonicalModel):
    model = canonicalModel.copy()
    model.canonicalForm()# Just in Case
    start = time.clock()
    result = scipy.sparse.linalg.inv(model.m).todense()
    end = time.clock()
    t = end -start
    #print(t)
    return t

def inversionTimeSparseToDenseSplit(canonicalModel):
    model = canonicalModel.copy()
    model.canonicalForm()# Just in Case
    start = time.clock()
    result = scipy.sparse.linalg.inv(model.m)
    #a = 5
    result = result.todense()
    end = time.clock()
    t = end -start
    #print(t)
    return t

def inversionTimeDense(meanModel):
    model = meanModel.copy()
    start = time.clock()
    result = np.linalg.inv(model.m)
    end = time.clock()
    t = end -start
    #print(t)
    return t

def inversionTimeDenseToSparse(meanModel):
    model = meanModel.copy()
    start = time.clock()
    scipy.sparse.csc_matrix(np.linalg.inv(model.m))
    end = time.clock()
    t = end -start
    #print(t)
    return t

#This happens in conversion from mean to canonical Form, calculating the information vector
def sparseMatrixDotVectorWithScipySparse(canonicalModel):
    model = canonicalModel.copy()
    start = time.clock()
    result = model.m.dot(model.v)
    end = time.clock()
    t = end -start
    #print(t)
    return t
#This is a test, what happens if numpy dot product gets an csc_matrix
def sparseMatrixDotVectorWithNumpy(canonicalModel):
    model = canonicalModel.copy()
    start = time.clock()
    result = np.dot(model.m,model.v)
    end = time.clock()
    t = end -start
    return t

def sparseMatrixDotMatrix(canonicalModel):
    model = canonicalModel.copy()
    start = time.clock()
    result = model.m.dot(model.m)
    end = time.clock()
    t = end -start
    return t

def denseMatrixDotMatrix(meanModel):
    model = meanModel.copy()
    start = time.clock()
    result = model.m.dot(model.m)
    end = time.clock()
    t = end -start
    return t

def denseMatrixDotVector(meanModel):
    model = meanModel.copy()
    start = time.clock()
    result = np.dot(model.m, model.v)
    end = time.clock()
    t = end -start
    #print(t)
    return t

def denseMatrixDotVectorFlattened(meanModel):
    model = meanModel.copy()
    start = time.clock()
    result = np.squeeze(np.asarray(np.dot(model.m, model.v)))
    end = time.clock()
    t = end - start
    #print(t)
    return t

def indexListComprehension(meanModel):
    model = meanModel.copy()
    start = time.clock()
    
    end = time.clock()
    t = end -start
    #print(t)
    return t

def sparseSubsetFI(model):
    indexSet = list(range(0, math.floor(model.m.shape[0]/2)))

    start = time.clock()
    result = model.m[:, indexSet][indexSet, :]
    result + result.T
    end = time.clock()

    mstart = time.clock()
    result + result.T
    mend = time.clock()
    mtime = mend - mstart

    t =  end - start - mtime
    return(t)

def sparseSubsetIX(model):
    indexSet = list(range(0, math.floor(model.m.shape[0]/2)))

    start = time.clock()
    result3 = model.m[np.ix_(indexSet,indexSet)]
    result3 + result3.T
    end = time.clock()

    mstart = time.clock()
    result3 + result3.T
    mend = time.clock()

    mtime = mend - mstart
    t = end - start -mtime
    return(t)

#Operations

def marginalizeCanonicalConditionCanonical(model):

    start = time.clock()
    #print(model.m.shape)
    marginalized = model.marginalize([0, 2, 4, 5,8 ,9])
    #print(marginalized.m.shape)
    conditioned = marginalized.condition([1, 3], np.random.rand(2))
    #print(conditioned.m.shape)
    end = time.clock()
    t = end -start
    #print('mccc: '+ str(t))
    return t




#Constrained Cases

def marginalizeMeanConditionCanonicalConvert(model):
    model.meanForm()

    start = time.clock()
    #print(model.m.shape)
    marginalized = model.marginalize([0, 2, 4, 5,8 ,9])

    model.canonicalForm()
    #print(marginalized.m.shape)
    conditioned = marginalized.condition([1, 3], np.random.rand(2))
    #print(conditioned.m.shape)

    end = time.clock()

    t = end -start
    #print('mccc: '+ str(t))
    return t

def ConvertMarginalizeMeanConditionCanonical(model):
    model.canonicalForm()

    start = time.clock()
    model.meanForm()
    #print(model.m.shape)
    marginalized = model.marginalize([0, 2, 4, 5,8 ,9])
    
    model.canonicalForm()
    #print(marginalized.m.shape)
    conditioned = marginalized.condition([1, 3], np.random.rand(2))
    #print(conditioned.m.shape)

    end = time.clock()

    t = end -start
    #print('mccc: '+ str(t))
    return t









def marginalizeCanonicalConditionMean(model):
    #print(model.m.shape)
    start = time.clock()
    marginalized = model.marginalize([0, 2, 4, 5,8 ,9])
    #print(marginalized.m.shape)
    marginalized.meanForm()
    #print(marginalized.m.shape)
    conditioned = marginalized.condition([1, 3], np.random.rand(2))
    #print(conditioned.m.shape)
    end = time.clock()
    t = end - start
    #print('mccm: '+ str(t))
    return t

def marginalizeMeanConditionMean(model):
    #print(model.m.shape)
    model.meanForm()
    start = time.clock()
    marginalized = model.marginalize([0, 2, 4, 5,8 ,9])
    #print(marginalized.m.shape)
    conditioned = marginalized.condition([1, 3], np.random.rand(2))
    #print(conditioned.m.shape)
    end = time.clock()

    t = end - start
    #print('mmcm: '+ str(t))
    return t

def marginalizeMeanConditionCanonical(model):
    #print(model.m.shape)
    model.meanForm()
    start = time.clock()
    marginalized = model.marginalize([0, 2, 4, 5,8 ,9])
    #print(marginalized.m.shape)
    marginalized.canonicalForm()
    #print(marginalized.m.shape)
    conditioned = marginalized.condition([1, 3], np.random.rand(2))
    #print(conditioned.m.shape)
    end = time.clock()
    t = end -start
    return t



#Time Condition only
give=0.5
def conditionOnlyMean(model):
    model.meanForm()
    dims = model.m.shape[0]
    conditionTo=list(range(0,round(give*dims)))
    conditionVals= np.random.rand(len(conditionTo))
    start = time.clock()
    result = model.condition(conditionTo,conditionVals)
    end = time.clock()
    t = end -start
    return t

def conditionOnlyCanonical(model):
    dims = model.m.shape[0]
    conditionTo=list(range(0,round(give*dims)))
    conditionVals= np.random.rand(len(conditionTo))
    start = time.clock()
    result = model.condition(conditionTo,conditionVals)
    end = time.clock()
    t = end -start
    return t


