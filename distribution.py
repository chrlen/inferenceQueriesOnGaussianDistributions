import numpy as np
import copy as cp
import scipy
import scipy.sparse
import scipy.sparse.linalg


class Distribution():
    """The Distribution Class represents a multivariate Gaussian Distribution"""
    def __init__(self, v, m, type):
        super(Distribution, self).__init__()
        self.type = type
        self.m = m
        self.v = v

    def copy(self):
        if self.type=='mean':
            return Distribution(np.copy(self.v), np.copy(self.m), cp.copy(self.type))
        else:
            return Distribution(np.copy(self.v), self.m.copy(), cp.copy(self.type))

    def equals(self, distribution):
        return (self.type == distribution.type) and np.all(np.equal(self.v, distribution.v)) and np.all(np.equal(self.m, distribution.m))

    def equalsVerbose(self, dist):
        if self.type != dist.type:
            print("not self.type != dist.type")
            return False
        if not np.all(np.equal(self.v, dist.v)):
            print("not np.all(np.equal(self.v, dist.v))")
            print(np.equal(self.v, dist.v))
            return False
        if not np.all(np.equal(self.m, dist.m)):
            print("not np.all(np.equal(self.m, dist.m))")
            return False
        return True

    def isCloseVerbose(self,dist):
        if self.type != dist.type:
            print("not self.type != dist.type")
            return False
        if not np.allclose(self.v, dist.v):
            print("not np.all(np.equal(self.v, dist.v))")
            print(np.equal(self.v, dist.v))
            return False
        if not np.allclose(self.m, dist.m):
            print("not np.all(np.equal(self.m, dist.m))")
            return False
        return True
            # return (self.type == dist.type) and np.all(np.equal(self.v, dist.v)) and np.all(np.equal(self.m, dist.m))

    def sparsity(self):
        if(self.type == 'canonical'):
            d = self.m.todense()
            return(d[d==0].size / d.size)

    def print(self):
        print("Type:" + self.type)
        print("Dims: " + str(self.m.shape[0]))
        print("V=" + str(self.v))
        print("v.shape: " + str(self.v.shape))
        print("type(self.v): " + str(type(self.v)))
        print("V=" + str(self.m))
        print("m.shape: " + str(self.m.shape))
        print("type(self.m): " + str(type(self.m)))

    def canonicalForm(self):
        if self.type == 'mean':
            self.m = scipy.sparse.csc_matrix(np.linalg.inv(self.m))
            self.v = self.m.dot(self.v)
            self.type = 'canonical'

    def meanForm(self):
        if self.type == 'canonical':
            self.m = scipy.sparse.linalg.inv(self.m)
            self.m = self.m.todense()
            self.v = np.squeeze(np.asarray(self.m.dot(self.v)))
            self.type = 'mean'

    def marginalize(self, indexSet):
        wholeSet = set(range(0, self.m.shape[0]))
        complementarySet = list(wholeSet - set(indexSet))

        # Calculate p(x_indexSet)
        if self.type == 'mean':
            m_new = np.take(np.take(self.m, indexSet, axis=0), indexSet, axis=1)
            v_new = np.take(self.v,indexSet)
            return Distribution(v_new, m_new, 'mean')

        # Calculate p(x_indexSet)
        else:
            m_11 = self.m[indexSet, :][:, indexSet]
            m_22_i = scipy.sparse.linalg.inv(
                self.m[complementarySet, :][:, complementarySet]
                )
            m_12 = self.m[indexSet, :][:, complementarySet]
            m_21 = m_12.T

            m_new = np.subtract(m_11, m_12.dot(m_22_i).dot(m_21))
            v_new = np.squeeze(np.asarray(
                np.subtract(np.take(self.v, indexSet),
                            m_12.dot(m_22_i).dot(
                                np.take(self.v, complementarySet).T).T)
                ))
            return Distribution(v_new, m_new, 'canonical')

    def condition(self, indexSet, values):
        wholeSet = set(range(0, self.m.shape[0]))
        complementarySet = list(wholeSet - set(indexSet))

        #Calculate p(x_complementarySet | x_indexSet)
        if self.type == 'mean':
            m_11_i = np.linalg.inv(
                np.take(np.take(self.m, indexSet, axis=0), indexSet, axis=1)
                )
            m_22 = np.take(
                np.take(self.m, complementarySet, axis=0),
                           complementarySet, axis=1)
            m_12 = np.take(
                np.take(self.m, indexSet, axis=0),
                complementarySet, axis=1)
            m_21 = m_12.T

            v_new = np.squeeze(np.asarray(
                np.subtract(
                    np.take(self.v, complementarySet),
                    m_21.dot(m_11_i).dot(np.subtract(
                        values,
                        np.take(self.v,indexSet)))
                    )
                ))

            m_new = np.subtract(m_22, m_21.dot(m_11_i).dot(m_12))

            return Distribution(v_new, m_new, 'mean')

        #Calculate p(x_complementarySet | x_indexSet)
        else:
            m_22 = self.m[complementarySet, :][:, complementarySet]
            m_21 = self.m[complementarySet, :][:, indexSet]
            v_new = np.squeeze(np.asarray(np.take(self.v, complementarySet) - m_21.dot(values)))
            return Distribution(v_new, m_22, 'canonical')

def meanDistFromTuple(t):
    return Distribution(t[0],t[1],'mean')

import helpers as hlp

def estimateCanonicalFromData(data,alpha):
    #print("Estimate Canonical Distribution, Dims: " + str(data.shape[1])+' alpha: ' + str(alpha))
    m = hlp.estimatePrecisionFromSet(data,alpha)
    v = scipy.sparse.linalg.inv(m).dot(data.mean(0)).T
    return Distribution(v, m ,'canonical')
