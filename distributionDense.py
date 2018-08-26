import numpy as np
import copy as cp

class DistributionDense():
    """The Distribution Class represents a multivariate Gaussian Distribution"""
    def __init__(self, v, m, type):
        super(DistributionDense, self).__init__()
        self.type = type
        self.m = m
        self.v = v
        
    def copy(self):
        return DistributionDense(np.copy(self.v), np.copy(self.m), cp.copy(self.type))

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
        return(self.m[self.m==0].size / self.m.size)

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
            self.m = np.linalg.inv(self.m)
            self.v = np.squeeze(np.asarray(self.m.dot(self.v)))
            self.type = 'canonical'

    def meanForm(self):
        if self.type == 'canonical':
            self.m = np.linalg.inv(self.m)
            self.v = np.squeeze(np.asarray(self.m.dot(self.v)))
            self.type = 'mean'


    def marginalize(self, indexSet):
        wholeSet = set(range(0, self.m.shape[0]))
        complementarySet = list(wholeSet - set(indexSet))


        # Calculate p(x_indexSet)
        if self.type == 'mean':
            m_new = np.take(np.take(self.m, indexSet, axis=0), indexSet, axis=1)
            v_new = np.take(self.v,indexSet)
            return DistributionDense(v_new, m_new, 'mean')


        # Calculate p(x_indexSet)
        else:
            m_11 = np.take(np.take(self.m, indexSet, axis=0), indexSet, axis=1)
            m_22_i = np.linalg.inv(np.take(np.take(self.m, complementarySet, axis=0), complementarySet, axis=1))
            m_12 = np.take(np.take(self.m, indexSet, axis=0), complementarySet, axis=1)
            m_21 = m_12.T

            m_new = np.subtract(m_11, m_12.dot(m_22_i).dot(m_21))
            #v_new = np.subtract(np.take(self.v,indexSet), m_12.dot(m_22_i).dot(np.take(self.v, complementarySet).T).T).T
            v_new = np.squeeze(np.asarray(np.subtract(np.take(self.v, indexSet), m_12.dot(m_22_i).dot(np.take(self.v, complementarySet).T).T)))
            return DistributionDense(v_new, m_new, 'canonical')

    def condition(self, indexSet, values):
        wholeSet = set(range(0, self.m.shape[0]))
        complementarySet = list(wholeSet - set(indexSet))

        #Calculate p(x_complementarySet | x_indexSet)
        if self.type == 'mean':
            m_11_i = np.linalg.inv(np.take(np.take(self.m, indexSet, axis=0), indexSet, axis=1))
            m_22 = np.take(np.take(self.m, complementarySet, axis=0), complementarySet, axis=1)
            m_12 = np.take(np.take(self.m, indexSet, axis=0), complementarySet, axis=1)
            m_21 = m_12.T

            v_new = np.squeeze(np.asarray(np.subtract(np.take(self.v, complementarySet), m_21.dot(m_11_i).dot(np.subtract(values,np.take(self.v,indexSet))))))


            m_new = np.subtract(m_22, m_21.dot(m_11_i).dot(m_12))

            return DistributionDense(v_new, m_new, 'mean')

        #Calculate p(x_complementarySet | x_indexSet)
        else:
            m_22 = np.take(np.take(self.m, complementarySet, axis=0), complementarySet, axis=1)
            m_21 = np.take(np.take(self.m, complementarySet, axis=0), indexSet, axis=1)
            v_new = np.squeeze(np.asarray(np.take(self.v, complementarySet) - m_21.dot(values)))
            return DistributionDense(v_new, m_22, 'canonical')
