import numpy as np
import scipy.sparse
import scipy.sparse.linalg

dim =5000
m = scipy.sparse.csc_matrix(np.random.rand(dim)+np.eye(dim))
i = scipy.sparse.linalg.inv(m)

