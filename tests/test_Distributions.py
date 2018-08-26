import helpers as hlp
import distribution as dst
import distributionDense as ddst
import numpy as np
import scipy.sparse as sp
import pickle
import timing

dims = 50


def test_copy():
    sigma = hlp.randomPositiveSemidefinite(dims)
    mu = np.random.rand(dims)
    d = dst.Distribution(mu, sigma, 'moment')
    c = d.copy()
    # Check if different objects
    assert c is not d
    # Check if entities hold equal values
    assert c.equalsVerbose(d)
    # Check if deep-copy succeeded
    assert c.v is not d.v
    # Check if deep-copy succeeded
    assert c.m is not d.m

def test_Conversion():
    sigma = hlp.randomPositiveSemidefinite(dims)
    mu = np.random.rand(dims).T
    d = dst.Distribution(mu, sigma, 'mean')
    dd = ddst.DistributionDense(mu, sigma, 'mean')
    print(type(dd))

    a = d.copy()
    b = dd.copy()
    # Check if multiple conversions don't change equality
    for i in range(1, 100):
        #print(i)
        a.canonicalForm()
        a.meanForm()

        b.canonicalForm()
        b.meanForm()
    assert a.isCloseVerbose(d)
    assert b.isCloseVerbose(dd)

def test_operationsSparse():
    marginalizeTo = [1, 2, 4, 5]
    condition = [0, 1]
    condVal = [0.5, 0.5]

    cov = np.matrix([[3, 0, 0, 2, 0, 2],
                     [0, 1, 0.5, 0, 0, 0],
                     [0, 0.5, 1, 0, 0, 0],
                     [2, 0, 0, 2, 0, 1],
                     [0, 0, 0, 0, 1, 0],
                     [2, 0, 0, 1, 0, 2]])

    mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # Invert matrix and compute information vector
    prec = np.linalg.inv(cov)
    inf = prec.dot(mean)

    # Construct Distribution objects
    sparse_dist = dst.Distribution(mean, cov, 'mean')
    sparse_dist_canonical = dst.Distribution(inf, sp.csc_matrix(prec), 'canonical')

    # Marginalize "by hand"
    cov_marginalized = np.take(np.take(cov, marginalizeTo, axis=0), marginalizeTo, axis=1)
    mean_marginalized = np.take(mean, marginalizeTo)

    # Marginalize both distributions with methods
    sparse_dist_marginalized = sparse_dist.marginalize(marginalizeTo)
    sparse_dist_canonical_marginalized = sparse_dist_canonical.marginalize(marginalizeTo)

    # models are close
    assert(np.allclose(sparse_dist_marginalized.m, cov_marginalized))
    assert(np.allclose(sparse_dist_marginalized.v, mean_marginalized))


    #Covert canonical to mean and check if close
    temp = sparse_dist_canonical_marginalized.copy()
    temp.meanForm()
    assert(np.allclose(temp.m, sparse_dist_marginalized.m))
    assert(np.allclose(temp.v, sparse_dist_marginalized.v))


    #Compute canonical 'by hand', a little bit redundant^^
    prec_marginalized = np.linalg.inv(cov_marginalized)
    inf_marginalized = prec_marginalized.dot(mean_marginalized)
    assert(np.allclose(prec_marginalized, sparse_dist_canonical_marginalized.m.todense()))
    assert(np.allclose(inf_marginalized, sparse_dist_canonical_marginalized.v))



    # Compute conditioned distribution by hand
    complementary_conditioning_set = list(set(range(0, prec_marginalized.shape[0])) - set(condition))
    
    prec_final = np.take(np.take(prec_marginalized,complementary_conditioning_set,axis=0),complementary_conditioning_set,axis=1)

    l_ji = np.take(np.take(prec_marginalized,complementary_conditioning_set,axis=0),condition,axis=1)
    l_jj_i = np.linalg.inv(np.take(np.take(prec_marginalized,complementary_conditioning_set,axis=0),complementary_conditioning_set,axis=1))

    inf_final = np.squeeze(np.asarray(np.subtract(np.take(inf_marginalized,complementary_conditioning_set),l_ji.dot(l_jj_i).dot(condVal))))

    cov_final = np.linalg.inv(prec_final)
    mean_final = cov_final.dot(inf_final.T)
    

    # Check conditioning on canonical form   
    canonicalMarginalizedConditioned = sparse_dist_canonical_marginalized.condition(condition,condVal)
    assert(np.allclose(prec_final, canonicalMarginalizedConditioned.m.todense()))
    assert(np.allclose(inf_final, canonicalMarginalizedConditioned.v))


    # Check conditioning on mean form
    sparse_dist_marginalized_conditioned = sparse_dist_marginalized.condition(condition,condVal)
    print('--------------------------------------')
    print(cov_final)
    print('--------------------------------------')
    print(sparse_dist_marginalized_conditioned.m)
    print('--------------------------------------')
    print(mean_final)
    print('--------------------------------------')
    print(sparse_dist_marginalized_conditioned.v)



    assert(np.allclose(cov_final, sparse_dist_marginalized_conditioned.m))
    assert(np.allclose(mean_final, sparse_dist_marginalized_conditioned.v))


    #Convert to canonical Form and check again
    sparse_dist_marginalized_conditioned.canonicalForm()

    assert(np.allclose(prec_final, sparse_dist_marginalized_conditioned.m.todense()))
    assert(np.allclose(inf_final, sparse_dist_marginalized_conditioned.v))




def test_operationsDense():
    marginalizeTo = [1, 2, 4, 5]
    condition = [0, 1]
    condVal = [0.5, 0.5]

    cov = np.matrix([[3, 0, 0, 2, 0, 2],
                     [0, 1, 0.5, 0, 0, 0],
                     [0, 0.5, 1, 0, 0, 0],
                     [2, 0, 0, 2, 0, 1],
                     [0, 0, 0, 0, 1, 0],
                     [2, 0, 0, 1, 0, 2]])

    mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # Invert matrix and compute information vector
    prec = np.linalg.inv(cov)
    inf = prec.dot(mean)

    # Construct Distribution objects
    sparse_dist = ddst.DistributionDense(mean, cov, 'mean')
    sparse_dist_canonical = ddst.DistributionDense(inf, prec, 'canonical')

    # Marginalize "by hand"
    cov_marginalized = np.take(np.take(cov, marginalizeTo, axis=0), marginalizeTo, axis=1)
    mean_marginalized = np.take(mean, marginalizeTo)

    # Marginalize both distributions with methods
    sparse_dist_marginalized = sparse_dist.marginalize(marginalizeTo)
    sparse_dist_canonical_marginalized = sparse_dist_canonical.marginalize(marginalizeTo)

    # models are close
    assert(np.allclose(sparse_dist_marginalized.m, cov_marginalized))
    assert(np.allclose(sparse_dist_marginalized.v, mean_marginalized))


    #Covert canonical to mean and check if close
    temp = sparse_dist_canonical_marginalized.copy()
    temp.meanForm()
    assert(np.allclose(temp.m, sparse_dist_marginalized.m))
    assert(np.allclose(temp.v, sparse_dist_marginalized.v))


    #Compute canonical 'by hand', a little bit redundant^^
    prec_marginalized = np.linalg.inv(cov_marginalized)
    inf_marginalized = prec_marginalized.dot(mean_marginalized)
    assert(np.allclose(prec_marginalized, sparse_dist_canonical_marginalized.m))
    assert(np.allclose(inf_marginalized, sparse_dist_canonical_marginalized.v))



    # Compute conditioned distribution by hand
    complementary_conditioning_set = list(set(range(0, prec_marginalized.shape[0])) - set(condition))
    
    prec_final = np.take(np.take(prec_marginalized,complementary_conditioning_set,axis=0),complementary_conditioning_set,axis=1)

    l_ji = np.take(np.take(prec_marginalized,complementary_conditioning_set,axis=0),condition,axis=1)
    l_jj_i = np.linalg.inv(np.take(np.take(prec_marginalized,complementary_conditioning_set,axis=0),complementary_conditioning_set,axis=1))

    inf_final = np.squeeze(np.asarray(np.subtract(np.take(inf_marginalized,complementary_conditioning_set),l_ji.dot(l_jj_i).dot(condVal))))

    cov_final = np.linalg.inv(prec_final)
    mean_final = cov_final.dot(inf_final.T)
    

    # Check conditioning on canonical form   
    canonicalMarginalizedConditioned = sparse_dist_canonical_marginalized.condition(condition,condVal)
    assert(np.allclose(prec_final, canonicalMarginalizedConditioned.m))
    assert(np.allclose(inf_final, canonicalMarginalizedConditioned.v))


    # Check conditioning on mean form
    sparse_dist_marginalized_conditioned = sparse_dist_marginalized.condition(condition,condVal)
    print('--------------------------------------')
    print(cov_final)
    print('--------------------------------------')
    print(sparse_dist_marginalized_conditioned.m)
    print('--------------------------------------')
    print(mean_final)
    print('--------------------------------------')
    print(sparse_dist_marginalized_conditioned.v)



    assert(np.allclose(cov_final, sparse_dist_marginalized_conditioned.m))
    assert(np.allclose(mean_final, sparse_dist_marginalized_conditioned.v))


    #Convert to canonical Form and check again
    sparse_dist_marginalized_conditioned.canonicalForm()
    
    assert(np.allclose(prec_final, sparse_dist_marginalized_conditioned.m))
    assert(np.allclose(inf_final, sparse_dist_marginalized_conditioned.v))








