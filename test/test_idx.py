import numpy as np
from scipy.spatial import distance
import itertools
from imgcmp import misc

def test_pdist():
    # play with pdist, which returns the upper triangle of the full squareform
    # distance matrix, test our indexing routines (thank you
    # http://stackoverflow.com)
    nvec = 100
    dim = 10
    triu_list = zip(*np.triu_indices(nvec, k=1))
    vecs = np.random.rand(nvec, dim)
    dist1d = distance.pdist(vecs)
    assert len(triu_list) == len(dist1d)
    dist2d = distance.squareform(dist1d)
    for ii,jj in itertools.product(range(nvec), repeat=2):
        if ii < jj:
            idx_1d = misc.idx_square2cond(ii, jj, nvec)
            assert dist1d[idx_1d] == dist2d[ii,jj]
            assert misc.idx_cond2square(idx_1d, nvec) == (ii,jj)
            assert triu_list[idx_1d] == (ii,jj)
