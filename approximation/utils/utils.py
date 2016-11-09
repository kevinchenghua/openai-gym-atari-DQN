import numpy as np

def ortho_weight(ndim):
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = np.random.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')
    
def norm_weight_4d(d1, d2, d3, d4, scale=0.01):
    """
    Random weights drawn from a Gaussian
    """
    W = scale * np.random.randn(d1,d2,d3,d4)
    return W.astype('float32')

def zero_bias(dim):
    b = np.zeros(dim)
    return b.astype('float32')