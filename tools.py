import numpy as np

def _get_slope(x, x_interval):
    x0 = np.mean(x_interval)
    dx = np.diff(x_interval)[0]/2
    return 1/(1+np.exp(-(x-x0)*np.log(10)/dx))

def get_slope(x, x_interval, symmetric=False):
    if symmetric:
        return _get_slope(x, x_interval)*_get_slope(x, x_interval[::-1])
    return _get_slope(x, x_interval)

def damp(x, x0=0, length=1, slope=1):
    xx = slope*(x-x0)/length*2
    return length*(1-np.exp(-xx))/(1+np.exp(-xx))

def get_relative_coordinates(x, v):
    v = np.array(v).reshape(-1, 2)
    x = np.array((-1,)+v.shape)
    v = v/np.linalg.norm(v, axis=-1)[:,None]
    v /= np.linalg.norm(v, axis=-1)[:,None]
    v = np.stack((v, np.einsum('ij,nj->ni', [[0, 1], [-1, 0]], v)), axis=-1)
    return np.squeeze(np.einsum('kij,nkj->nki', v, x))
