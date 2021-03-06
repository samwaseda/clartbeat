import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

class MyPCA(PCA):
    def get_relative_points(self, points):
        return np.einsum(
            'ij,i,nj->ni',
            self.components_,
            0.5/np.sqrt(self.explained_variance_),
            np.asarray(points).reshape(-1, 2)-self.mean_
        ).reshape(np.shape(points))

    def get_scaled_distance(self, points):
        return np.linalg.norm(self.get_relative_points(points=points), axis=-1)

    def get_absolute_points(self, points):
        return np.einsum(
            'ji,j,nj->ni',
            self.components_,
            2*np.sqrt(self.explained_variance_),
            np.asarray(points).reshape(-1, 2)
        ).reshape(np.shape(points))+self.mean_

    def get_principal_vectors(self, normalized=False):
        if normalized:
            return self.components_
        return np.einsum(
            'i,ij->ij', np.sqrt(self.explained_variance_)*2, self.components_
        )

def _get_slope(x, x_interval):
    x0 = np.mean(x_interval)
    dx = np.diff(x_interval)[0]/2
    return 1/(1+np.exp(-(x-x0)*np.log(10)/dx))

def get_softplus(x, slope=1):
    return np.log10(1+10**(slope*x))

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

def find_common_labels(unique_labels, labels):
    dist, _ = cKDTree(
        np.unique(unique_labels).reshape(-1, 1)
    ).query(labels.reshape(-1, 1))
    return dist == 0

def large_chunk(labels, min_fraction, keep_noise=False, f=np.max):
    unique_labels, counts = np.unique(labels, return_counts=True)
    if not keep_noise:
        counts = counts[unique_labels!=-1]
        unique_labels = unique_labels[unique_labels!=-1]
    unique_labels = unique_labels[counts > min_fraction*f(counts)]
    return find_common_labels(unique_labels, labels)

def get_extrema(values, maximum=True):
    if maximum:
        cond = values > np.max([np.roll(values, 1), np.roll(values, -1)], axis=0)
    else:
        cond = values < np.min([np.roll(values, 1), np.roll(values, -1)], axis=0)
    cond[[0,-1]] = False
    return cond

def abridge(condition, *args):
    return (a[condition] for a in args)

