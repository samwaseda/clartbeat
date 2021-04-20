import numpy as np
from sklearn.decomposition import PCA

class MyPCA(PCA):
    def f(self, phi):
        x = np.atleast_1d(phi)
        cossin = np.stack((np.cos(x), np.sin(x)), axis=-1)
        xx = np.einsum(
            'ij,ni,i->nj',
            self.components_,
            cossin,
            2*np.sqrt(self.explained_variance_)
        )
        return np.squeeze(xx+self.mean_)
