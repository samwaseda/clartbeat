import numpy as np
from sklearn.decomposition import PCA

class MyPCA(PCA):
    def get_relative_points(self, points):
        return np.einsum(
            'ij,nj->ni',
            self.pca.components_,
            points-self.pca.mean_
        )
