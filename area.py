import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

class Area:
    def __init__(self, points):
        if points is None:
            return None
        self.points = points
        self._initialize_pca()
        self._delaunay = None

    @property
    def delaunay(self):
        if self._delaunay is None:
            self.run_delaunay()
        return self._delaunay

    def run_delaunay(self):
        self._delaunay = Delaunay(self.points)

    def _initialize_pca(self):
        self.pca = PCA().fit(self.points)
        
    def get_principal_vectors(self, normalized=False):
        if normalized:
            return self.pca.components_
        return np.einsum('i,ij->ij', np.sqrt(self.pca.explained_variance_)*2, self.pca.components_)

    def get_length(self, reduced=True):
        if reduced:
            return np.sqrt(self.pca.explained_variance_)*2
        else:
            x = np.einsum('ij,nj->ni', self.pca.components_, self.points)
            return x.max(axis=0)-x.min(axis=0)

    def get_relative_points(self, points=None):
        if points is None:
            points = self.points
        return np.einsum(
            'ij,nj->ni',
            self.pca.components_,
            points-self.pca.mean_
        )

    def get_scaled_distance(self, points=None):
        r = self.get_relative_points(points=points)*0.5/np.sqrt(self.pca.explained_variance_)
        return np.linalg.norm(r, axis=-1)
        
    def get_center(self, mean_f=np.mean, ref_point=None, max_diff=0.01):
        mean_point = mean_f(self.points, axis=0)
        if ref_point is None:
            return mean_f(self.points, axis=0)
        ref_v = mean_point-ref_point
        ref_v /= np.linalg.norm(ref_v)
        delta_r = self.points-ref_point
        angle = np.einsum('ni,i,n->n', delta_r, ref_v, 1/np.linalg.norm(delta_r, axis=-1))
        return mean_f(self.points[angle>1-max_diff], axis=0)
    
    def _get_internal_triangles(self, forbidden_triangles):
        candidate = np.any(self.delaunay.neighbors==-1, axis=-1)
        to_delete = candidate*forbidden_triangles
        for _ in range(100):
            candidate = np.unique(self.delaunay.neighbors[to_delete])
            candidate = candidate[candidate!=-1]
            candidate = candidate[~to_delete[candidate]]
            if np.sum(forbidden_triangles[candidate])==0:
                break
            to_delete[candidate[forbidden_triangles[candidate]]] = True
        return ~to_delete

    def _get_forbidden_triangles(self, max_distance=10):
        d = self.points[self.delaunay.simplices]
        d = d-np.roll(d, 1, axis=-2)
        d = np.linalg.norm(d, axis=-1).max(axis=-1)
        return d>max_distance

    def get_delaunay_triangles(self, max_distance=10, keep_intern=True):
        cond = self._get_forbidden_triangles(max_distance=max_distance)
        if keep_intern:
            return self.delaunay.simplices[self._get_internal_triangles(cond)]
        else:
            return self.delaunay.simplices[~cond]

    def get_points(self, reduced=True):
        x = self.points.copy()
        if reduced:
            x = x[self.get_scaled_distance()<1]
        return x

    @property
    def hull(self):
        if self._hull is None:
            self.hull = Surface(ConvexHull(self.points))
        return self._hull

    def get_delaunay_vertices(self, max_distance=5, cluster=True):
        forbidden_triangles = self._get_forbidden_triangles(
            max_distance=max_distance
        )
        triangles = self._get_internal_triangles(forbidden_triangles)
        indices = np.where(forbidden_triangles)[0]
        neighbors = self.delaunay.neighbors.copy().flatten()
        neighbors[np.any(neighbors[:,None]==indices[None,:], axis=1)] = -1
        neighbors = neighbors.reshape(-1, 3)
        edge_indices = neighbors[np.any(neighbors==-1, axis=-1)]
        edge_indices = self.delaunay.simplices[edge_indices[edge_indices!=-1]]
        edge_indices = np.unique(edge_indices)
        if not cluster:
            return edge_indices
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=max_distance, linkage='single'
        ).fit(self.points[edge_indices])
        unique, counts = np.unique(cluster.labels_, return_counts=True)
        return edge_indices[cluster.labels_==unique[counts.argmax()]]

    def get_volume(self, mode='hull', reduced=True, max_distance=10, keep_intern=True):
        if mode=='hull':
            x = self.get_points(reduced=reduced)
            return ConvexHull(x).volume
        elif mode=='delaunay':
            y = self.points[self.get_delaunay_triangles(max_distance=max_distance, keep_intern=keep_intern)]
            y = y[:,1:]-y[:,:1]
            y = np.absolute(np.cross(y[:,0], y[:,1]))
            return np.sum(y)/2
        elif mode=='points':
            return len(self.points)
        else:
            raise ValueError('mode not recognized')

class Surface:
    def __init__(self, x):
        self._x = x.copy()
        self._delaunay = None

    @property
    def x(self):
        return self._x

    @property
    def surface(self):
        return np.linalg.norm(self.x-np.roll(x, 1, axis=0), axis=-1).sum()

    @property
    def volume(self):
        y = self.x-np.roll(x, 1, axis=0)
        return np.absolute(np.sum(np.cross(y[:,0], y[:,1])))/2

