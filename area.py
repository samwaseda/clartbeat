import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import AgglomerativeClustering
from tools import MyPCA
from surface import Surface

class Area:
    def __init__(self, points, perimeter=None):
        self.points = points
        if points is None:
            return None
        self._delaunay_class = None
        self._delaunay = None
        self._hull = None
        self._perimeter = None
        if perimeter is not None:
            self._perimeter = perimeter
        if len(points)==0:
            return
        self._initialize_pca()

    @property
    def perimeter(self):
        return self._perimeter

    @property
    def delaunay(self):
        if self._delaunay is None:
            self._delaunay = Surface(self.points[self.get_delaunay_vertices()])
        return self._delaunay

    @property
    def delaunay_class(self):
        if self._delaunay_class is None:
            self._delaunay_class = Delaunay(self.points)
        return self._delaunay_class

    def _initialize_pca(self):
        self.pca = MyPCA().fit(self.points)
        
    def get_length(self, reduced=True):
        if reduced:
            return np.sqrt(self.pca.explained_variance_)*2
        else:
            x = np.einsum('ij,nj->ni', self.pca.components_, self.points)
            return x.max(axis=0)-x.min(axis=0)

    def get_canvas(self, shape):
        img = np.zeros(shape)
        img[self.points[:,0], self.points[:,1]] = 1
        return img

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
        candidate = np.any(self.delaunay_class.neighbors==-1, axis=-1)
        to_delete = candidate*forbidden_triangles
        for _ in range(100):
            candidate = np.unique(self.delaunay_class.neighbors[to_delete])
            candidate = candidate[candidate!=-1]
            candidate = candidate[~to_delete[candidate]]
            if np.sum(forbidden_triangles[candidate])==0:
                break
            to_delete[candidate[forbidden_triangles[candidate]]] = True
        return ~to_delete

    def _get_forbidden_triangles(self, max_distance=10):
        d = self.points[self.delaunay_class.simplices]
        d = d-np.roll(d, 1, axis=-2)
        d = np.linalg.norm(d, axis=-1).max(axis=-1)
        return d>max_distance

    def get_delaunay_triangles(self, max_distance=10, keep_intern=True):
        cond = self._get_forbidden_triangles(max_distance=max_distance)
        if keep_intern:
            return self.delaunay_class.simplices[self._get_internal_triangles(cond)]
        else:
            return self.delaunay_class.simplices[~cond]

    @property
    def hull(self):
        if self._hull is None:
            vertices = ConvexHull(self.points).vertices
            self._hull = Surface(self.points[vertices])
        return self._hull

    def get_delaunay_vertices(self, max_distance=5, cluster=True):
        forbidden_triangles = self._get_forbidden_triangles(
            max_distance=max_distance
        )
        triangles = self._get_internal_triangles(forbidden_triangles)
        indices = np.where(forbidden_triangles)[0]
        neighbors = self.delaunay_class.neighbors.copy().flatten()
        neighbors[np.any(neighbors[:,None]==indices[None,:], axis=1)] = -1
        neighbors = neighbors.reshape(-1, 3)
        edge_indices = neighbors[np.any(neighbors==-1, axis=-1)]
        edge_indices = self.delaunay_class.simplices[edge_indices[edge_indices!=-1]]
        edge_indices = np.unique(edge_indices)
        if not cluster:
            return edge_indices
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=max_distance, linkage='single'
        ).fit(self.points[edge_indices])
        unique, counts = np.unique(cluster.labels_, return_counts=True)
        return edge_indices[cluster.labels_==unique[counts.argmax()]]

    def get_relative_angle(self, ref_center):
        y = self.get_center()-ref_center
        return np.arctan2(y[1], y[0])

    def get_polar_coordinates(self, ref_center):
        x = self.points-ref_center
        angle = self.get_relative_angle(ref_center=ref_center)
        r = np.linalg.norm(x, axis=-1)
        if r.min()==0:
            raise ValueError('Error calculating polar coordinates')
        phi = np.mod(np.arctan2(x[:,1], x[:,0])-angle+np.pi, 2*np.pi)-np.pi
        return r, phi

    def trace_pca(
        self, ref_center, polynomial_order=None, angle_threshold=0.6, number_of_points=360
    ):
        r, phi = self.get_polar_coordinates(ref_center=ref_center)
        if polynomial_order is None:
            if phi.ptp() > angle_threshold:
                polynomial_order = 3
            else:
                polynomial_order = 1
        coeff = np.polyfit(phi, r, polynomial_order)
        all_angle = np.linspace(0, 2*np.pi, number_of_points)
        pca = MyPCA().fit(np.stack((phi, r-np.polyval(coeff, phi)), axis=-1))
        phi_range = phi.min()+(phi.max()-phi.min())*(np.cos(all_angle)+1)/2
        r_fit = np.stack((phi_range, np.polyval(coeff, phi_range)), axis=-1)
        r_fit[:,1] += pca.get_absolute_points(
            np.stack((np.sin(all_angle), np.cos(all_angle)), axis=-1)
        )[:,1]
        r_fit[:,0] += self.get_relative_angle(ref_center=ref_center)
        xy_fit = np.stack(
            (r_fit[:,1]*np.cos(r_fit[:,0]), r_fit[:,1]*np.sin(r_fit[:,0])), axis=-1
        )
        return xy_fit+ref_center

    def _get_delaunay_volume(self, max_distance=10, keep_intern=True):
        t = self.points[
            self.get_delaunay_triangles(max_distance=max_distance, keep_intern=keep_intern)
        ]
        t = t[:,1:]-t[:,:1]
        return np.cross(t[:,0], t[:,1]).sum()/2

    def get_volume(self, mode='hull', reduced=True, max_distance=10, keep_intern=True):
        if self.points is None:
            return 0
        elif mode=='hull':
            return self.hull.volume
        elif mode=='delaunay':
            return self._get_delaunay_volume(max_distance=max_distance, keep_intern=keep_intern)
        elif mode=='points':
            return len(self.points)
        elif mode=='pca':
            return np.prod(self.get_length())*np.pi
        else:
            raise ValueError('mode not recognized')

