import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from area import Area
import matplotlib.image as mpimg
from scipy.spatial import ConvexHull
import skimage.feature
from skimage import filters
from sklearn.cluster import AgglomerativeClustering
from tools import damp, get_slope

def get_minim_white(img, x_min=400, sigma=4):
    for _ in range(10):
        dx = x_min-img
        exp = np.exp(-dx**2/(2*sigma**2))
        div = (np.sqrt(np.pi)*sigma)
        h = exp.sum()/div
        dhdx = -np.sum(dx*exp)/sigma**2/div
        ddhddx = np.sum((-1/sigma**2+dx**2/sigma**4)*exp)/div
        x_min -= dhdx/np.absolute(ddhddx)
    return x_min

def get_white_color_threshold(img, bins=100, min_value=175):
    norm = img
    if len(norm.shape)==3 and norm.shape[-1]==3:
        norm = np.mean(img, axis=-1)
    if len(norm.shape)!=2:
        raise ValueError('invalid norm shape')
    v = np.rint(np.mean(img, axis=-1))
    v = v[v>min_value]
    l, c = np.unique(v, return_counts=True)
    return l[c.argmin()]

def cleanse_edge(img, erase_edge=10):
    img_new = img.copy()
    img_new[:erase_edge,:,:] = np.array(3*[255])
    img_new[:,:erase_edge,:] = np.array(3*[255])
    img_new[-erase_edge:,:,:] = np.array(3*[255])
    img_new[:,-erase_edge:,:] = np.array(3*[255])
    return img_new

def _clean_noise(img, threshold, eps=5):
    x = np.stack(np.where(np.mean(img, axis=-1)<threshold), axis=-1)
    cluster = DBSCAN(eps=eps).fit(x)
    labels, counts = np.unique(cluster.labels_, return_counts=True)
    y = x[cluster.labels_!=labels[counts.argmax()]]
    img[y[:,0], y[:,1]] = np.array(3*[255])
    return img

def get_edge(img, base, parameters=None):
    if parameters is None:
        parameters = {
            'sigma': 18.684489401039166,
            'low': 6.157785258898054,
            'high': 7.670065207650683
        }
    return skimage.feature.canny(
        image=img,
        sigma=parameters['sigma'],
        low_threshold=parameters['low']*base,
        high_threshold=parameters['high']*base,
    )

def get_local_linear_fit(y_i, x_i, w):
    w = w/w.sum(axis=1)[:,None]
    wx = np.sum(w*x_i, axis=-1)
    wy = np.sum(w*y_i, axis=-1)
    wxx = np.sum(w*x_i**2, axis=-1)
    wxy = np.sum(w*x_i*y_i, axis=-1)
    w = np.sum(w, axis=-1)
    return (w*wxy-wx*wy)/(wxx*w-wx**2), (-wx*wxy+wxx*wy)/(wxx*w-wx**2)

class ProcessImage:
    def __init__(
        self,
        file_name,
        resolution=0.000486,
        target_size=1e6,
        white_color_threshold=None,
        erase_edge=10,
        clean_noise=True,
        stich_high_angles=True,
        elastic_net=True,
    ):
        self.file_name = file_name
        img = mpimg.imread(file_name)
        self._reduction = np.rint(np.sqrt(np.prod(img.shape[:2])/target_size)).astype(int)
        self._reduction = np.max([1, self._reduction])
        img = img[::self._reduction,::self._reduction]
        self.resolution = (resolution*self._reduction)**2
        self._norm = None
        if erase_edge > 0:
            self._img = cleanse_edge(img=img, erase_edge=erase_edge)
        else:
            self._img = img
        if white_color_threshold is None:
            white_color_threshold = get_white_color_threshold(self._img)
        self.white_color_threshold = white_color_threshold
        if clean_noise:
            self._img = _clean_noise(self._img, white_color_threshold)
        self._clustering = {}
        self.cluster = {}
        self.resolution = resolution
        edge = get_edge(self.get_image(mean=True), self.get_base_color()/255)
        self.run_total_area(np.stack(np.where(edge), axis=-1))
        if stich_high_angles:
            self.stich_high_angles()
        if elastic_net:
            self.run_elastic_net()
        self.determine_total_area()

    def run_total_area(
        self,
        points,
        x_range=np.linspace(0, 2*np.pi, 360, endpoint=False),
        sigma=0.05,
        height_unit=40,
        min_fraction=0.05
    ):
        dbscan = DBSCAN(eps=5).fit(points)
        labels, counts = np.unique(dbscan.labels_, return_counts=True)
        labels = labels[counts>min_fraction*len(dbscan.labels_)]
        hull = ConvexHull(points)
        labels = labels[
            np.array([len(set(np.where(l==dbscan.labels_)[0]).intersection(hull.vertices))>0 for l in labels])
        ]
        cond = np.any(labels[:,None]==dbscan.labels_, axis=0)
        mean = np.median(points, axis=0)
        p = points[cond]-mean
        x_i = np.arctan2(p[:,1], p[:,0])
        y_i = np.linalg.norm(p, axis=-1)
        x_i = np.concatenate((x_i-2*np.pi, x_i, x_i+2*np.pi))
        y_i = np.concatenate((y_i, y_i, y_i))
        dist = x_range[:,None]-x_i[None,:]
        dist -= np.rint(dist/np.pi/2)*2*np.pi
        w = np.exp((y_i[None,:]-y_i.mean())/height_unit-dist**2/(2*sigma**2))
        slope, intersection = get_local_linear_fit(y_i, x_i, w)
        xx = (slope*x_range+intersection)*np.cos(x_range)+mean[0]
        yy = (slope*x_range+intersection)*np.sin(x_range)+mean[1]
        self.total_perimeter = np.stack([xx, yy], axis=-1)

    def determine_total_area(self):
        canvas = np.ones_like(self.get_image(mean=True))
        mean = np.mean(self.total_perimeter, axis=0)
        x = canvas*np.arange(canvas.shape[0])[:,None]
        y = canvas*np.arange(canvas.shape[1])
        x -= mean[0]
        y -= mean[1]
        canvas_r = np.sqrt(x**2+y**2)
        canvas_angle = np.arctan2(y, x)
        x = self.total_perimeter-mean
        r = np.linalg.norm(x, axis=-1)
        angle = np.arctan2(x[:,1], x[:,0])
        argmin = np.argmin(np.absolute(canvas_angle[:,:,None]-angle[None,None,:]), axis=-1)
        self.total_area = canvas_r<r[argmin]

    def stich_high_angles(
        self,
        run_all=True,
        sigma_angle=5,
        max_angle=0.045,
    ):
        v = self.total_perimeter.copy()
        v -= np.roll(v, -1, axis=0)
        v_norm = np.linalg.norm(v, axis=-1)
        sin = np.arcsin(np.cross(v, np.roll(v, 1, axis=0))/(v_norm*np.roll(v_norm, 1)))
        sin[0] = sin[-1] = 0
        self._edged_total_perimeter = self.total_perimeter.copy()
        total_number = len(self.total_perimeter)
        high_angles = ndimage.gaussian_filter1d(sin, sigma=sigma_angle)>max_angle
        high_angles = np.arange(len(high_angles))[high_angles]
        if len(high_angles)<2:
            return
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1.1, linkage='single'
        ).fit(high_angles.reshape(-1, 1))
        indices = np.sort([
            np.rint(high_angles[cluster.labels_==l].mean()).astype(int)
            for l in np.unique(cluster.labels_)
        ])
        if len(indices)!=2:
            return
        if np.diff(indices)[0]>0.5*total_number:
            indices = np.roll(indices, 1)
        self.total_perimeter = np.roll(self.total_perimeter, -indices[0], axis=0)
        indices = (np.diff(indices)+total_number)[0]%total_number
        dr = (np.arange(indices)/indices)[:,None]*(
            self.total_perimeter[indices]-self.total_perimeter[0]
        )
        self.total_perimeter[:indices] = dr+self.total_perimeter[0]

    def run_elastic_net(
        self,
        sigma_sobel=15,
        sigma_gauss=5,
        line_tension=0.2,
        dt=0.1,
        max_iter=1000,
        max_gradient=0.1,
    ):
        sobel = filters.sobel(
            ndimage.gaussian_filter(self.get_image(mean=True), sigma=sigma_sobel)
        )
        gauss = 0.01*ndimage.gaussian_filter(self.get_image(mean=True), sigma=sigma_gauss)
        f_sobel_x = sobel-np.roll(sobel, -1, axis=0)
        f_sobel_y = sobel-np.roll(sobel, -1, axis=1)
        f_gauss_x = gauss-np.roll(gauss, -1, axis=0)
        f_gauss_y = gauss-np.roll(gauss, -1, axis=1)
        for i in range(1000):
            f_spring = 2*self.total_perimeter
            f_spring -= np.roll(self.total_perimeter, 1, axis=0)
            f_spring -= np.roll(self.total_perimeter, -1, axis=0)
            f_spring *= line_tension
            p = np.rint(self.total_perimeter).astype(int)
            f_sobel = np.stack((f_sobel_x[p[:,0], p[:,1]], f_sobel_y[p[:,0], p[:,1]]), axis=-1)
            f_gauss = np.stack((f_gauss_x[p[:,0], p[:,1]], f_gauss_y[p[:,0], p[:,1]]), axis=-1)
            f_total = f_spring+f_sobel+f_gauss
            self.total_perimeter -= f_total*dt
            if np.linalg.norm(f_total, axis=-1).max()<max_gradient:
                break

    def get_image(self, mean=False):
        if mean:
            return np.mean(self._img, axis=-1)
        return self._img.copy()

    def get_base_color(self, mean=True):
        mean_color = np.mean(self._img, axis=-1)
        if mean:
            return np.mean(self._img[mean_color<self.white_color_threshold])
        return np.mean(self._img[mean_color<self.white_color_threshold], axis=0)

    def get_original(self, reduction=None):
        img = mpimg.imread(self.file_name)
        if reduction is not None:
            img = img[::int(reduction),::int(reduction)]
        return img

    @property
    def norm(self):
        return self._norm*self.total_area

    @norm.setter
    def norm(self, n):
        self._norm = n

    def apply_gaussian(self, sigma=2):
        norm = np.mean(self._img, axis=-1)
        if sigma > 0:
            norm = ndimage.gaussian_filter(norm, sigma=sigma)
        self.norm = norm

    def apply_mininum(self, size=6):
        norm = np.mean(self._img, axis=-1)
        if size > 0:
            norm = ndimage.minimum_filter(norm, size=size)
        self.norm = norm
        
    def apply_maximum(self, size=1):
        norm = np.mean(self._img, axis=-1)
        if size > 0:
            norm = ndimage.maximum_filter(norm, size=size)
        self.norm = norm

    def apply_median(self, size=1):
        norm = np.mean(self._img, axis=-1)
        if size > 0:
            norm = ndimage.median_filter(norm, size=size)
        self.norm = norm

    def get_area(self, key, smoothened=True):
        if key=='white':
            if smoothened:
                return self.norm > self._threshold
            else:
                return np.mean(self._img, axis=-1) > self._threshold
        elif key=='heart':
            return self.total_area.copy()
        else:
            raise ValueError('key not recognized')

    def _get_max_angle(self, x):
        center = np.stack(np.where(self.total_area), axis=-1).mean(axis=0)
        x = x.copy()-center
        return np.min([np.arctan2(x[:,1], x[:,0]).ptp(), np.arctan2(x[:,1], -x[:,0]).ptp()])

    def _get_biased_coordinates(self, x, bias):
        center = np.stack(np.where(self.total_area), axis=-1).mean(axis=0)
        x = x.copy()-center
        phi = np.arctan2(x[:,1], x[:,0])
        r = np.linalg.norm(x, axis=-1)
        r *= bias[0]
        phi *= bias[1]
        return np.stack((r*np.cos(phi), r*np.sin(phi)), axis=-1)

    def _find_neighbors(self, key, max_dist, indices, indices_to_avoid=None, bias=None, max_angle=45/180*np.pi):
        x = self.cluster[key][indices[0]].copy()
        if self._get_max_angle(x) > max_angle:
            return indices
        if bias is not None: 
            x = self._get_biased_coordinates(x, bias)
        tree = cKDTree(x)
        for ii,xx in enumerate(self.cluster[key]):
            if bias is not None:
                xx = self._get_biased_coordinates(xx, bias)
            if tree.query(xx)[0].min()>max_dist:
                continue
            if indices_to_avoid is not None and ii in indices_to_avoid:
                continue
            indices.append(ii) 
        return indices
        
    def get_index(self, ventricle='left', max_search=5, max_dist=5, bias=np.ones(2)):
        cluster = self.cluster['white'][:max_search]
        if ventricle=='left':
            x = np.array([xx.mean(axis=0) for xx in cluster])
            s = np.array([len(xx) for xx in cluster])
            indices = [np.argmin(np.linalg.norm(x-self.cluster['heart'][0].mean(axis=0), axis=-1)/s)]
            if max_dist>0:
                indices = self._find_neighbors('white', max_dist, indices)
            return indices
        elif ventricle=='right':
            left = np.array(self.get_index(ventricle='left', max_dist=max_dist, max_search=max_search))
            ratios = np.array([PCA().fit(xx).explained_variance_ratio_[0] for xx in cluster])
            ratios[left[left<max_search]] = 0
            ratios *= np.array([len(xx)for xx in cluster])
            points = self.cluster['heart'][0]
            center = points.mean(axis=0)
            farthest_point = points[np.mean(points-center, axis=-1).argmax()]
            distances = np.array([np.linalg.norm(farthest_point-np.mean(xx, axis=0), axis=-1) for xx in cluster])
            ratios /= 100+distances
            indices = [np.argmax(ratios)]
            if max_dist>0:
                indices = self._find_neighbors('white', max_dist, indices, left, bias=np.array([1.,0.25]))
            return indices

    @property
    def _threshold(self):
        return self.white_color_threshold

    def _sort(self, key, size):
        w = np.where(self.get_area(key, True))
        tree = cKDTree(data=np.stack(w, axis=-1))
        self.apply_median(size=size)
        x = np.stack(np.where(self.get_area(key, True)), axis=-1)
        dist, indices = tree.query(x, p=np.inf)
        indices = indices[dist<size]
        x = x[dist<size]
        dist = dist[dist<size]
        labels, counts = np.unique(self._clustering[key].labels_[indices], return_counts=True)
        self.cluster[key] = []
        if key=='white':
            self.background = []
        for l in labels[np.argsort(counts)[::-1]]:
            xx = x[self._clustering[key].labels_[indices]==l]
            if l==-1:
                continue
            if key=='white' and not np.all(xx<np.array(self._img.shape)[:-1]-1):
                self.background = np.append(self.background, xx).reshape(-1, 2).astype(int)
                continue
            if key=='white' and not np.all(xx>0):
                self.background = np.append(self.background, xx).reshape(-1, 2).astype(int)
                continue
            self.cluster[key].append(xx)
    
    def run_cluster(self, key, eps=3, min_samples=2, size=None, apply_filter=True):
        if key not in ['white', 'heart']:
            raise ValueError('key must be white or heart')
        if size is None and key=='white':
            size = 6
        elif size is None:
            size = 1
        if apply_filter and key=='white':
            self.apply_mininum(size=size)
        elif apply_filter and key=='heart':
            self.apply_maximum(size=size)
        leere = np.stack(np.where(self.get_area(key)), axis=-1)
        self._clustering[key] = DBSCAN(eps=eps, min_samples=min_samples).fit(leere)
        self._sort(key, size=size)

    def get_points(self, key, index=0):
        index = np.array([index]).flatten()
        x = self.cluster[key][index[0]]
        if len(index)>1:
            for ii in index[1:]:
                x = np.concatenate((x, self.cluster[key][ii]))
        return x

    def get_data(self, key, index=0):
        return Area(self.get_points(key, index), resolution=self.resolution)
