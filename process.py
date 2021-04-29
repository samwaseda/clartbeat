import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from area import Area
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull
import skimage.feature
from skimage import filters
from sklearn.cluster import AgglomerativeClustering
from tools import damp, get_slope, MyPCA
from surface import Surface

class ProcessImage:
    def __init__(
        self,
        file_name,
        parameters,
    ):
        self._clustering = {}
        self.cluster = {}
        self._indices = {}
        self._reduction = None
        self._canny_edge_all = None
        self._canny_edge_perimeter = None
        self._elastic_net_perimeter = None
        self._white_color_threshold = None
        self._total_area = None
        self.file_name = file_name
        self._img = None
        self.parameters = parameters

    @property
    def img(self):
        if self._img is None:
            self._img = cleanse_edge(
                img=self.load_image(target_size=self.parameters['target_size']),
                erase_edge=self.parameters['erase_edge']
            )
            self._white_color_threshold = get_white_color_threshold(
                self._img, **self.parameters['white_color']
            )
            self._img = clear_dirt(
                self._img,
                self.white_color_threshold,
                **self.parameters['clear_dirt'],
            )
            self._img = _clean_noise(
                self._img,
                self.white_color_threshold,
                eps=self.parameters['clean_noise']['eps']
            )
        return self._img

    @property
    def white_color_threshold(self):
        if self._white_color_threshold is None:
            _ = self.img
        return self._white_color_threshold 

    def load_image(self, file_name=None, reduction=None, target_size=None):
        if file_name is None and not hasattr(self, 'file_name'):
            raise ValueError('file_name not specified')
        if file_name is None:
            file_name = self.file_name
        img = plt.imread(file_name)
        if target_size is not None:
            reduction = np.rint(np.sqrt(np.prod(img.shape[:2])/target_size)).astype(int)
            reduction = np.max([1, reduction])
            if self._reduction is None:
                self._reduction = reduction
                self.resolution = (self.parameters['resolution']*self._reduction)**2
        if reduction is None:
            reduction = self._reduction
        return img[::reduction,::reduction]

    @property
    def non_white_points(self):
        return np.mean(self.load_image(), axis=-1) < self.white_color_threshold

    @property
    def canny_edge_all(self):
        if self._canny_edge_all is None:
            self._canny_edge_all = get_edge(
            self.get_image(mean=True), self.get_base_color()/255, **self.parameters['canny_edge']
        )
        return np.stack(np.where(self._canny_edge_all), axis=-1)

    def get_total_area(
        self,
        number_of_points=360,
        sigma=0.05,
        height_unit=40,
        eps_areas=5,
        min_fraction=0.05
    ):
        x_range = np.linspace(0, 2*np.pi, number_of_points, endpoint=False)
        dbscan = DBSCAN(eps=eps_areas).fit(self.canny_edge_all)
        labels, counts = np.unique(dbscan.labels_, return_counts=True)
        labels = labels[counts>min_fraction*len(dbscan.labels_)]
        hull = ConvexHull(self.canny_edge_all)
        labels = labels[np.array([
            len(set(np.where(l==dbscan.labels_)[0]).intersection(hull.vertices))>0
            for l in labels
        ])]
        cond = np.any(labels[:,None]==dbscan.labels_, axis=0)
        mean = np.median(self.canny_edge_all, axis=0)
        p = self.canny_edge_all[cond]-mean
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
        xx[xx<0] = 0
        yy[yy<0] = 0
        shape = self.get_image().shape[:-1]
        xx[xx>=shape[0]] = shape[0]-1
        yy[yy>=shape[1]] = shape[1]-1
        return np.stack([xx, yy], axis=-1)

    @property
    def canny_edge_perimeter(self):
        if self._canny_edge_perimeter is None:
            self._canny_edge_perimeter = self.get_total_area(**self.parameters['total_area'])
            self._elastic_net_perimeter = Surface(self._canny_edge_perimeter.copy())
        return self._canny_edge_perimeter

    @property
    def total_area(self):
        if self._total_area is None:
            self._total_area = self.determine_total_area()
        return self._total_area

    def determine_total_area(self):
        canvas = np.ones_like(self.get_image(mean=True))
        mean = np.mean(self.total_perimeter.x, axis=0)
        x = canvas*np.arange(canvas.shape[0])[:,None]
        y = canvas*np.arange(canvas.shape[1])
        x -= mean[0]
        y -= mean[1]
        canvas_r = np.sqrt(x**2+y**2)
        canvas_angle = np.arctan2(y, x)
        x = self.total_perimeter.x-mean
        r = np.linalg.norm(x, axis=-1)
        angle = np.arctan2(x[:,1], x[:,0])
        argmin = np.argmin(np.absolute(canvas_angle[:,:,None]-angle[None,None,:]), axis=-1)
        return canvas_r<r[argmin]

    def stich_high_angles(
        self,
        sigma=5,
        max_angle=0.045,
    ):
        if max_angle > 0.5*np.pi or max_angle < 0:
            return
        v = self.canny_edge_perimeter.copy()
        v -= np.roll(v, -1, axis=0)
        v_norm = np.linalg.norm(v, axis=-1)
        sin = np.arcsin(np.cross(v, np.roll(v, 1, axis=0))/(v_norm*np.roll(v_norm, 1)))
        total_number = len(self.canny_edge_perimeter)
        high_angles = ndimage.gaussian_filter1d(sin, sigma=sigma)>max_angle
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
        self._elastic_net_perimeter.x = np.roll(self._elastic_net_perimeter.x, -indices[0], axis=0)
        indices = (np.diff(indices)+total_number)[0]%total_number
        i_range = np.arange(indices)/indices
        dr = i_range[:,None]*(self._elastic_net_perimeter.x[indices]-self._elastic_net_perimeter.x[0])
        self._elastic_net_perimeter.x[:indices] = dr+self._elastic_net_perimeter.x[0]
        center = np.mean(self._elastic_net_perimeter.x, axis=0)
        r_a = np.linalg.norm(self._elastic_net_perimeter.x[0]-center)
        r_b = np.linalg.norm(self._elastic_net_perimeter.x[indices]-center)
        inner_prod = np.dot(
            self._elastic_net_perimeter.x[0]-center, self._elastic_net_perimeter.x[indices]-center
        )
        magnifier = i_range*r_a+(1-i_range)*r_b
        magnifier /= np.sqrt(
            i_range**2*r_a**2+(1-i_range)**2*r_b**2+2*i_range*(1-i_range)*inner_prod
        )
        self._elastic_net_perimeter.x[:indices] = magnifier[:,None]*(
            self._elastic_net_perimeter.x[:indices]-center
        )
        self._elastic_net_perimeter.x[:indices] += center

    @property
    def total_perimeter(self):
        if self._elastic_net_perimeter is None:
            self.stich_high_angles(**self.parameters['stich_high_angles'])
            self.run_elastic_net(**self.parameters['elastic_net'])
        return self._elastic_net_perimeter

    def run_elastic_net(
        self,
        sigma_sobel=15,
        sigma_gauss=5,
        line_tension=0.2,
        dt=0.1,
        max_iter=1000,
        max_gradient=0.1,
        repel_strength=0.01,
    ):
        if max_iter < 1:
            return
        sobel = filters.sobel(
            ndimage.gaussian_filter(self.get_image(mean=True), sigma=sigma_sobel)
        )
        gauss = repel_strength*ndimage.gaussian_filter(
            self.get_image(mean=True), sigma=sigma_gauss
        )
        self._elastic_net_perimeter.set_energy_field(sobel)
        self._elastic_net_perimeter.set_energy_field(gauss)
        for i in range(1000):
            f_spring = line_tension*self._elastic_net_perimeter.dhook
            f_total = self._elastic_net_perimeter.force_field+f_spring
            self._elastic_net_perimeter.x -= f_total*dt
            if np.linalg.norm(f_total, axis=-1).max()<max_gradient:
                break

    def get_image(self, mean=False):
        if mean:
            return np.mean(self.img, axis=-1)
        return self.img.copy()

    def get_base_color(self, mean=True):
        mean_color = np.mean(self.img, axis=-1)
        if mean:
            return np.mean(self.img[mean_color<self.white_color_threshold])
        return np.mean(self.img[mean_color<self.white_color_threshold], axis=0)

    @property
    def norm(self):
        return self._norm*self.total_area

    @norm.setter
    def norm(self, n):
        self._norm = n

    def apply_minimum(self, size=6):
        norm = np.mean(self.img, axis=-1)
        if size > 0:
            norm = ndimage.minimum_filter(norm, size=size)
        self.norm = norm
        
    def apply_maximum(self, size=1):
        norm = np.mean(self.img, axis=-1)
        if size > 0:
            norm = ndimage.maximum_filter(norm, size=size)
        self.norm = norm

    def apply_median(self, size=1):
        norm = np.mean(self.img, axis=-1)
        if size > 0:
            norm = ndimage.median_filter(norm, size=size)
        self.norm = norm

    def get_area(self, key, smoothened=True):
        if key=='white':
            if smoothened:
                return self.norm > self._threshold
            else:
                return np.mean(self.img, axis=-1) > self._threshold
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

    def _find_neighbors(
        self,
        key,
        max_dist,
        indices,
        indices_to_avoid=None,
        bias=None,
        max_angle=45/180*np.pi,
        recursion=0
    ):
        x = np.concatenate([self.cluster[key][ind] for ind in indices])
        if max_angle is not None and self._get_max_angle(x) > max_angle:
            return indices
        if bias is not None: 
            x = self._get_biased_coordinates(x, bias)
        tree = cKDTree(x)
        for ii,xx in enumerate(self.cluster[key]):
            if ii in indices:
                continue
            if bias is not None:
                xx = self._get_biased_coordinates(xx, bias)
            if tree.query(xx)[0].min()>max_dist:
                continue
            if indices_to_avoid is not None and ii in indices_to_avoid:
                continue
            indices.append(ii) 
            if recursion > len(indices):
                return self._find_neighbors(
                    key=key,
                    max_dist=max_dist,
                    indices=indices,
                    indices_to_avoid=indices_to_avoid,
                    bias=bias,
                    max_angle=max_angle,
                    recursion=recursion
                )
        return indices

    @property
    def total_mean_radius(self):
        return np.sqrt(np.sum(self.total_area)/np.pi)

    def _satisfies_criteria(self, size, dist, dist_interval=None, fraction_interval=None):
        if dist_interval is None or fraction_interval is None:
            return True
        fraction_criterion = get_slope(size/np.sum(self.total_area), fraction_interval)
        dist_criterion = get_slope(dist/self.total_mean_radius, dist_interval)
        return np.any(fraction_criterion*dist_criterion > 0.5)

    def get_index(
        self,
        ventricle='left',
        max_search=5,
        max_dist=5,
        bias=np.ones(2),
        dist_interval=None,
        fraction_interval=None,
        indices_to_avoid=None,
        recursion=0,
    ):
        if ventricle=='right':
            if 'left' in self._indices.keys() and self._indices['left'] is None:
                return None
        cluster = self.cluster['white'][:max_search]
        heart_center = self.cluster['heart'][0].mean(axis=0)
        distances = np.array([
            np.linalg.norm(heart_center-np.mean(xx, axis=0), axis=-1) for xx in cluster
        ])
        size = np.array([len(xx) for xx in cluster])
        if not self._satisfies_criteria(size, distances, dist_interval, fraction_interval):
            self._indices[ventricle] = None
            return None
        if ventricle=='left':
            x = np.array([xx.mean(axis=0) for xx in cluster])
            indices = [
                np.argmin(np.linalg.norm(x-self.cluster['heart'][0].mean(axis=0), axis=-1)/size)
            ]
            if max_dist>0:
                indices = self._find_neighbors('white', max_dist, indices, max_angle=None)
            self._indices['left'] = indices
        elif ventricle=='right':
            if indices_to_avoid is None and 'left' in self._indices.keys():
                indices_to_avoid = np.array(self._indices['left'])
            ratios = np.array([PCA().fit(xx).explained_variance_ratio_[0] for xx in cluster])
            ratios[indices_to_avoid[indices_to_avoid<max_search]] = 0
            ratios *= size
            ratios *= np.log(distances)
            indices = [np.argmax(ratios)]
            if max_dist>0:
                indices = self._find_neighbors(
                    'white',
                    max_dist,
                    indices,
                    indices_to_avoid,
                    bias=np.array(bias),
                    recursion=recursion
                )
        return np.unique(indices)

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
            if key=='white' and not np.all(xx<np.array(self.img.shape)[:-1]-1):
                self.background = np.append(self.background, xx).reshape(-1, 2).astype(int)
                continue
            if key=='white' and not np.all(xx>0):
                self.background = np.append(self.background, xx).reshape(-1, 2).astype(int)
                continue
            self.cluster[key].append(xx)

    def run_cluster(self, key, eps=3, size=None, apply_filter=True):
        if key not in ['white', 'heart']:
            raise ValueError('key must be white or heart')
        if size is None and key=='white':
            size = 6
        elif size is None:
            size = 1
        if apply_filter and key=='white':
            self.apply_minimum(size=size)
        elif apply_filter and key=='heart':
            self.apply_maximum(size=size)
        leere = np.stack(np.where(self.get_area(key)), axis=-1)
        self._clustering[key] = DBSCAN(eps=eps).fit(leere) # min_samples = 2 ?
        self._sort(key, size=size)

    def get_points(self, key, index=0):
        if index is None:
            return np.array([])
        index = np.array([index]).flatten()
        x = self.cluster[key][index[0]]
        if len(index)>1:
            for ii in index[1:]:
                x = np.concatenate((x, self.cluster[key][ii]))
        return x

    def get_data(self, key, index=0):
        if key=='heart':
            return Area(self.get_points(key, index), perimeter=self.total_perimeter)
        return Area(self.get_points(key, index))

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

def get_white_color_threshold(img, bins=100, min_value=175, value=None, **args):
    if value is not None:
        return value
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
    if erase_edge==0:
        return img_new
    img_new[:erase_edge,:,:] = np.array(3*[255])
    img_new[:,:erase_edge,:] = np.array(3*[255])
    img_new[-erase_edge:,:,:] = np.array(3*[255])
    img_new[:,-erase_edge:,:] = np.array(3*[255])
    return img_new

def clear_dirt(img, white_threshold, filter_size=10, brightness_range=10, radius_threshold=0.1):
    img_mean = np.mean(img, axis=-1)
    pca = MyPCA().fit(np.stack(np.where(img_mean<white_threshold), axis=-1))
    x = np.arange(img_mean.shape[0])
    y = np.arange(img_mean.shape[1])
    f = np.stack(np.meshgrid(x, y), axis=-1)
    distance_cond = get_slope(
        pca.get_scaled_distance(f), np.array([1, 1+radius_threshold])
    ).T
    filtered = ndimage.median_filter(img_mean, size=filter_size)-white_threshold
    color_cond = get_slope(filtered, np.array([-1, 1])*brightness_range)
    img[distance_cond*color_cond>0.5] = np.array(3*[255])
    return img

def _clean_noise(img, threshold, eps=5, min_fraction=0.03):
    x = np.stack(np.where(np.mean(img, axis=-1)<threshold), axis=-1)
    cluster = DBSCAN(eps=eps).fit(x)
    labels, counts = np.unique(cluster.labels_, return_counts=True)
    counts = counts[labels!=-1]
    labels = labels[labels!=-1]
    labels = labels[counts/counts.sum()>min_fraction]
    y = x[np.all(cluster.labels_[:,None]!=labels[None,:], axis=-1)]
    img[y[:,0], y[:,1]] = np.array(3*[255])
    return img

def get_edge(img, base, sigma=18.684, low=6.1578, high=7.6701):
    return skimage.feature.canny(
        image=img,
        sigma=sigma,
        low_threshold=low*base,
        high_threshold=high*base,
    )

def get_local_linear_fit(y_i, x_i, w):
    w = w/w.sum(axis=1)[:,None]
    wx = np.sum(w*x_i, axis=-1)
    wy = np.sum(w*y_i, axis=-1)
    wxx = np.sum(w*x_i**2, axis=-1)
    wxy = np.sum(w*x_i*y_i, axis=-1)
    w = np.sum(w, axis=-1)
    return (w*wxy-wx*wy)/(wxx*w-wx**2), (-wx*wxy+wxx*wy)/(wxx*w-wx**2)

