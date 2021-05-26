import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from area import Area
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull
from skimage import feature
from skimage import filters
from sklearn.cluster import AgglomerativeClustering
from tools import damp, get_slope, MyPCA, find_common_labels, get_softplus
from surface import Surface

class ProcessImage:
    def __init__(
        self,
        ref_job,
        file_name,
        parameters,
    ):
        self.ref_job = ref_job
        self._contact_peri = None
        self._reduction = None
        self._canny_edge_all = None
        self._canny_edge_perimeter = None
        self._elastic_net_perimeter = None
        self._white_color_threshold = None
        self._total_area = None
        self._white_area = None
        self._stiched = False
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

    def _get_main_edges(self, eps_areas=5, min_fraction=0.2):
        labels = DBSCAN(eps=eps_areas).fit(self.canny_edge_all).labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        large_enough = find_common_labels(
            unique_labels[counts/counts[unique_labels!=-1].max() > min_fraction], labels
        )
        hull = ConvexHull(self.canny_edge_all[large_enough])
        return self.canny_edge_all[
            find_common_labels(labels[large_enough][hull.vertices], labels)
        ]

    def get_total_area(
        self,
        number_of_points=360,
        sigma=0.05,
        height_unit=40,
        eps_areas=5,
        min_fraction=0.04
    ):
        p = self._get_main_edges(eps_areas=eps_areas, min_fraction=min_fraction).astype(float)
        mean = np.mean(p, axis=0)
        p -= mean
        x_i = np.arctan2(p[:,1], p[:,0])
        y_i = np.linalg.norm(p, axis=-1)
        x_i = np.concatenate((x_i-2*np.pi, x_i, x_i+2*np.pi))
        y_i = np.concatenate((y_i, y_i, y_i))
        x_range = np.linspace(0, 2*np.pi, number_of_points, endpoint=False)
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
            self._canny_edge_perimeter = Surface(
                self.get_total_area(**self.parameters['total_area'])
            )
            self._elastic_net_perimeter = self._canny_edge_perimeter.copy()
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
        max_angle=16.2,
    ):
        total_number = len(self.canny_edge_perimeter.x)
        high_angles = -self.canny_edge_perimeter.get_curvature(sigma=sigma)>max_angle
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
        self._elastic_net_perimeter.x = np.roll(
            self._elastic_net_perimeter.x, -indices[0], axis=0
        )
        indices = (np.diff(indices)+total_number)[0]%total_number
        i_range = np.arange(indices)/indices
        dr = i_range[:,None]*(
            self._elastic_net_perimeter.x[indices]-self._elastic_net_perimeter.x[0]
        )
        self._elastic_net_perimeter.x[:indices] = dr+self._elastic_net_perimeter.x[0]
        center = np.mean(self._elastic_net_perimeter.x, axis=0)
        r_a = np.linalg.norm(self._elastic_net_perimeter.x[0]-center)
        r_b = np.linalg.norm(self._elastic_net_perimeter.x[indices]-center)
        inner_prod = np.dot(
            self._elastic_net_perimeter.x[0]-center,
            self._elastic_net_perimeter.x[indices]-center
        )
        magnifier = i_range*r_a+(1-i_range)*r_b
        magnifier /= np.sqrt(
            i_range**2*r_a**2+(1-i_range)**2*r_b**2+2*i_range*(1-i_range)*inner_prod
        )
        self._elastic_net_perimeter.x[:indices] = magnifier[:,None]*(
            self._elastic_net_perimeter.x[:indices]-center
        )
        self._elastic_net_perimeter.x[:indices] += center
        self._stiched = True

    @property
    def total_perimeter(self):
        if self._elastic_net_perimeter is None:
            self.stich_high_angles(**self.parameters['stich_high_angles'])
            self.run_elastic_net(**self.parameters['elastic_net'])
        return self._elastic_net_perimeter

    def unstich(self):
        if not self._stiched:
            return
        self.ref_job.initialize()
        self._elastic_net_perimeter = self.canny_edge_perimeter.copy()
        self.run_elastic_net(**self.parameters['elastic_net'])

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
        max_dist,
        indices,
        indices_to_avoid=None,
        bias=None,
        max_angle=45/180*np.pi,
        recursion=0
    ):
        x = np.concatenate([self.white_area[ind] for ind in indices])
        #if max_angle is not None and self._get_max_angle(x) > max_angle:
        #    return indices
        if bias is not None: 
            x = self._get_biased_coordinates(x, bias)
        tree = cKDTree(x)
        for ii,xx in enumerate(self.white_area):
            if ii in indices:
                continue
            if bias is not None:
                xx = self._get_biased_coordinates(xx, bias)
            # if tree.query(xx)[0].min()>max_dist:
            if indices_to_avoid is not None and ii in indices_to_avoid:
                continue
            min_size = max_dist*(2*np.sqrt(np.pi*len(xx))-np.pi*max_dist)
            if tree.count_neighbors(cKDTree(xx), max_dist) < min_size:
                continue
            indices.append(ii) 
            if recursion > len(indices):
                return self._find_neighbors(
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

    def _left_lumen_exists(self, size, dist, dist_interval=None, fraction_interval=None):
        if dist_interval is None or fraction_interval is None:
            return True
        fraction_criterion = get_slope(size/np.sum(self.total_area), fraction_interval)
        dist_criterion = get_slope(dist/self.total_mean_radius, dist_interval)
        return np.any(fraction_criterion*dist_criterion > 0.5)

    def get_left_lumen(
        self,
        max_dist=5,
        dist_interval=None,
        fraction_interval=[0.001, 0.006],
        recursion=0,
    ):
        if 'left' in self.white_area.tags:
            return self.white_area.get_all_positions('left')
        heart_center = self.heart_area.mean(axis=0)
        distances = np.array([
            np.linalg.norm(heart_center-np.mean(xx, axis=0), axis=-1)
            for xx in self.white_area.get_positions(tag='unknown')
        ])
        size = self.white_area.get_counts(tag='unknown')
        if not self._left_lumen_exists(size, distances, dist_interval, fraction_interval):
            return None
        x = self._get_radial_mean_value()
        indices = np.argmin(np.linalg.norm(x-heart_center, axis=-1)**2/size)
        # if max_dist > 0:
        #     indices = self._find_neighbors(max_dist, indices, max_angle=None)
        indices = np.unique(indices)
        self.white_area[indices] = 'left'
        return self.white_area.get_all_positions('left')

    def _get_rl_contact_counts(self, tree, r_max, contact_interval, tag='unknown'):
        if tree is None:
            return 0
        indices, values = self._get_contact_counts(tree=tree, r_max=r_max, tag=tag)
        return self.white_area.fill(
            get_slope(values, contact_interval), indices, filler=1.0, tag=tag
        )

    def _get_rl_perimeter(self, r_max=3, contact_interval=[0.3, 0], tag='unknown'):
        return self._get_rl_contact_counts(
            self.ref_job.heart.perimeter.tree,
            r_max=r_max,
            contact_interval=contact_interval,
            tag=tag
        )

    def _get_rl_left(self, r_max=5, contact_interval=[0.3, 0], tag='unknown'):
        return self._get_rl_contact_counts(
            self.ref_job.left.tree, r_max=r_max, contact_interval=contact_interval, tag=tag
        )

    def _get_rl_size(self, tag='unknown'):
        return self.white_area.fill(
            self.white_area.get_counts(tag=tag)/len(self.heart_area), tag=tag
        )

    def _get_rl_distance(self, tag='unknown'):
        distance = np.log(
            self.ref_job.left.pca.get_scaled_distance(self._get_radial_mean_value(tag=tag))
        )
        distance += np.log(self.ref_job.left.get_length().mean())
        distance -= np.log(self.ref_job.heart.get_length().mean())
        return self.white_area.fill(get_softplus(distance), tag=tag)

    def _get_rl_curvature(
        self,
        sigmas=[20, 30],
        sigma_interval=[0.08, 0.12],
        curvature_interval=[0.002, -0.002],
        tag='unknown'
    ):
        sigma = sigmas[0]+get_slope(
            np.sqrt(self.white_area.get_counts(tag=tag).max()/len(self.heart_area)),
            sigma_interval
        )*np.diff(sigmas)[0]
        return self.white_area.fill(get_slope([
            self.ref_job.heart.perimeter.get_crossing_curvature(
                self.ref_job.left.get_center(),
                np.mean(x, axis=0),
                sigma=sigma,
                laplacian=True
            )
            for x in self.white_area.get_positions(tag=tag)
        ], curvature_interval), tag=tag)

    def get_rl_weights(
        self,
        r_perimeter=3,
        r_left=5,
        contact_interval=[0.3, 0],
        curvature_sigmas=[20, 30],
        curvature_sigma_interval=[0.08, 0.12],
        curvature_interval=[0.002, -0.002],
        tag='unknown',
    ):
        w = self._get_rl_perimeter(r_max=r_perimeter, contact_interval=contact_interval, tag=tag)
        w *= self._get_rl_left(r_max=r_left, contact_interval=contact_interval, tag=tag)
        w *= self._get_rl_size(tag=tag)
        w *= self._get_rl_distance(tag=tag)
        w *= self._get_rl_curvature(
            sigmas=curvature_sigmas,
            sigma_interval=curvature_sigma_interval,
            curvature_interval=curvature_interval,
            tag=tag
        )
        return w

    def get_right_lumen(
        self,
        max_dist=5,
        bias=[1.5, 0.2],
        dist_interval=None,
        recursion=0,
        r_perimeter=3,
        r_left=5,
        contact_interval=[0.3, 0],
        curvature_sigmas=[20, 30],
        curvature_sigma_interval=[0.08, 0.12],
        curvature_interval=[0.002, -0.002],
        min_weight=0.002
    ):
        if 'right' in self.white_area.tags:
            return self.white_area.get_all_positions('right')
        if not self.ref_job.left.exists():
            return None
        weights = self.get_rl_weights(
            r_perimeter=r_perimeter,
            r_left=r_left,
            contact_interval=contact_interval,
            curvature_sigmas=curvature_sigmas,
            curvature_sigma_interval=curvature_sigma_interval,
            curvature_interval=curvature_interval
        )
        if weights.max() < min_weight:
            return None
        indices = np.argmax(weights)
        # if max_dist > 0:
        #     indices = self._find_neighbors(
        #         max_dist,
        #         indices,
        #         bias=np.array(bias),
        #         recursion=recursion
        #     )
        self.white_area.tags[indices] = 'right'
        return self.white_area.get_all_positions('right')

    def _get_radial_mean_value(self, center=None, tag='unknown'):
        if center is None:
            center = self.heart_area.mean(axis=0)
        x_mean_lst = []
        for x in self.white_area.get_positions(tag=tag):
            xx = x-center
            r_mean = np.linalg.norm(xx, axis=-1).mean()
            x_mean_lst.append(xx.mean(axis=0)/np.linalg.norm(xx.mean(axis=0))*r_mean+center)
        return np.array(x_mean_lst)

    def _get_white_area(self, eps=1, min_samples=5, size=5, **kwargs):
        x = self.apply_filter(ndimage.minimum_filter, size=size)
        tree = cKDTree(data=x)
        x = self.apply_filter(ndimage.median_filter, size=size)
        dist = tree.query(x, p=np.inf, distance_upper_bound=size)[0]
        x_core = x[dist<size]
        tree = cKDTree(data=x)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
        labels = labels[tree.query(x_core)[1]]
        cond = labels!=-1
        return WhiteArea(x_core[cond], labels[cond])

    def apply_filter(self, filter_to_apply, size, area=None):
        if area is None:
            area = self.get_image(mean=True)
        area = filter_to_apply(area, size=size)
        return np.stack(np.where(area*self.total_area > self.white_color_threshold), axis=-1)

    @property
    def white_area(self):
        if self._white_area is None:
            self._white_area = self._get_white_area(**self.parameters['white'])
            if len(self._white_area)==0:
                raise AssertionError('No white area detected')
            if 'perimeter_contact_interval' in self.parameters['white'].keys():
                self.parameters['white']['max_ratio'] = np.max(
                    self.parameters['white']['perimeter_contact_interval']
                )
            self._remove_perimeter_white_area(**self.parameters['white'])
        return self._white_area

    def _get_contact_counts(self, tree, r_max=3, tag='unknown'):
        dist, _ = tree.query(
            self.white_area.get_all_positions(tag=tag),
            distance_upper_bound=r_max
        )
        indices, counts = np.unique(
            self.white_area.get_all_indices(tag=tag)[dist<np.inf],
            return_counts=True
        )
        return indices, counts/r_max**2/np.sqrt(
            self.white_area.counts[indices]
        )

    def _remove_perimeter_white_area(
        self, r_perimeter=3, max_ratio=0.3, **kwargs
    ):
        indices, values = self._get_contact_counts(
            tree=self.ref_job.heart.perimeter.tree, r_max=r_perimeter
        )
        cond = values > max_ratio
        self.white_area.tags[indices[cond]] = 'excess'

    def get_canvas(self, x, values=1, fill_value=np.nan):
        img = np.full(self.img.shape[:-1], fill_value=fill_value)
        if isinstance(x, Area):
            x = x.points
        img[x[:,0], x[:,1]] = values
        return img

    @property
    def heart_area(self):
        return np.stack(np.where(self.total_area), axis=-1)

    def get_data(self, key):
        if key=='heart':
            return Area(self.heart_area, perimeter=self.total_perimeter)
        elif key=='left':
            return Area(self.get_left_lumen(**self.parameters['left']))
        elif key=='right':
            return Area(self.get_right_lumen(**self.parameters['right']))
        else:
            raise KeyError(key + ' not recognized')

class WhiteArea:
    def __init__(self, positions, labels):
        self.x = positions
        _, self.all_indices = np.unique(labels, return_inverse=True)
        unique_labels, counts = np.unique(self.all_indices, return_counts=True)
        unique_labels = unique_labels[counts.argsort()[::-1]]
        self.all_indices = np.argsort(unique_labels)[self.all_indices]
        self.counts = np.sort(counts)[::-1]
        self.tags = np.array(len(unique_labels)*['unknown'])

    def __len__(self):
        return len(self.counts)

    def get_counts(self, tag='unknown'):
        return self.counts[self.get_indices(tag=tag, unique=True, boolean=True)]

    def get_positions(self, tag='unknown'):
        indices = self.get_indices(tag=tag, unique=True, boolean=False)
        for i in indices:
            yield self.x[self.all_indices==i]

    def get_all_positions(self, tag):
        return self.x[self.get_indices(tag=tag, unique=True, boolean=True)[self.all_indices]]

    def get_all_indices(self, tag):
        return self.all_indices[
            self.get_indices(tag=tag, unique=True, boolean=True)[self.all_indices]
        ]

    def fill(self, values, indices=None, filler=0.0, tag='unknown'):
        if indices is None:
            indices = self.get_indices(tag=tag, unique=True, boolean=True)
        arr = np.array(len(self)*[filler])
        arr[indices] = values
        return arr

    def get_indices(self, tag='unknown', unique=False, boolean=False):
        if unique:
            tag_lst = self.tags
        else:
            tag_lst = self.tags[self.all_indices]
        if isinstance(tag, str):
            if tag == 'all':
                v = np.array(len(tag_lst)*[True])
            else:
                v = tag_lst == tag
        else:
            v = np.any(tag_lst[:,None]==np.asarray(tag)[None,:], axis=1)
        if boolean:
            return v
        else:
            return np.where(v)[0]

    def __setitem__(self, index, tag):
        self.tags[np.where(self.tags=='unknown')[0][index]] = tag

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
    return feature.canny(
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

