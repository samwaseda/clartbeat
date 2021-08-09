import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from clartbeat.area import Area
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull
from skimage import feature
from skimage import filters
from sklearn.cluster import AgglomerativeClustering
from clartbeat.tools import *
from clartbeat.surface import Surface

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
        self._base_color = None
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
        return get_reduced_mean(img, reduction)

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
        large_enough = large_chunk(labels, min_fraction=min_fraction)
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
        x_i = np.arctan2(*p.T[::-1])
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
        max_angle_diff=100,
    ):
        total_number = len(self.canny_edge_perimeter.x)
        high_angles = -self.canny_edge_perimeter.get_curvature(sigma=sigma)>max_angle
        high_angles = np.arange(len(high_angles))[high_angles]
        if len(high_angles)<2:
            return
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1.1, linkage='single'
        ).fit_predict(high_angles.reshape(-1, 1))
        indices = np.sort([
            np.rint(high_angles[labels==l].mean()).astype(int)
            for l in np.unique(labels)
        ])
        if len(indices)!=2:
            return
        d = np.diff(indices)[0]
        if np.absolute(d-np.rint(d/total_number)*total_number) > max_angle_diff:
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

    @property
    def non_white_area(self):
        return self.get_image(mean=True) < self.white_color_threshold

    @staticmethod
    def _find_maximum(indices, sigma=8, n_items=256, min_fraction=0.5):
        count = np.zeros(n_items)
        np.add.at(count, indices, 1)
        count = ndimage.gaussian_filter(count, sigma)
        cond = np.where((count[1:-1]>count[:-2])*(count[1:-1]>count[2:]))[0]
        if np.sum(cond)==0:
            return count.argmax()
        cond = cond[count[cond]/count[cond].max()>min_fraction]
        return cond[0]

    def get_base_color(self, mean=True, sigma=6, min_fraction=0.5):
        if self._base_color is None:
            all_colors = self.get_image()[self.non_white_area]
            unique_colors, counts = np.unique(all_colors, return_counts=True, axis=0)
            field = np.zeros((256, 256, 256))
            field[tuple(unique_colors.T)] = counts
            field = ndimage.gaussian_filter(field, sigma=sigma)
            cond = (field==ndimage.maximum_filter(field, size=sigma))*(field!=0)
            colors = np.stack(np.where(cond)).T
            colors = colors[field[cond]>min_fraction*field[cond].max()]
            self._base_color = colors[np.std(colors, axis=-1).argmax()]
        if mean:
            return np.mean(self._base_color)
        return self._base_color

    @property
    def relative_distance_from_base_color(self):
        img = self.get_image()-self.get_base_color(mean=False)
        img = np.linalg.norm(img, axis=-1)
        return img/img.max()

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

    def _get_relative_coordinates(self, x, theta_0=0):
        xx = self.ref_job.heart.pca.get_relative_points(x)
        theta = np.arctan2(*xx.T)
        theta -= theta_0
        theta -= np.rint(theta*0.5/np.pi)*2*np.pi
        r = np.linalg.norm(xx, axis=-1)
        return np.stack((r, theta), axis=-1)

    def _polar_to_cartesian(self, rt):
        return self.ref_job.heart.pca.get_absolute_points(
            np.stack(rt[:,0]*np.array([np.cos(rt[:,1]), np.sin(rt[:,1])]), axis=-1)
        )

    def _find_neighbors(self, key, bias=None, max_dist=20, min_counts=1):
        x_current = self.white_area.get_all_positions(key)
        if bias is not None:
            theta_0 = np.mean(self._get_relative_coordinates(x_current)[:,1])
            rt_l = self._get_relative_coordinates(x_current, theta_0)
            x_current = self._polar_to_cartesian(rt_l*bias)
        tree = cKDTree(x_current)
        for ii,x in zip(
            self.white_area.get_indices('unknown', unique=True), self.white_area.get_positions()
        ):
            if bias is not None:
                x = self._polar_to_cartesian(self._get_relative_coordinates(x, theta_0)*bias)
            counts = tree.count_neighbors(cKDTree(x), r=max_dist)/max_dist**3
            if counts > min_counts:
                self.white_area.tags[ii] = key
                self._find_neighbors(key, bias=bias, max_dist=max_dist, min_counts=min_counts)

    @property
    def total_mean_radius(self):
        return np.sqrt(np.sum(self.total_area)/np.pi)

    def _left_lumen_exists(self, size, dist, dist_interval=None, fraction_interval=None):
        if dist_interval is None or fraction_interval is None:
            return True
        fraction_criterion = get_slope(size/np.sum(self.total_area), fraction_interval)
        dist_criterion = get_slope(dist/self.total_mean_radius, dist_interval)
        return np.any(fraction_criterion*dist_criterion > 0.5)

    def _remove_excess(
        self,
        points,
        eps=1.5,
        size=0.05,
        min_samples=5,
        min_fraction=0.2,
    ):
        if size*eps==0:
            return points
        size = np.rint(np.sqrt(len(points))*size).astype(int)
        area = self.get_canvas(points, fill_value=0)
        x = np.stack(np.where(ndimage.minimum_filter(area, size=size)>0), axis=-1)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
        tree = cKDTree(data=x)
        dist, indices = tree.query(points, p=np.inf, distance_upper_bound=size)
        x, indices = abridge(dist<size, points, indices)
        labels = labels[indices]
        return x[large_chunk(labels, min_fraction=min_fraction)]

    def get_left_lumen(
        self,
        max_dist=20,
        dist_interval=None,
        fraction_interval=[0.001, 0.006],
        recursion=0,
        min_counts=1,
        eps_excess=1.5,
        size_excess=0.05,
        min_samples=5,
        min_fraction=0.2
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
        indices = np.unique(indices)
        self.white_area[indices] = 'left'
        if max_dist > 0:
            self._find_neighbors('left', max_dist=max_dist, min_counts=min_counts)
        x = self.white_area.get_all_positions('left')
        return self._remove_excess(
            x,
            eps=eps_excess,
            size=size_excess,
            min_samples=min_samples,
            min_fraction=min_fraction
        )

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
        sigmas=[20, 35],
        sigma_interval=[0.08, 0.15],
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
        max_dist=20,
        bias=[1.0, 0.2],
        min_counts=1,
        dist_interval=None,
        recursion=0,
        r_perimeter=3,
        r_left=5,
        contact_interval=[0.3, 0],
        curvature_sigmas=[20, 30],
        curvature_sigma_interval=[0.08, 0.12],
        curvature_interval=[0.002, -0.002],
        min_weight=0.0017,
        eps_excess=1.5,
        size_excess=0.05,
        min_samples=5,
        min_fraction=0.2
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
        self.white_area.tags[indices] = 'right'
        if max_dist > 0:
            self._find_neighbors(
                'right',
                bias=bias,
                max_dist=max_dist,
                min_counts=min_counts,
            )
        x = self.white_area.get_all_positions('right')
        return self._remove_excess(
            x,
            eps=eps_excess,
            size=size_excess,
            min_samples=min_samples,
            min_fraction=min_fraction
        )

    def _get_radial_mean_value(self, center=None, tag='unknown'):
        if center is None:
            center = self.heart_area.mean(axis=0)
        x_mean_lst = []
        for x in self.white_area.get_positions(tag=tag):
            xx = x-center
            r_mean = np.linalg.norm(xx, axis=-1).mean()
            x_mean_lst.append(xx.mean(axis=0)/np.linalg.norm(xx.mean(axis=0))*r_mean+center)
        return np.array(x_mean_lst)

    def _get_white_area(self, eps=1, min_samples=5, size=6, max_regroup_fraction=0.1):
        x_min = self.apply_filter(ndimage.minimum_filter, size=size)
        tree = cKDTree(data=x_min)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_min)
        x = self.apply_filter(ndimage.median_filter, size=size)
        dist = tree.query(x, p=np.inf, distance_upper_bound=size)[0]
        x_core = x[dist<size]
        if len(np.unique(labels[large_chunk(labels, max_regroup_fraction)]))==1:
            tree = cKDTree(data=x)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
        labels = labels[tree.query(x_core)[1]]
        return WhiteArea(*abridge(labels!=-1, x_core, labels))

    def apply_filter(self, filter_to_apply, size):
        area = filter_to_apply(self.get_image(mean=True), size=size)
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

def get_reduced_mean(img, reduction):
    size = np.array(img.shape[:2])
    new_size = reduction*np.floor(size/reduction).astype(int)
    img = img[:new_size[0], :new_size[1]]
    img = img.reshape(int(new_size[0]/reduction), reduction, int(new_size[1]/reduction), reduction, 3)
    img = np.median(img, axis=(1,3))
    return np.rint(img).astype(int)

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

def get_white_color_threshold(img, bins=1000, sigma=3):
    x_range = np.linspace(0, 255, bins)
    values, counts = np.unique(np.rint(img.mean(axis=-1).flatten()), return_counts=True)
    gaussian = np.sum(
        np.log10(counts)/np.sqrt(sigma)*np.exp(-(values-x_range[:,None])**2/(2*sigma**2)),
        axis=1
    )
    max_lim = np.where(get_extrema(gaussian, True))[0][-1]
    return x_range[np.where(get_extrema(gaussian, False)[:max_lim])[0][-1]]

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
    f = np.stack(np.meshgrid(*(np.arange(s) for s in img_mean.shape)), axis=-1)
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
    counts, labels = abridge(labels!=-1, counts, labels)
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

