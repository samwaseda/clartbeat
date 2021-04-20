import numpy as np
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from area import Area
from skimage.filters import frangi
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from tools import get_slope

class Tissue:
    def __init__(
        self,
        img,
        total_area,
        white_areas=None,
        n_components=9,
        scar_coeff=np.array([
            2.69008370e-02,
            -1.16449179e+02,
            -5.88774824e+01,
            6.50569742e+01, 
            6.02359123e+01, 
            7.63830882e+01,
            5.55613592
        ]),
        screening_coeff=np.array([1.85159101e-02, 9.44901345e+01]),
        final_coeff=np.array([1.60073308e-02, 8.66532441e+01]),
        wrinkle_coeff=np.array([6.28414588e-02, 1.08703990e+02, 85]),
        min_dist_num_ratio=100,
        sigmas=[4],
        min_ave=50,
        color_enhancement=10,
        filter_size=6,
        variance=10,
        fill_white=True,
    ):
        self._data = {}
        self._names = np.array(['muscle', 'scar', 'fibrous_tissue', 'wrinkle', 'residue'])
        self.fibrous_tissue_cluster = None
        self.img = img.copy()
        self.min_ave = min_ave
        self.positions = self._remove_white(total_area, sigmas, white_areas)
        if fill_white:
            self.img = self._fill_white()
        self.img = self.get_median_filter(size=filter_size)
        self.run_cluster(n_components=n_components, color_enhancement=color_enhancement)
        self._sort_labels()
        self._remove_excess(min_dist_num_ratio=min_dist_num_ratio)
        self._remove_wrinkle(coeff=wrinkle_coeff)
        self._classify_labels(screening_coeff=screening_coeff, final_coeff=final_coeff)

    @property
    def _has_colors(self):
        return len(self.img.shape)==3

    def _remove_excess(self, min_dist_num_ratio=100):
        dist_num_ratio = self.get_values('distances')/self.get_values('counts')*len(self.all_labels)
        self._data['classification'][dist_num_ratio>min_dist_num_ratio] = self._get_zones('residue')

    def _fill_white(self):
        mean_color = np.mean(self.img[self.positions[:,0], self.positions[:,1]], axis=0)
        img = np.ones_like(self.img)*mean_color
        img[self.positions[:,0], self.positions[:,1]] = self.img[self.positions[:,0], self.positions[:,1]]
        return img

    def get_median_filter(self, size=5):
        if size > 0:
            if self._has_colors:
                return np.array([ndimage.median_filter(img, size=size) for img in self.img.T]).T
            else:
                return ndimage.median_filter(self.img, size=size)
        return self.img

    def get_mean_distance(self, n_neighbors=4):
        dist_lst = []
        for index in self._data['unique_labels']:
            x = self.positions[self.all_labels==index]
            tree = cKDTree(x)
            dist_lst.append(tree.query(x, k=n_neighbors+1)[0][:,1:].mean())
        return np.array(dist_lst)/np.sqrt(n_neighbors)

    def run_cluster(self, n_components=9, color_enhancement=10, use_positions=False):
        p = self.positions.T
        x = self.img[p[0], p[1]]
        if not self._has_colors:
            x = x.reshape(-1, 1)
        if use_positions:
            x = np.concatenate(
                (self.positions, x*color_enhancement),
            axis=-1)
        self.cluster = GaussianMixture(n_components=n_components).fit(x)
        self.all_labels = self.cluster.predict(x)

    def _sort_labels(self):
        unique_labels, counts = np.unique(self.all_labels, return_counts=True)
        colors = []
        for l in unique_labels:
            cond = self.all_labels==l
            colors.append(np.mean(self.img[self.positions[cond,0], self.positions[cond,1]], axis=0))
        colors = np.array(colors)
        argsort = np.argsort(self._get_brightness(colors))
        self._data['unique_labels'] = unique_labels[argsort]
        self._data['counts'] = counts[argsort]
        self._data['colors'] = colors[argsort]
        self._data['distances'] = self.get_mean_distance()
        self._data['classification'] = np.zeros(len(argsort))

    def _get_zones(self, name):
        return np.where(self._names==name)[0][0]

    def get_values(self, key, index=None):
        if index is None:
            return self._data[key]
        elif isinstance(index, str):
            return self._data[key][self._data['classification']==self._get_zones(index)]
        else:
            return self._data[key][self._data['classification']==index]

    def get_mean_colors(self):
        return 0.5*(self._data['colors'][1:]+self._data['colors'][:-1])

    def _remove_white(self, total_area, sigmas=None, white_areas=None, ridge_threshold=0.5):
        if sigmas is not None:
            img = self._get_brightness(self.img.copy())
            img_white = frangi(img, sigmas=sigmas, black_ridges=False)
            total_area[img_white>ridge_threshold] = False
        if white_areas is not None:
            for x in white_areas:
                total_area[x[:,0], x[:,1]] = False
        return np.stack(np.where(total_area), axis=-1)

    def _get_brightness(self, color):
        if self._has_colors:
            return np.mean(color, axis=-1)
        return color

    def _remove_wrinkle(self, coeff):
        colors = self._get_brightness(self.get_values('colors'))
        mean_color = np.average(colors, weights=self.get_values('counts'))
        threshold = coeff[2]/(1+np.exp(-coeff[0]*(mean_color-coeff[1])))
        cond = colors < threshold
        cond = cond*(self._data['classification']==self._get_zones('muscle'))
        self._data['classification'][cond] = self._get_zones('wrinkle')

    def _classify_labels(self, screening_coeff, final_coeff):
        for ii, coeff in enumerate([screening_coeff]+2*[final_coeff]):
            colors = self._get_brightness(self.get_values('colors'))
            if ii==0:
                mean_color = np.average(colors, weights=self.get_values('counts'))
            else:
                muscle_color = self._get_brightness(self.get_values('colors', 'muscle'))
                mean_color = np.average(muscle_color, weights=self.get_values('counts', 'muscle'))
            threshold = 255/(1+np.exp(-coeff[0]*(mean_color-coeff[1])))
            cond = colors > threshold
            cond = cond*(self._data['classification']==self._get_zones('muscle'))
            self._data['classification'][cond] = self._get_zones('scar')

    def get_area(self, key):
        if isinstance(key, str):
            cond = np.any(self.get_values('unique_labels', key)[:,None]==self.all_labels[None,:], axis=0)
        else:
            cond = np.any(np.asarray(key)[:,None]==self.all_labels[None,:], axis=0)
        return self.positions[cond]

    def get_indices(self, name):
        return self._data['unique_labels'][np.where(self._get_zones(name)==self._data['classification'])[0]]

    @staticmethod
    def _get_local_normal(x):
        v = -np.roll(x, 2, axis=0)
        v += 8*np.roll(x, 1, axis=0)
        v -= 8*np.roll(x, -1, axis=0)
        v += np.roll(x, -2, axis=0)
        v /= np.linalg.norm(v, axis=-1)[:,None]
        v = np.einsum('ij,nj->ni', [[0, 1], [-1, 0]], v)
        return v

    def get_boolean_area(self, key):
        img = np.zeros(self.img.shape[:2])
        x = self.get_area(key)
        if len(x)==0:
            return
        img[x[:,0], x[:,1]] = 1
        return img

    def get_fibrous_tissue(
        self,
        total_perimeter,
        sigma_image=5,
        sigma_reliability=3,
        max_dist=100,
        total_reliability_ratio=[0.1, -0.1],
        tissue_limit_area=[60, 20],
        total_density_range=[0.25, 0.75],
        tissue_ratio=[0, -0.2],
    ):
        img_scar = self.get_boolean_area('scar')
        img_muscle = self.get_boolean_area('muscle')
        if img_scar is None or img_muscle is None:
            return
        img_diff = ndimage.gaussian_filter(img_muscle-img_scar, sigma=sigma_image)
        img_total = ndimage.gaussian_filter(img_muscle+img_scar, sigma=sigma_image)
        v = self._get_local_normal(total_perimeter)
        dist_range = np.arange(max_dist)
        all_positions = np.rint(
            np.einsum('k,nj->knj', dist_range, v)+total_perimeter
        ).astype(int)
        tree_peri = cKDTree(total_perimeter)
        tree_scar = cKDTree(self.get_area('scar'))
        mat = tree_peri.sparse_distance_matrix(tree_scar, max_distance=max_dist-1)
        coo = mat.tocoo()
        data = coo.data
        data[data>0] += 0.5
        data = data.astype(int)
        all_diff = ndimage.gaussian_filter1d(
            img_diff[all_positions.T[0], all_positions.T[1]],
            sigma=sigma_reliability,
            axis=-1
        )
        all_diff[:,int(2*sigma_image):] = np.maximum.accumulate(all_diff[:,int(2*sigma_image):], axis=-1)
        all_total = ndimage.gaussian_filter1d(
            img_total[all_positions.T[0], all_positions.T[1]],
            sigma=sigma_reliability,
            axis=-1
        )
        total_reliability = get_slope(all_diff, total_reliability_ratio)
        total_reliability *= get_slope(dist_range, tissue_limit_area)*get_slope(all_total, total_density_range)
        local_reliability = get_slope(all_diff, tissue_ratio)*get_slope(all_total, total_density_range)
        local_reliability *= total_reliability.max(axis=-1)[:,None]
        prob = np.zeros(len(self.get_area('scar')))
        np.maximum.at(prob, np.array(coo.col), local_reliability[coo.row][np.arange(len(data)), data])
        return prob

    def get_fibrous_tissue_old(
        self,
        total_perimeter,
        eps=3,
        rerun=False,
        min_cluster_size=20,
        r_max=0.95, 
        phi_mesh_size=180,
        phi_sigma = np.pi*0.01,
        minimum_angle=10/180*np.pi,
        max_distance_from_edge=3,
    ):
        x_p = total_perimeter.copy()
        x_s = self.get_area('scar').astype(float)
        if self.fibrous_tissue_cluster is None or rerun:
            self.fibrous_tissue_cluster  = DBSCAN(eps=3).fit(x_s)
        labels, counts = np.unique(self.fibrous_tissue_cluster.labels_, return_counts=True)
        tree = cKDTree(x_p)
        all_dist = (max_distance_from_edge+1)*np.ones(len(x_s))
        for l, c in zip(labels, counts):
            if c>=min_cluster_size and l!=-1:
                cc = self.fibrous_tissue_cluster.labels_==l
                all_dist[cc] = tree.query(x_s[cc])[0].min()
        cond = np.any(
            self.fibrous_tissue_cluster.labels_[:,None]==labels[counts>min_cluster_size][None,:],
            axis=-1
        )
        cond *= self.fibrous_tissue_cluster.labels_!=-1
        x_s = x_s[cond]
        all_dist = all_dist[cond]
        x_c = np.mean(x_p, axis=0)
        x_s -= x_c
        x_p -= x_c
        rs = np.stack([np.arctan2(x_s[:,1], x_s[:,0]), np.linalg.norm(x_s, axis=-1)], axis=-1)
        rp = np.stack([np.arctan2(x_p[:,1], x_p[:,0]), np.linalg.norm(x_p, axis=-1)], axis=-1)
        r = rs[:,1]/rp[np.absolute(rs[:,None,0]-rp[None,:,0]).argmin(axis=1), 1]
        all_dist = all_dist[r>r_max]
        phi = rs[r>r_max, 0]
        x_s = x_s[r>r_max]
        r = r[r>r_max]
        phi_space = np.linspace(-np.pi, np.pi, phi_mesh_size)
        dphi = phi_space[:,None]-phi[None,:]
        dphi -= np.rint(dphi/np.pi)*2*np.pi
        gauss = np.exp(-dphi**2/(2*phi_sigma **2))
        r_ave = np.sum((1-r)*gauss, axis=-1)/(np.sum(gauss, axis=-1)+1.0e-8)
        d_angle = 2*np.pi/phi_mesh_size
        n = np.max([np.rint(minimum_angle/d_angle).astype(int), 1])
        r_expect = (1-r_max)/2
        arr = r_ave/r_expect<1-1/np.sqrt(3)
        indices = np.arange(len(arr))[arr]
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1.5, linkage='single'
        ).fit(indices.reshape(-1, 1))
        all_labels = cluster.labels_.copy()
        if arr[-1]:
            all_labels[all_labels==all_labels[-1]] = all_labels[0]
        labels, counts = np.unique(all_labels, return_counts=True)
        labels = labels[counts>n]
        indices = indices[np.any(all_labels[:,None]==labels[None,:], axis=-1)]
        x_f = x_s[np.any(dphi.argmin(axis=0)[:,None]==indices[None,:], axis=-1)*(all_dist<max_distance_from_edge)]
        x_f += x_c
        return x_f

