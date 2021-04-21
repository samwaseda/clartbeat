import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.filters import frangi
from tools import get_slope

class Tissue:
    def __init__(
        self,
        img,
        total_area,
        white_areas=None,
        scar_coeff=np.array([
            2.69008370e-02,
            -1.16449179e+02,
            -5.88774824e+01,
            6.50569742e+01, 
            6.02359123e+01, 
            7.63830882e+01,
            5.55613592
        ]),
        sigmas=[4],
        color_enhancement=10,
        filter_size=6,
        fill_white=True,
    ):
        self._data = {}
        self._names = np.array(['muscle', 'scar', 'fibrous_tissue', 'wrinkle', 'residue'])
        self.all_labels = None
        self.fibrous_tissue_cluster = None
        self.img = img.copy()
        self.positions = self._remove_white(total_area, sigmas, white_areas)
        if fill_white:
            self.img = self._fill_white()
        self.img = self.get_median_filter(size=filter_size)
        self._classify_labels(coeff=np.array(scar_coeff))

    @property
    def _has_colors(self):
        return len(self.img.shape)==3

    def _get_brightness(self, color):
        if self._has_colors:
            return np.mean(color, axis=-1)
        return color

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

    def _remove_white(self, total_area, sigmas=None, white_areas=None, ridge_threshold=0.5):
        if sigmas is not None:
            img = self._get_brightness(self.img.copy())
            img_white = frangi(img, sigmas=sigmas, black_ridges=False)
            total_area[img_white>ridge_threshold] = False
        if white_areas is not None:
            for x in white_areas:
                total_area[x[:,0], x[:,1]] = False
        return np.stack(np.where(total_area), axis=-1)

    def _get_zones(self, name):
        return np.where(self._names==name)[0][0]

    def _classify_labels(self, coeff):
        colors = self.img[self.positions[:,0], self.positions[:,1]]
        base_color = np.mean(colors, axis=0)
        values = np.sum(colors*coeff[:3], axis=-1)
        values += np.sum(base_color*coeff[3:6])+coeff[-1]
        self.all_labels = np.tile(self._get_zones('muscle'), len(self.positions))
        self.all_labels[values<0] = self._get_zones('scar')

    def get_area(self, key):
        if isinstance(key, str):
            index = self._get_zones(key)
        return self.positions[self.all_labels==index]

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
        tree_peri = cKDTree(total_perimeter)
        tree_scar = cKDTree(self.get_area('scar'))
        mat = tree_peri.sparse_distance_matrix(tree_scar, max_distance=max_dist-1)
        coo = mat.tocoo()
        data = coo.data
        data[data>0] += 0.5
        data = data.astype(int)
        np.maximum.at(prob, np.array(coo.col), local_reliability[coo.row][np.arange(len(data)), data])
        return prob

