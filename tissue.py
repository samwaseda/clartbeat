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
            90.28205419,
            76.98283148,
            126.75059511,
            106.03962742,
            -305.73676848,
            -142.75818544,
            8.75126572
        ]),
        wrinkle_coeff=np.array([
            105.81948491,
            -20.37814521,
            212.99928111,
            -425.62869598,
            24.75423721,
            -163.16110443,
            -26.80759178
        ]),
        sigmas=[4],
        color_enhancement=10,
        filter_size=6,
        fill_white=False,
    ):
        self._data = {}
        self._names = np.array(['muscle', 'scar', 'fibrous_tissue', 'wrinkle', 'residue'])
        self.all_labels = None
        self.fibrous_tissue_cluster = None
        self.img = img.copy()
        self.scar_coeff = np.array(scar_coeff)/np.linalg.norm(scar_coeff[:-1])
        self.wrinkle_coeff = np.array(wrinkle_coeff)/np.linalg.norm(wrinkle_coeff[:-1])
        self.positions = self._remove_white(total_area.copy(), sigmas, white_areas)
        if fill_white:
            self.img = self._fill_white()
        self.img = self.get_median_filter(size=filter_size)
        self._classify_labels('wrinkle')
        self._classify_labels('scar')

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
        img[self.positions[:,0], self.positions[:,1]] = self.img[
            self.positions[:,0], self.positions[:,1]
        ]
        return img

    def get_median_filter(self, size=6):
        if size > 0:
            tree = cKDTree(self.positions)
            distances, indices = tree.query(
                self.positions,
                k=np.rint(np.pi*(1+size)**2).astype(int),
                distance_upper_bound=size
            )
            positions = self.positions[indices[distances<np.inf]]
            if self._has_colors:
                rgb = np.empty(distances.shape+(3,))
            else:
                rgb = np.empty(distances.shape)
            rgb[:] = np.nan
            rgb[distances<np.inf] = self.img[positions[:,0], positions[:,1]]
            rgb = np.nanmedian(rgb, axis=1)
            self.img[self.positions[:,0], self.positions[:,1]] = rgb
        return self.img

    def _remove_white(self, total_area, sigmas=None, white_areas=None, ridge_threshold=0.5):
        if sigmas is not None:
            img = self._get_brightness(self.img.copy())
            img_white = frangi(img, sigmas=sigmas, black_ridges=False)
            total_area[img_white>ridge_threshold] = False
        if white_areas is not None:
            total_area[white_areas.x[:,0], white_areas.x[:,1]] = False
        return np.stack(np.where(total_area), axis=-1)

    def _get_zones(self, name):
        return np.where(self._names==name)[0][0]

    def get_distance(self):
        colors = self.img[self.positions[:,0], self.positions[:,1]]
        base_color = np.mean(colors, axis=0)
        self.all_labels = np.tile(self._get_zones('muscle'), len(self.positions))
        values = np.sum(colors*self.scar_coeff[3:6], axis=-1)
        values += np.sum(base_color*self.scar_coeff[:3])+self.scar_coeff[-1]
        return values

    def get_error(self, error_distance=10):
        dist = self.get_distance()
        dist = get_slope(dist, np.array([1, -1])*error_distance)
        return np.sum(np.absolute(dist-np.rint(dist)))

    def _classify_labels(self, key):
        if key=='wrinkle':
            values = self.get_distance()
        elif key=='scar':
            values = self.get_distance()
        self.all_labels[values>=0] = self._get_zones('muscle')
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
        sigma_neighborhood=4,
        max_dist=100,
        total_reliability_ratio=[-0.1, 0.1],
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
        tree_peri = cKDTree(all_positions.reshape(-1, 2))
        scar_area = self.get_area('scar')
        tree_scar = cKDTree(scar_area)
        coo = tree_scar.sparse_distance_matrix(tree_peri, max_distance=3*sigma_neighborhood).tocoo()
        gauss = np.exp(-0.5*coo.data**2/sigma_neighborhood**2)
        p_denom = np.zeros(len(scar_area))
        p_num = np.zeros(len(scar_area))
        np.add.at(p_denom, coo.row, local_reliability.T.flatten()[coo.col]*gauss[coo.col])
        np.add.at(p_num, coo.row, gauss[coo.col])
        prob = p_denom/(p_num+1*(p_num<1.0e-8))
        return prob

