import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from clartbeat.tools import get_slope
from skimage import filters

class Tissue:
    def __init__(
        self,
        ref_job,
        parameters,
    ):
        self._data = {}
        self._names = np.array(['muscle', 'scar', 'fibrous_tissue', 'wrinkle', 'residue'])
        self.all_labels = None
        self.fibrous_tissue_cluster = None
        self.ref_job = ref_job
        self.scar_coeff = np.array(parameters['scar_coeff'])
        self.scar_coeff /= np.linalg.norm(parameters['scar_coeff'][:-1])
        self._img = None
        self._positions = None
        self.parameters = parameters
        self._classify_labels('wrinkle')
        self._classify_labels('scar')

    @property
    def _img_reverse_color(self, v_max=255):
        img_bw = self.ref_job.image.get_image(mean=True)
        img_bw -= self.ref_job.image.get_base_color()
        img_bw = np.absolute(img_bw)
        img_bw *= v_max/img_bw.max()
        return img_bw

    def _get_frangi_ridges(
        self, reverse_color=True, sigmas=[2, 3, 4], black_ridges=False, threshold=0.85
    ):
        if reverse_color:
            img_bw = self._img_reverse_color
        else:
            img_bw = self.ref_job.image.get_image(mean=True)
        img_white = filters.frangi(img_bw, sigmas=sigmas, black_ridges=black_ridges)
        cond = img_white > threshold
        cond *= ~self.ref_job.image.non_white_area
        return np.stack(np.where(cond), axis=-1)

    @property
    def positions(self):
        if self._positions is None:
            total_area = self.ref_job.image.total_area.copy()
            total_area[tuple(self.ref_job.image.white_area.x.T)] = False
            x = self._get_frangi_ridges(**self.parameters['frangi_ridges'])
            total_area[tuple(x.T)] = False
            self._positions = np.stack(np.where(total_area), axis=-1)
        return self._positions

    @property
    def img(self):
        if self._img is None:
            self._img = self.ref_job.get_image()
            self._img = self.get_median_filter(size=self.parameters['filter_size'])
        return self._img

    def get_median_filter(self, size=6):
        img = self._img.copy()
        if size > 0:
            tree = cKDTree(self.positions)
            distances, indices = tree.query(
                self.positions,
                k=np.rint(np.pi*(1+size)**2).astype(int),
                distance_upper_bound=size
            )
            positions = self.positions[indices[distances<np.inf]]
            rgb = np.empty(distances.shape+(3,))
            rgb[:] = np.nan
            rgb[distances<np.inf] = img[positions[:,0], positions[:,1]]
            rgb = np.nanmedian(rgb, axis=1)
            img[tuple(self.positions.T)] = rgb
        return img

    def _get_zones(self, name):
        return np.where(self._names==name)[0][0]

    def get_distance(self):
        colors = self.img[tuple(self.positions.T)]
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
        img[tuple(x.T)] = 1
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

