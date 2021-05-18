from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.mixture import GaussianMixture
from learn import Learn
from scipy.spatial import cKDTree
from collections import defaultdict
import random
import pandas as pd
import matplotlib.pylab as plt

class CalibrateColors(Learn):
    def __init__(self, **arg):
        self.data = defaultdict(list)
        self.label_lst = []
        self.position_lst = []
        self.img_lst = []
        self.job_index = 0
        self.current_index = 0
        self.check_scar = False
        self.check_wrinkle = False
        self.fit_scar = SGDClassifier(warm_start=True)
        self.fit_wrinkle = SGDClassifier(warm_start=True)
        self.choices = {
            'muscle': self.muscle,
            'scar': self.scar,
            'wrinkle': self.wrinkle,
            'excess': self.excess,
        }
        super().__init__(**arg)

    def load_data(self, data):
        for k,v in data.items():
            if k=='job_index' and len(self.data['job_index'])>0:
                self.data[k].extend([d+np.max(self.data['job_index'])+1 for d in v])
            else:
                self.data[k].extend(v)

    def append(self, color):
        self.position_lst.append(color.ref_job.tissue.positions)
        self.img_lst.append(color.ref_job.get_image())
        self.label_lst.append(color.all_labels)
        if len(self.data['job_index'])==0:
            max_index = -1
        else:
            max_index = np.max(self.data['job_index'])
        self.data['job_index'].extend(len(color._data['unique_labels'])*[max_index+1])
        for k,v in color._data.items():
            self.data[k].extend(v)
        self.data['tag'].extend(len(color._data['unique_labels'])*[-1])
        self.proceed()

    def fit(self, key):
#         indices = self.get_indices([0, 1, 2])
        if key=='scar':
            clf = self.fit_scar
            indices = self.get_indices([0, 1])
            base_color = self.get_base_colors(muscle=True)
        if key=='wrinkle':
            clf = self.fit_wrinkle
            indices = self.get_indices([1, 2])
            base_color = self.get_base_colors(muscle=False)
        c = np.concatenate([
            base_color[indices],
            np.array(self.data['colors'])[indices]
        ], axis=-1)
        l = np.array(self.data['tag'])[indices]
        w = self.get_weights()[indices]
        clf.fit(c, l, sample_weight=w)

    @property
    def n_jobs(self):
        return np.max(self.data['job_index'])

    @property
    def n_labels(self):
        return np.sum(self.data['job_index']==self.job_index)

    def set_data(self, value):
        self.data['tag'][self.current_index] = value
        if self.check_scar:
            if 0 in self.data['tag'] and 1 in self.data['tag']:
                self.fit('scar')
            self.check_scar = False
        if self.check_wrinkle:
            if 2 in self.data['tag'] and 1 in self.data['tag']:
                self.fit('wrinkle')
            self.check_wrinkle = False
        self.proceed()

    def _get_individual_distances(self, clf, muscle, max_dist=255**2):
        if not hasattr(clf, 'coef_'):
            return len(self.data['tag'])*[max_dist]
        values = np.sum(clf.coef_*np.concatenate([
            self.get_base_colors(muscle=muscle),
            np.array(self.data['colors'])
        ], axis=-1), axis=-1)/np.linalg.norm(clf.coef_)+clf.intercept_
        return np.absolute(values)

    def get_distances(self):
        return np.stack((
            self._get_individual_distances(self.fit_scar, muscle=True),
            self._get_individual_distances(self.fit_wrinkle, muscle=False)
        ), axis=-1).min(axis=-1)

    def get_weights(self):
        counts = np.array([
            np.sum(np.asarray(self.data['counts'])[np.asarray(self.data['job_index'])==index])
            for index in np.unique(self.data['job_index'])
        ])
        return np.asarray(self.data['counts'])/counts[np.array(self.data['job_index'])]

    def proceed(self):
        permitted_indices = np.asarray(self.data['tag'])==-1
        weights = self.get_weights()/self.get_distances()
        self.current_index = np.squeeze(random.choices(
            np.arange(len(self.data['tag']))[permitted_indices],
            weights=weights[permitted_indices]
        ))
        self.job_index = self.data['job_index'][self.current_index]
        self.current_img = self.img_lst[self.job_index]
        self.current_positions = np.asarray(self.position_lst[self.job_index])
        self.current_labels = np.asarray(self.label_lst[self.job_index])

    def get_indices(self, tags):
        tags = np.atleast_1d(tags)
        return np.any(np.array(self.data['tag'])[:,None]==tags[None,:], axis=-1)

    def get_base_colors(self, muscle=True):
        mean_colors = []
        for index in np.unique(self.data['job_index']):
            indices = np.asarray(self.data['job_index'])==index
            if muscle:
                indices *= (np.asarray(self.data['tag'])==0) | (np.asarray(self.data['tag'])==-1)
            mean_colors.append(np.mean(np.asarray(self.data['colors'])[indices], axis=0))
        return np.array(mean_colors)[np.array(self.data['job_index'])]

    def save_data(self, file_name):
        import json
        data_to_store = self.data.copy()
        for k,v in data_to_store.items():
            data_to_store[k] = np.array(v).tolist()
        with open(file_name, 'w') as fp:
            json.dump(data_to_store, fp)

    def scar(self):
        self.check_scar = True
        self.set_data(0)

    def muscle(self):
        self.check_scar = True
        self.check_wrinkle = True
        self.set_data(1)

    def wrinkle(self):
        self.check_wrinkle = True
        self.set_data(2)

    def excess(self):
        self.set_data(3)

    def plot(self):
        _, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax[0].imshow(np.mean(self.current_img, axis=-1), cmap='Greys')
        l = self.data['unique_labels'][self.current_index]
        ax[0].scatter(
            self.current_positions[self.current_labels==l, 1],
            self.current_positions[self.current_labels==l, 0],
            marker='.', s=0.1, color='red'
        )
        ax[1].imshow(self.current_img)

class Colors:
    def __init__(
        self,
        ref_job,
        n_components=20,
        color_enhancement=10,
    ):
        self._data = {}
        self.ref_job = ref_job
        self.run_cluster(n_components=n_components, color_enhancement=color_enhancement)
        self._sort_labels()

    def get_colors(self):
        p = self.ref_job.tissue.positions.T
        return self.ref_job.tissue.img[p[0], p[1]]

    def get_mean_distance(self, n_neighbors=4):
        dist_lst = []
        for index in self._data['unique_labels']:
            x = self.ref_job.tissue.positions[self.all_labels==index]
            tree = cKDTree(x)
            dist_lst.append(tree.query(x, k=n_neighbors+1)[0][:,1:].mean())
        return np.array(dist_lst)/np.sqrt(n_neighbors)
    
    def run_cluster(self, n_components=20, color_enhancement=10, use_positions=False):
        x = self.get_colors()
        if not self._has_colors:
            x = x.reshape(-1, 1)
        if use_positions:
            x = np.concatenate(
                (self.ref_job.tissue.positions, x*color_enhancement),
            axis=-1)
        self.cluster = GaussianMixture(n_components=n_components).fit(x)
        self.all_labels = self.cluster.predict(x)

    def _sort_labels(self):
        unique_labels, counts = np.unique(self.all_labels, return_counts=True)
        colors = []
        for l in unique_labels:
            cond = self.all_labels==l
            colors.append(np.mean(
                self.ref_job.tissue.img[
                    self.ref_job.tissue.positions[cond,0], self.ref_job.tissue.positions[cond,1]
                ], axis=0
            ))
        colors = np.array(colors)
        argsort = np.argsort(self._get_brightness(colors))
        self._data['unique_labels'] = unique_labels[argsort]
        self._data['counts'] = counts[argsort]
        self._data['colors'] = colors[argsort]
        self._data['distances'] = self.get_mean_distance()

    @property
    def _has_colors(self):
        return len(self.ref_job.tissue.img.shape)==3

    def _get_brightness(self, color):
        if self._has_colors:
            return np.mean(color, axis=-1)
        return color

