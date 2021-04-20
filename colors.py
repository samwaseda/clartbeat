import numpy as np
from sklearn.mixture import GaussianMixture
from learn import Learn
from scipy.spatial import cKDTree
from collections import defaultdict
import pandas as pd

class CalibrateColors(Learn):
    def __init__(self, colors_lst, **arg):
        self.colors_lst = colors_lst
        self.data = defaultdict(list)
        self.initialize()
        self.job_index = 0
        self.data_index = 0
        self.color = self.colors_lst[self.job_index]
        self.choices = {
            'muscle': self.muscle,
            'scar': self.scar,
            'excess': self.excess,
            'wrinkle': self.wrinkle,
        }
        self.proceed()
        super().__init__(**arg)

    @property
    def n_jobs(self):
        return len(self.colors_lst)

    @property
    def n_labels(self):
        return len(self.colors_lst[self.job_index]._data['unique_labels'])

    def initialize(self):
        for ii, color in enumerate(self.colors_lst):
            self.data['job_index'].extend([len(color._data['unique_labels'])*[ii]])
            for k,v in color._data.items():
                self.data[k].extend(v)
            self.data['tag'].extend(len(color._data['unique_labels'])*[-1])
        for k,v in self.data.items():
            if 'color' not in k:
                self.data[k] = np.array(v).flatten()
            else:
                self.data[k] = np.array(v).reshape(-1, 3)

    def set_data(self, value):
        index = np.arange(len(self.data['tag']))[self.job_index==self.data['job_index']][self.data_index]
        self.data['tag'][index] = value
        self.proceed()

    def proceed(self):
        self.data_index += 1
        if self.data_index >= self.n_labels:
            self.data_index = 0
            self.job_index += 1
            if self.job_index >= self.n_jobs:
                self.count = np.inf
            else:
                self.color = self.colors_lst[self.job_index]

    def save_data(self, file_name):
        import json
        data_to_store = self.data.copy()
        for k,v in data_to_store.items():
            data_to_store[k] = v.tolist()
        with open(file_name, 'w') as fp:
            json.dump(data_to_store, fp)

    def scar(self):
        self.set_data(0)

    def muscle(self):
        self.set_data(1)

    def excess(self):
        self.set_data(2)

    def wrinkle(self):
        self.set_data(3)

    def plot(self):
        _, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax[0].imshow(self.color.ref_job.get_image(True), cmap='Greys')
        l = self.color._data['unique_labels'][self.data_index]
        ax[0].scatter(
            self.color.ref_job.tissue.positions[self.color.all_labels==l, 1],
            self.color.ref_job.tissue.positions[self.color.all_labels==l, 0],
            marker='.', s=0.1, color='red'
        )
        ax[1].imshow(self.color.ref_job.get_image())

class Colors:
    def __init__(
        self,
        ref_job,
        n_components=30,
        color_enhancement=10,
    ):
        self._data = {}
        self.ref_job = ref_job
        self.run_cluster(n_components=n_components, color_enhancement=color_enhancement)
        self._sort_labels()

    def get_mean_distance(self, n_neighbors=4):
        dist_lst = []
        for index in self._data['unique_labels']:
            x = self.ref_job.tissue.positions[self.all_labels==index]
            tree = cKDTree(x)
            dist_lst.append(tree.query(x, k=n_neighbors+1)[0][:,1:].mean())
        return np.array(dist_lst)/np.sqrt(n_neighbors)

    def run_cluster(self, n_components=30, color_enhancement=10, use_positions=False):
        x = self.get_colors()
        if not self._has_colors:
            x = x.reshape(-1, 1)
        if use_positions:
            x = np.concatenate(
                (self.ref_job.tissue.positions, x*color_enhancement),
            axis=-1)
        self.cluster = GaussianMixture(n_components=n_components).fit(x)
        self.all_labels = self.cluster.predict(x)

    def get_colors(self):
        p = self.ref_job.tissue.positions.T
        return self.ref_job.tissue.img[p[0], p[1]]

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
