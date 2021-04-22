import numpy as np
from process import ProcessImage
from tissue import Tissue
from area import Area
from left_ventricle import LeftVentricle
import json

class Analyse:
    def __init__(self, file_name, parameters=None, initialize='all'):
        if parameters is None:
            with open('default_parameters.txt', 'r') as f:
                parameters = json.load(f)
        self.parameters = parameters
        self._data = {}
        self.image = ProcessImage(file_name=file_name, parameters=self.parameters['processing'])
        self.initialize(initialize=initialize)

    def get_base_color(self, mean=True):
        return self.image.get_base_color(mean=mean)

    def get_original(self, reduction=None):
        return self.image.get_original(reduction=reduction)

    def initialize(self, initialize='all'):
        elements = np.array(['none', 'heart', 'white', 'scar', 'ventricle', 'all'])
        if initialize not in elements:
            raise ValueError('invalid initialization element: {}'.format(initialize))
        elem_dict = {k:v for v,k in enumerate(elements)}
        if elem_dict[initialize] >= elem_dict['heart']:
            self.image.run_cluster('heart', **self.parameters['heart'])
            self.heart = self.image.get_data('heart')
        if elem_dict[initialize] >= elem_dict['white']:
            self.image.run_cluster('white', **self.parameters['white'])
            self.left = self.image.get_data('white', self.image.get_index('left'))
            self.right = self.image.get_data('white', self.image.get_index('right'))
        if elem_dict[initialize] >= elem_dict['scar']:
            self.tissue = Tissue(
                img=self.get_image(),
                total_area=self.image.get_area('heart'),
                white_areas=self.image.cluster['white'],
                **self.parameters['tissue']
            )
        if elem_dict[initialize] >= elem_dict['ventricle']:
            self.left_ventricle = LeftVentricle(self)

    @property
    def data(self):
        return self._data

    def get_edges(self, max_dist=10):
        edges = []
        ep = np.array([self.er[1], -self.er[0]])
        for i, t in enumerate([self.right, self.left]):
            x = t.points.copy().astype(float)
            x -= self.heart.get_center()
            y = np.einsum('ij,nj->ni', np.stack((self.er, ep)), x)
            cond = np.absolute(y[:,1])<max_dist
            if np.sum(cond)==0:
                return np.zeros(2)
            if i==0:
                edges.append(y[cond, 0].min())
            else:
                edges.append(y[cond, 0].max())
        return edges

    def get_image(self, mean=False):
        return self.image.get_image(mean=mean)

