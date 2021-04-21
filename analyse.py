import numpy as np
from process import ProcessImage
from tissue import Tissue
from area import Area
from scipy.optimize import minimize
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
            self.run_left_ventricle()

    @property
    def data(self):
        return self._data

    def get_edges(self, max_dist=10):
        edges = []
        ep = np.array([self.er[1], -self.er[0]])
        for i, t in enumerate([self.right, self.left]):
            x = t.points.copy().astype(float)
            x -= self.left.get_center()
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

    @property
    def er(self):
        if 'vec_H_RV' not in self.data.keys():
            self.data['vec_H_RV'] = self.right.get_center()-self.heart.get_center()
        return self.data['vec_H_RV']/np.linalg.norm(self.data['vec_H_RV'])

    def get_optimum_radius(self):
        center = self.heart.get_center()
        def error_f(r, er=self.er, center=center, points=self.right.points):
            return np.sum(np.absolute(np.linalg.norm(points-(r[0]*er+center), axis=-1)-r[1]))
        opt = minimize(error_f, [1, 100], method='Nelder-Mead')
        return opt

    def run_left_ventricle(self, reduced=True, use_pca=True, k_ratio=1):
        points = self.heart.get_points(reduced=True)
        if use_pca:
            a = np.linalg.norm(self.heart.get_principal_vectors()[0])
        else:
            a = np.max(np.einsum('i,ni->n', self.er, points-self.heart.get_center()))
        dx = a-np.linalg.norm(self.heart.get_center()-self.right.get_center(ref_point=self.heart.get_center()))
        x_t = a-2*dx
        edges = self.get_edges()
        x_m = np.absolute(np.mean(edges))
        if reduced and x_m!=0:
            l = np.absolute((edges[1]-edges[0])/np.mean(edges))
            x_opt = (x_m+k_ratio*l*x_t)/(1+k_ratio*l)*self.er+self.heart.get_center()
        else:
            x_opt = np.max([x_t, x_m])*self.er+self.heart.get_center()
        opt_r = self.get_optimum_radius().x[0]
        if opt_r < a:
            opt_r = a
        max_r = 2*np.linalg.norm(self.heart.get_center()-self.right.get_center(ref_point=self.heart.get_center()))
        if opt_r > max_r or not reduced:
            opt_r = max_r
        x_c = x_opt-opt_r*self.er
        x = points[np.linalg.norm(points-x_c, axis=-1)<opt_r]
        left_ventricle = Area(x, resolution=self.image.resolution)
        left_ventricle.center = x_c
        left_ventricle.radius = opt_r
        self.ventricle = left_ventricle

