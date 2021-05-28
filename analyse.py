import numpy as np
from process import ProcessImage
from tissue import Tissue
from area import Area
from left_ventricle import LeftVentricle
import json

class Analyse:
    def __init__(self, file_name, parameters=None):
        if parameters is None:
            with open('default_parameters.txt', 'r') as f:
                parameters = json.load(f)
        self.parameters = parameters
        self.image = ProcessImage(
            ref_job=self, file_name=file_name, parameters=self.parameters['processing']
        )
        self.initialize()

    def initialize(self):
        self._heart = None
        self._right = None
        self._left = None
        self._right = None
        self._tissue = None
        self._left_ventricle = None

    @property
    def heart(self):
        if self._heart is None:
            self._heart = self.image.get_data('heart')
        return self._heart

    @property
    def left(self):
        if self._left is None:
            if self._heart is None:
                _ = self.heart
            self._left = self.image.get_data('left')
        return self._left

    @property
    def right(self):
        if self._right is None:
            if self._left is None:
                _ = self.left
            self._right = self.image.get_data('right')
        return self._right

    @property
    def tissue(self):
        if self._tissue is None:
            self._tissue = Tissue(
                img=self.get_image(),
                total_area=self.image.total_area,
                white_areas=self.image.white_area,
                **self.parameters['tissue']
            )
        return self._tissue

    @property
    def left_ventricle(self):
        if self._left_ventricle is None:
            if len(self.left)==0 or len(self.right)==0:
                return None
            self._left_ventricle = LeftVentricle(self)
        return self._left_ventricle

    def get_base_color(self, mean=True):
        return self.image.get_base_color(mean=mean)

    def get_image(self, mean=False):
        return self.image.get_image(mean=mean)

    def collect_output(self):
        rr = self.image.resolution
        r = np.sqrt(rr)
        output = {
            'file_name': self.image.file_name.split('/')[-1],
            'resolution': rr,
            'white_color_threshold': self.image.white_color_threshold,
            'H_area_elastic_net': self.heart.perimeter.volume*rr,
        }
        for tag, cl in zip(['H_', 'LL_', 'RL_'], [self.heart, self.left, self.right]):
            output[tag+'area_point_counts'] = cl.get_volume(mode='points')*rr
            output[tag+'area_principal_component_analysis'] = cl.get_volume(mode='pca')*rr
            output[tag+'area_delaunay_tesselation'] = cl.get_volume(mode='delaunay')*rr
            output[tag+'area_convex_hull'] = cl.get_volume(mode='hull')*rr
            output[tag+'surface_delaunay_tesselation'] = len(cl.delaunay.x)*rr
        output['RL_area_adapted_principal_component_analysis'] = self.right.trace_pca(
            self.heart.get_center()
        ).volume*rr
        output['LL_RL_contact'] = self.left.count_neighbors(self.right.points, r=2)*rr
        output['LL_contact_with_perimeter'] = self.left.count_neighbors(
            self.heart.perimeter.x, r=2
        )*r
        output['RL_contact_with_perimeter'] = self.right.count_neighbors(
            self.heart.perimeter.x, r=2
        )*r
        if self.left_ventricle is not None:
            output['RV_area_wo_cracks'] = np.sum(
                self.left_ventricle.separate_points(self.tissue.positions)
            )*rr
            output['RV_area'] = np.sum(self.left_ventricle.separate_points(self.heart.points))*rr
            output['distance_left_to_right_lumen'] = np.linalg.norm(
                self.left_ventricle.left_to_right
            )*r
        else:
            output['RV_area_wo_cracks'] = 0
            output['RV_area'] = 0
            output['distance_left_to_right_lumen'] = 0
        output['area_total_white_clusters'] = len(self.image.white_area.x)*rr
        if len(self.right) > 0:
            output['curvature_crossing'] = self.heart.perimeter.get_crossing_curvature(
                self.heart.get_center(), self.right.get_center()
            )-2*np.pi
        else:
            output['curvature_crossing'] = 0
        output['curvature_ptp'] = self.heart.perimeter.get_curvature(sigma=5).ptp()
        output['curvature_std'] = np.std(self.heart.perimeter.get_curvature(sigma=5))
        output['scar_total'] = np.sum(self.tissue.get_distance()<0)*rr
        output['scar_err'] = self.tissue.get_error()*rr
        return output


