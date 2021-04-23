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
            indices_left = self.image.get_index('left', **self.parameters['left'])
            self.left = self.image.get_data(
                'white', indices_left
            )
            if self.left is not None:
                self.right = self.image.get_data(
                    'white', self.image.get_index(
                        'right',
                        **self.parameters['right'],
                        indices_to_avoid=np.array(indices_left)
                    )
                )
            else:
                self.right = None
        if elem_dict[initialize] >= elem_dict['scar']:
            self.tissue = Tissue(
                img=self.get_image(),
                total_area=self.image.get_area('heart'),
                white_areas=self.image.cluster['white'],
                **self.parameters['tissue']
            )
        if elem_dict[initialize] >= elem_dict['ventricle'] and self.left is not None:
            self.left_ventricle = LeftVentricle(self)
        else:
            self.left_ventricle = None

    def get_image(self, mean=False):
        return self.image.get_image(mean=mean)

    def collect_output(self, detail_level=0):
        global_info = {
            'resolution': self.image.resolution,
            'white_color_threshold': self.image.white_color_threshold, 
            'excluded_objects': np.sum(self.image.non_white_points*(~self.image.total_area)),
        }
        # total_area = {
        #     'area': {
        #         'convex_hull':
        #         'delaunay_tesselation':
        #         'principal_component_analysis':
        #         'canny_edge_detection':
        #         'elastic_net':
        #     }
        # }

